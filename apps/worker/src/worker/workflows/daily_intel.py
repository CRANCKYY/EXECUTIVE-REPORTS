"""Daily executive briefing workflow for reinsurance intelligence.

Every scheduled run:
1. Reads recent Gmail inbox traffic (Gmail REST API with a refresh token from secrets).
2. Extracts executive actions (requests, commitments, deadlines) and tracks them over time.
3. Researches reinsurance market developments for Kenya, Tanzania and Mauritius with web search.
4. Compares current regulatory frameworks (capital requirements, licensing, supervision)
   against the stored country baseline and reports only what changed.
5. Produces strategic analysis, business opportunities and a decision matrix.
6. Emails the briefing to the configured recipient and persists the new state in a
   Mistral document library so the next run only reports deltas.

Required runtime secrets (workspace secrets bound to the deployment environment):
    GMAIL_CLIENT_ID, GMAIL_CLIENT_SECRET, GMAIL_REFRESH_TOKEN, RECIPIENT_EMAIL
"""

import base64
import json
import os
from datetime import datetime, timedelta
from email.message import EmailMessage
from typing import Any, Literal

import httpx
from mistralai import workflows
from mistralai.client.models import WebSearchTool
from mistralai.workflows.plugins.mistralai import (
    Agent,
    ChatCompletionRequest,
    LocalSession,
    Runner,
    UserMessage,
    chat_parse_to_model,
)
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class DailyIntelInput(BaseModel):
    dry_run: bool = False
    lookback_hours: int = 36
    max_emails: int = 25


class DailyIntelOutput(BaseModel):
    report_markdown: str
    email_sent: bool
    open_actions: int
    email_items_analysed: int
    market_items: int


class EmailItem(BaseModel):
    id: str
    subject: str = ""
    sender: str = ""
    date: str = ""
    body: str = ""


class NewAction(BaseModel):
    action: str
    requester: str = ""
    due_date: str = ""
    email_subject: str = ""


class ActionUpdate(BaseModel):
    matched_action: str
    status: Literal["resolved", "progressing", "blocked"] = "progressing"
    update: str = ""


class SignificantEmail(BaseModel):
    subject: str
    sender: str = ""
    summary: str
    why_it_matters: str = ""


class EmailAnalysis(BaseModel):
    new_actions: list[NewAction] = []
    action_updates: list[ActionUpdate] = []
    significant_emails: list[SignificantEmail] = []
    noise_count: int = 0


class MarketItem(BaseModel):
    country: str
    title: str
    date: str = ""
    summary: str = ""
    source_url: str = ""


class RegulatoryChange(BaseModel):
    country: str
    what_changed: str = ""
    previous_state: str = ""
    new_state: str = ""
    source_url: str = ""
    confidence: Literal["confirmed", "likely", "rumour"] = "likely"


class CountryScan(BaseModel):
    updates: list[MarketItem] = []
    regulatory_changes: list[RegulatoryChange] = []
    baseline_current: str = ""


class MarketScan(BaseModel):
    kenya: CountryScan
    tanzania: CountryScan
    mauritius: CountryScan

    def all_updates(self) -> list[MarketItem]:
        items: list[MarketItem] = []
        for scan in [self.kenya, self.tanzania, self.mauritius]:
            items.extend(scan.updates)
        return items

    def all_regulatory_changes(self) -> list[RegulatoryChange]:
        changes: list[RegulatoryChange] = []
        for scan in [self.kenya, self.tanzania, self.mauritius]:
            changes.extend(scan.regulatory_changes)
        return changes


class Opportunity(BaseModel):
    area: str = ""
    description: str
    rationale: str = ""


class Risk(BaseModel):
    description: str
    urgency: Literal["high", "medium", "low"] = "medium"


class DecisionMatrixRow(BaseModel):
    action: str
    category: str = "general"
    impact: int = Field(default=3, ge=1, le=5)
    urgency: int = Field(default=3, ge=1, le=5)
    effort: int = Field(default=3, ge=1, le=5)
    recommendation: str = ""


class StrategicAnalysis(BaseModel):
    strategic_summary: str = ""
    opportunities: list[Opportunity] = []
    risks: list[Risk] = []
    decision_matrix: list[DecisionMatrixRow] = []


class IntelState(BaseModel):
    open_actions: list[dict[str, Any]] = []
    regulatory_baseline: dict[str, Any] = {}
    seen_events: list[dict[str, Any]] = []
    last_run_at: str = ""


# ---------------------------------------------------------------------------
# Gmail REST API helpers (activity-side only)
# ---------------------------------------------------------------------------

GMAIL_TOKEN_URL = "https://oauth2.googleapis.com/token"
GMAIL_API = "https://gmail.googleapis.com/gmail/v1/users/me"


async def _gmail_access_token() -> str:
    client_id = os.environ.get("GMAIL_CLIENT_ID", "").strip()
    client_secret = os.environ.get("GMAIL_CLIENT_SECRET", "").strip()
    refresh_token = os.environ.get("GMAIL_REFRESH_TOKEN", "").strip()
    if not (client_id and client_secret and refresh_token):
        raise ValueError(
            "Missing Gmail credentials: set GMAIL_CLIENT_ID, GMAIL_CLIENT_SECRET and "
            "GMAIL_REFRESH_TOKEN as workspace secrets bound to the deployment."
        )
    async with httpx.AsyncClient(timeout=30.0) as http:
        response = await http.post(
            GMAIL_TOKEN_URL,
            data={
                "grant_type": "refresh_token",
                "client_id": client_id,
                "client_secret": client_secret,
                "refresh_token": refresh_token,
            },
        )
        response.raise_for_status()
        return response.json()["access_token"]


async def _gmail_get(
    token: str, path: str, params: dict[str, Any] | None = None
) -> Any:
    async with httpx.AsyncClient(timeout=30.0) as http:
        response = await http.get(
            f"{GMAIL_API}{path}",
            params=params,
            headers={"Authorization": f"Bearer {token}"},
        )
        response.raise_for_status()
        return response.json()


def _extract_text_body(payload: dict[str, Any]) -> str:
    parts: list[str] = []

    def walk(node: dict[str, Any]) -> None:
        if node.get("mimeType") in ("text/plain", "text/html") and node.get("body"):
            data = node["body"].get("data")
            if data:
                text = base64.urlsafe_b64decode(data).decode("utf-8", errors="replace")
                if node.get("mimeType") == "text/plain":
                    parts.insert(0, text)
                else:
                    parts.append(text)
        for child in node.get("parts", []) or []:
            walk(child)

    walk(payload)
    body = parts[0] if parts else ""
    return body[:6000]


# ---------------------------------------------------------------------------
# Activities
# ---------------------------------------------------------------------------


@workflows.activity()
async def fetch_recent_emails(
    since_date: str, max_messages: int = 25
) -> list[EmailItem]:
    token = await _gmail_access_token()
    query_list = await _gmail_get(
        token,
        "/messages",
        params={"q": f"after:{since_date}", "maxResults": max_messages},
    )
    items: list[EmailItem] = []
    for header in query_list.get("messages", []) or []:
        message = await _gmail_get(token, f"/messages/{header['id']}")
        headers = {
            h["name"].lower(): h["value"] for h in message["payload"].get("headers", [])
        }
        items.append(
            EmailItem(
                id=header["id"],
                subject=headers.get("subject", "(no subject)"),
                sender=headers.get("from", ""),
                date=headers.get("date", ""),
                body=_extract_text_body(message["payload"])
                or message.get("snippet", ""),
            )
        )
    return items


STATE_LIBRARY_NAME = "reinsurance-intel-state"
STATE_DOCUMENTS = ("open_actions", "regulatory_baseline", "seen_events", "last_run_at")


def _mistral_api_config() -> tuple[str, str]:
    from env.mistral import env as mistral_env

    base_url = (
        mistral_env.mistral_base_url
        or os.environ.get("MISTRAL_BASE_URL")
        or "https://api.mistral.ai"
    ).rstrip("/")
    api_key = mistral_env.mistral_api_key or os.environ.get("MISTRAL_API_KEY", "")
    if not api_key:
        raise ValueError("No Mistral API key available for state storage.")
    return base_url, api_key


def _mistral_headers(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"}


async def _find_state_library(base_url: str, api_key: str) -> dict[str, Any] | None:
    page = 0
    async with httpx.AsyncClient(timeout=30.0) as http:
        while page < 10:
            response = await http.get(
                f"{base_url}/v1/libraries",
                params={"page_size": 100, "page": page},
                headers=_mistral_headers(api_key),
            )
            response.raise_for_status()
            payload = response.json()
            for library in payload.get("data", []) or []:
                if library.get("name") == STATE_LIBRARY_NAME:
                    return library
            if payload.get("has_more") is not True:
                break
            page += 1
    return None


async def _create_state_library(base_url: str, api_key: str) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=30.0) as http:
        response = await http.post(
            f"{base_url}/v1/libraries",
            json={
                "name": STATE_LIBRARY_NAME,
                "description": "State storage for the reinsurance daily briefing workflow.",
            },
            headers=_mistral_headers(api_key),
        )
        response.raise_for_status()
        return response.json()


async def _list_state_documents(
    base_url: str, api_key: str, library_id: str
) -> dict[str, str]:
    ids: dict[str, str] = {}
    page = 0
    async with httpx.AsyncClient(timeout=30.0) as http:
        while page < 10:
            response = await http.get(
                f"{base_url}/v1/libraries/{library_id}/documents",
                params={"page_size": 100, "page": page},
                headers=_mistral_headers(api_key),
            )
            response.raise_for_status()
            payload = response.json()
            for document in payload.get("data", []) or []:
                if document.get("name"):
                    ids[document["name"]] = document["id"]
            if payload.get("has_more") is not True:
                break
            page += 1
    return ids


async def _delete_state_document(
    base_url: str, api_key: str, library_id: str, document_id: str
) -> None:
    async with httpx.AsyncClient(timeout=30.0) as http:
        response = await http.delete(
            f"{base_url}/v1/libraries/{library_id}/documents/{document_id}",
            headers=_mistral_headers(api_key),
        )
        if response.status_code not in (200, 204, 404):
            response.raise_for_status()


async def _upload_state_document(
    base_url: str, api_key: str, library_id: str, name: str, content: str
) -> None:
    async with httpx.AsyncClient(timeout=60.0) as http:
        response = await http.post(
            f"{base_url}/v1/libraries/{library_id}/documents",
            files={"file": (name, content.encode("utf-8"), "text/plain")},
            headers=_mistral_headers(api_key),
        )
        response.raise_for_status()


async def _read_state_document(
    base_url: str, api_key: str, library_id: str, document_id: str
) -> str:
    async with httpx.AsyncClient(timeout=30.0) as http:
        response = await http.get(
            f"{base_url}/v1/libraries/{library_id}/documents/{document_id}/text_content",
            headers=_mistral_headers(api_key),
        )
        if response.status_code == 404:
            return ""
        response.raise_for_status()
        return str(response.json().get("text", ""))


@workflows.activity()
async def load_intel_state() -> IntelState:
    base_url, api_key = _mistral_api_config()
    library = await _find_state_library(base_url, api_key)
    if library is None:
        return IntelState()

    document_ids = await _list_state_documents(base_url, api_key, library["id"])
    state = IntelState()
    for key in STATE_DOCUMENTS:
        document_id = document_ids.get(f"{key}.txt")
        if not document_id:
            continue
        text = await _read_state_document(base_url, api_key, library["id"], document_id)
        if not text.strip():
            continue
        if key == "last_run_at":
            state.last_run_at = text.strip()
            continue
        try:
            setattr(state, key, json.loads(text))
        except json.JSONDecodeError:
            continue
    if not isinstance(state.last_run_at, str):
        state.last_run_at = ""
    return state


@workflows.activity()
async def save_intel_state(state: IntelState) -> None:
    base_url, api_key = _mistral_api_config()
    library = await _find_state_library(
        base_url, api_key
    ) or await _create_state_library(base_url, api_key)
    document_ids = await _list_state_documents(base_url, api_key, library["id"])

    payloads: dict[str, str] = {
        "open_actions.txt": json.dumps(state.open_actions, indent=2),
        "regulatory_baseline.txt": json.dumps(state.regulatory_baseline, indent=2),
        "seen_events.txt": json.dumps(state.seen_events, indent=2),
        "last_run_at.txt": state.last_run_at,
    }
    for name, content in payloads.items():
        existing_id = document_ids.get(name)
        if existing_id:
            await _delete_state_document(base_url, api_key, library["id"], existing_id)
        await _upload_state_document(base_url, api_key, library["id"], name, content)


@workflows.activity()
async def send_briefing_email(subject: str, body_text: str) -> bool:
    token = await _gmail_access_token()
    profile = await _gmail_get(token, "/profile")
    recipient = os.environ.get("RECIPIENT_EMAIL", "").strip() or profile.get(
        "emailAddress", ""
    )
    if not recipient:
        raise ValueError("No recipient: set RECIPIENT_EMAIL as a workspace secret.")

    message = EmailMessage()
    message["To"] = recipient
    message["Subject"] = subject
    message.set_content(body_text)
    raw = base64.urlsafe_b64encode(message.as_bytes()).decode("ascii")

    async with httpx.AsyncClient(timeout=30.0) as http:
        response = await http.post(
            f"{GMAIL_API}/messages/send",
            json={"raw": raw},
            headers={"Authorization": f"Bearer {token}"},
        )
        response.raise_for_status()
    return True


# ---------------------------------------------------------------------------
# The workflow
# ---------------------------------------------------------------------------

EMAIL_ANALYST_INSTRUCTIONS = """You are an executive assistant analysing a business inbox.
Given recent emails and the current open action tracker, produce an EmailAnalysis:
- new_actions: concrete things the recipient (or their team) must DO. Include who asked
  (requester), any stated due date, and the email subject. Skip marketing, newsletters,
  notifications and webinars unless the recipient is explicitly expected to act.
- action_updates: for each existing open action, decide whether an email in the list
  advances, resolves or blocks it, with a one-line update.
- significant_emails: non-action emails that still matter strategically (results, broker
  correspondence, client signals), with a one-sentence summary each.
- noise_count: emails that required no attention.
Treat empty email list as nothing to report."""

MARKET_ANALYST_INSTRUCTIONS = """You are a reinsurance market intelligence analyst covering
Kenya, Tanzania and Mauritius. Use web search and return ONLY a JSON object (no commentary):
{
  "kenya": {"updates": [...], "regulatory_changes": [...], "baseline_current": "<one-line status>"},
  "tanzania": {...}, "mauritius": {...}
}
Rules:
- updates: items where each entry has country, title, date (YYYY-MM-DD if known), summary
  (1-2 sentences), source_url. Focus on items from the last 7 days: market renewals, treaty
  placements, rate movement, capacity changes, M&A, new entrants/exit, cat events with
  reinsurance relevance, IRA/REGS/TIRA/FSC news with market impact.
- regulatory_changes: ONLY if compared to the provided baseline summary something CHANGED
  (capital requirements, regulatory framework, licensing, VAT/levy on premiums, new circulars
  or regulations). "what_changed", "previous_state", "new_state", "source_url" are required.
  If nothing changed, return an empty list for that country — do NOT repeat the baseline.
- baseline_current: one line confirming the current framework status.
Never invent sources. If you cannot verify an item, omit it."""

STRATEGIST_INSTRUCTIONS = """You are a strategy advisor for an insurance/reinsurance executive.
Given the executive action tracker, market intelligence and regulatory changes, produce:
- strategic_summary: 3-5 sentences: what today's picture means for the business.
- opportunities: concrete business opportunities implied by the intelligence.
- risks: things that could hurt us, with urgency.
- decision_matrix: the KEY actions to consider now, scored impact (1-5), urgency (1-5),
  effort (1-5), each with a crisp recommendation (decide / delegate / schedule / ignore).
Prioritise by impact*urgency and be specific — reference the actual events.
Return only valid structured output."""


@workflows.workflow.define(name="daily_intel_report")
class DailyIntelReport:
    @workflows.workflow.entrypoint
    async def run(self, data: DailyIntelInput) -> DailyIntelOutput:
        now = workflows.workflow.now()
        state = await load_intel_state()

        last_run: datetime | None = None
        if state.last_run_at:
            try:
                last_run = datetime.fromisoformat(state.last_run_at)
            except ValueError:
                last_run = None
        lookback = timedelta(hours=data.lookback_hours)
        since = last_run or (now - lookback)
        since_date = since.strftime("%Y/%m/%d")

        emails = await fetch_recent_emails(since_date, data.max_emails)
        email_digest = [
            email.model_dump(exclude={"id"}) | {"id": email.id} for email in emails
        ]

        email_request = ChatCompletionRequest(
            model="mistral-medium-latest",
            messages=[
                UserMessage(
                    content=(
                        f"{EMAIL_ANALYST_INSTRUCTIONS}\n\n"
                        f"### Current open actions\n```json\n{json.dumps(state.open_actions, indent=2)}\n```\n\n"
                        f"### Recent emails (since {since.isoformat()})\n```json\n{json.dumps(email_digest, indent=2)}\n```"
                    )
                )
            ],
        )
        email_analysis: EmailAnalysis = await chat_parse_to_model(
            EmailAnalysis, email_request
        )

        market_agent = Agent(
            model="mistral-large-latest",
            name="reinsurance-market-analyst",
            description="Researches reinsurance market intelligence for East Africa islands and Kenya/Tanzania.",
            instructions=MARKET_ANALYST_INSTRUCTIONS,
            tools=[WebSearchTool()],
        )
        market_prompt = (
            f"Today is {now.date().isoformat()}. Research since {since.date().isoformat()}. "
            f"Coverage countries: Kenya, Tanzania, Mauritius.\n\n"
            f"### Current regulatory baselines\n```json\n{json.dumps(state.regulatory_baseline, indent=2)}\n```\n\n"
            f"### Already-reported events (do not repeat unless there is new development)\n"
            f"```json\n{json.dumps(state.seen_events[-40:], indent=2)}\n```"
        )
        market_outputs = await Runner.run(
            agent=market_agent,
            inputs=market_prompt,
            session=LocalSession(),
            max_turns=15,
        )
        market_text = (
            market_outputs if isinstance(market_outputs, str) else str(market_outputs)
        )
        market_scan = _parse_json_block(market_text, MarketScan)

        strategic_request = ChatCompletionRequest(
            model="mistral-large-latest",
            messages=[
                UserMessage(
                    content=(
                        f"{STRATEGIST_INSTRUCTIONS}\n\n"
                        f"### Email analysis\n```json\n{email_analysis.model_dump_json(indent=2)}\n```\n\n"
                        f"### Market scan\n```json\n{market_scan.model_dump_json(indent=2)}\n```\n\n"
                        f"### Current open actions\n```json\n{json.dumps(state.open_actions, indent=2)}\n```"
                    )
                )
            ],
        )
        strategic: StrategicAnalysis = await chat_parse_to_model(
            StrategicAnalysis, strategic_request
        )

        report = _compose_report(now, email_analysis, market_scan, strategic)

        new_state = _merge_state(now, state, email_analysis, market_scan)
        await save_intel_state(new_state)

        email_sent = False
        if not data.dry_run:
            email_sent = await send_briefing_email(
                subject=f"Daily Executive Briefing — {now.date().isoformat()}",
                body_text=report,
            )

        return DailyIntelOutput(
            report_markdown=report,
            email_sent=email_sent,
            open_actions=len(new_state.open_actions),
            email_items_analysed=len(emails),
            market_items=len(market_scan.all_updates())
            + len(market_scan.all_regulatory_changes()),
        )


def _parse_json_block(text: str, model: Any) -> Any:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON object found in agent output: {text[:200]}")
    return model.model_validate_json(text[start : end + 1])


def _merge_state(
    now: Any,
    state: IntelState,
    email_analysis: EmailAnalysis,
    market_scan: MarketScan,
) -> IntelState:
    actions_by_action = {
        str(item.get("action", "")): item for item in state.open_actions
    }
    for update in email_analysis.action_updates:
        tracked = actions_by_action.get(update.matched_action)
        if tracked is not None:
            tracked["status"] = update.status
            tracked["last_update"] = update.update
    for new_action in email_analysis.new_actions:
        key = new_action.action
        if any(
            key == str(existing.get("action", "")) for existing in state.open_actions
        ):
            continue
        actions_by_action[key] = {
            "action": key,
            "requester": new_action.requester,
            "due_date": new_action.due_date,
            "source_subject": new_action.email_subject,
            "opened_at": now.date().isoformat(),
            "status": "open",
            "last_update": "Opened",
        }
    open_actions = sorted(
        (
            item
            for item in actions_by_action.values()
            if item.get("status") != "resolved"
        ),
        key=lambda item: str(item.get("opened_at", "")),
        reverse=True,
    )

    baseline = dict(state.regulatory_baseline)
    for change in market_scan.all_regulatory_changes():
        if change.confidence == "rumour":
            continue
        baseline[change.country] = {
            "summary": change.new_state
            or baseline.get(change.country, {}).get("summary", ""),
            "updated": now.date().isoformat(),
            "change_log": baseline.get(change.country, {}).get("change_log", [])
            + [
                {
                    "date": now.date().isoformat(),
                    "what_changed": change.what_changed,
                    "source_url": change.source_url,
                }
            ],
        }

    existing_seen = list(state.seen_events)
    existing_seen.extend(
        {
            "id": f"{item.country}:{item.title}",
            "country": item.country,
            "title": item.title,
            "date": item.date or now.date().isoformat(),
        }
        for item in market_scan.all_updates()
    )
    seen_events = existing_seen[-200:]

    return IntelState(
        open_actions=open_actions,
        regulatory_baseline=baseline,
        seen_events=seen_events,
        last_run_at=now.isoformat(),
    )


def _compose_report(
    now: Any,
    email_analysis: EmailAnalysis,
    market_scan: MarketScan,
    strategic: StrategicAnalysis,
) -> str:
    lines: list[str] = [f"# Daily Executive Briefing — {now.date().isoformat()}", ""]

    lines += ["## 1. Email Intelligence & Executive Action Tracker", ""]
    if email_analysis.new_actions:
        lines.append("### New actions")
        lines.append("| # | Action | Requested by | Due |")
        lines.append("| --- | --- | --- | --- |")
        for index, action in enumerate(email_analysis.new_actions, 1):
            lines.append(
                f"| {index} | {action.action} | {action.requester or '—'} | {action.due_date or '—'} |"
            )
        lines.append("")
    else:
        lines.append("_No new actions today._")
        lines.append("")
    if email_analysis.action_updates:
        lines.append("### Tracked action updates")
        for update in email_analysis.action_updates:
            lines.append(
                f"- **[{update.status}]** {update.matched_action}: {update.update}"
            )
        lines.append("")
    if email_analysis.significant_emails:
        lines.append("### Emails worth knowing")
        for email in email_analysis.significant_emails[:8]:
            lines.append(f"- **{email.subject}** ({email.sender}): {email.summary}")
        lines.append("")
    if email_analysis.noise_count:
        lines.append(f"_{email_analysis.noise_count} low-value emails filtered out._")
        lines.append("")

    lines += ["## 2. Reinsurance Market Intelligence", ""]
    scans = [
        ("Kenya", market_scan.kenya),
        ("Tanzania", market_scan.tanzania),
        ("Mauritius", market_scan.mauritius),
    ]
    for country, scan in scans:
        lines.append(f"### {country}")
        if not scan.updates and not scan.regulatory_changes:
            lines.append("_No new developments reported._")
            lines.append("")
            continue
        for item in scan.updates[:6]:
            date_part = f" ({item.date})" if item.date else ""
            source_part = f" — [source]({item.source_url})" if item.source_url else ""
            lines.append(f"- **{item.title}**{date_part}: {item.summary}{source_part}")
        if scan.regulatory_changes:
            lines.append("")
            lines.append("**⚠ Regulatory / framework changes:**")
            for change in scan.regulatory_changes:
                lines.append(
                    f"- {change.what_changed}"
                    + (
                        f" — was: {change.previous_state}; now: {change.new_state}"
                        if change.previous_state
                        else ""
                    )
                    + (f" — [source]({change.source_url})" if change.source_url else "")
                    + f" ({change.confidence})"
                )
        lines.append("")

    lines += ["## 3. Strategic Analysis & Business Opportunities", ""]
    if strategic.strategic_summary:
        lines.append(strategic.strategic_summary)
        lines.append("")
    if strategic.opportunities:
        lines.append("### Opportunities")
        for opportunity in strategic.opportunities[:6]:
            header = f" **{opportunity.area}**" if opportunity.area else ""
            lines.append(f"-{header}: {opportunity.description}")
        lines.append("")
    if strategic.risks:
        lines.append("### Risks")
        for risk in strategic.risks[:6]:
            lines.append(f"- **[{risk.urgency}]** {risk.description}")
        lines.append("")

    lines += ["## 4. Decision Matrix", ""]
    if strategic.decision_matrix:
        rows = sorted(
            strategic.decision_matrix,
            key=lambda row: row.impact * row.urgency,
            reverse=True,
        )
        lines.append(
            "| Priority | Action | Category | Impact | Urgency | Effort | Recommendation |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for rank, row in enumerate(rows, 1):
            score = row.impact * row.urgency
            lines.append(
                f"| {rank} ({score}) | {row.action} | {row.category} | {row.impact} | "
                f"{row.urgency} | {row.effort} | {row.recommendation} |"
            )
    else:
        lines.append("_No decisions recommended today._")
    lines.append("")

    return "\n".join(lines)
