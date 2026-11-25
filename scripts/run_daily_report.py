"""Standalone daily executive briefing runner, executed by GitHub Actions.

Reuses the workflow's models, prompts and helper functions and talks to the Gmail
and Mistral APIs directly, so no workflow worker or managed deployment is needed.

Required environment (GitHub Actions secrets):
    MISTRAL_API_KEY, GMAIL_CLIENT_ID, GMAIL_CLIENT_SECRET,
    GMAIL_REFRESH_TOKEN, RECIPIENT_EMAIL
"""

import argparse
import asyncio
import base64
import json
import os
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from email.message import EmailMessage
from pathlib import Path
from typing import Any

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "apps" / "worker" / "src"))

from mistralai.client import (
    Mistral,
)
from worker.workflows.daily_intel import (
    EMAIL_ANALYST_INSTRUCTIONS,
    GMAIL_API,
    MARKET_ANALYST_INSTRUCTIONS,
    STRATEGIST_INSTRUCTIONS,
    EmailAnalysis,
    IntelState,
    MarketScan,
    StrategicAnalysis,
    _compose_report,
    _create_state_library,
    _delete_state_document,
    _extract_text_body,
    _find_state_library,
    _gmail_access_token,
    _gmail_get,
    _list_state_documents,
    _merge_state,
    _mistral_api_config,
    _parse_json_block,
    _read_state_document,
    _upload_state_document,
)

STATE_DOCUMENTS = ("open_actions", "regulatory_baseline", "seen_events", "last_run_at")

EMAIL_MODEL = os.environ.get("EMAIL_ANALYST_MODEL", "mistral-medium-latest")
MARKET_MODEL = os.environ.get("MARKET_ANALYST_MODEL", "mistral-large-latest")
STRATEGY_MODEL = os.environ.get("STRATEGY_ANALYST_MODEL", "mistral-large-latest")

RETRY_ATTEMPTS = 5
RETRY_BASE_DELAY_SECONDS = 20.0


async def with_retries(factory: Callable[[], Any], what: str) -> Any:
    """Run an API call, backing off on 429 rate limits."""
    delay = RETRY_BASE_DELAY_SECONDS
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            return await factory()
        except Exception as error:
            message = str(error)
            if "429" not in message and "rate" not in message.lower():
                raise
            if attempt == RETRY_ATTEMPTS:
                print(
                    f"{what}: still rate limited after {RETRY_ATTEMPTS} attempts. "
                    "Enable pay-as-you-go billing on your Mistral account to lift free-tier limits."
                )
                raise
            print(
                f"{what}: rate limited (attempt {attempt}/{RETRY_ATTEMPTS}); waiting {delay:.0f}s"
            )
            await asyncio.sleep(delay)
            delay *= 2
    raise RuntimeError(f"{what}: exhausted retries")


async def fetch_emails(since_date: str, max_messages: int = 25) -> list[dict[str, Any]]:
    token = await _gmail_access_token()
    listing = await _gmail_get(
        token,
        "/messages",
        params={"q": f"after:{since_date}", "maxResults": max_messages},
    )
    items: list[dict[str, Any]] = []
    for header in listing.get("messages", []) or []:
        message = await _gmail_get(token, f"/messages/{header['id']}")
        headers = {
            entry["name"].lower(): entry["value"]
            for entry in message["payload"].get("headers", [])
        }
        items.append(
            {
                "id": header["id"],
                "subject": headers.get("subject", "(no subject)"),
                "sender": headers.get("from", ""),
                "date": headers.get("date", ""),
                "body": _extract_text_body(message["payload"])
                or message.get("snippet", ""),
            }
        )
    return items


async def load_state(base_url: str, api_key: str) -> IntelState:
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
    return state


async def save_state(base_url: str, api_key: str, state: IntelState) -> None:
    library = await _find_state_library(
        base_url, api_key
    ) or await _create_state_library(base_url, api_key)
    document_ids = await _list_state_documents(base_url, api_key, library["id"])
    payloads = {
        "open_actions.txt": json.dumps(state.open_actions, indent=2),
        "regulatory_baseline.txt": json.dumps(state.regulatory_baseline, indent=2),
        "seen_events.txt": json.dumps(state.seen_events, indent=2),
        "last_run_at.txt": state.last_run_at,
    }
    for name, content in payloads.items():
        if name in document_ids:
            await _delete_state_document(
                base_url, api_key, library["id"], document_ids[name]
            )
        await _upload_state_document(base_url, api_key, library["id"], name, content)


def last_output_text(response: Any) -> str:
    texts: list[str] = []
    for output in getattr(response, "outputs", []) or []:
        content = getattr(output, "content", "")
        if isinstance(content, str):
            texts.append(content)
        elif isinstance(content, list):
            texts.extend(
                getattr(chunk, "text", "") or ""
                for chunk in content
                if hasattr(chunk, "text")
            )
    return "\n".join(text for text in texts if text.strip())


async def send_email(subject: str, body_text: str) -> bool:
    token = await _gmail_access_token()
    profile = await _gmail_get(token, "/profile")
    recipient = os.environ.get("RECIPIENT_EMAIL", "").strip() or profile.get(
        "emailAddress", ""
    )
    if not recipient:
        raise ValueError("No recipient: set RECIPIENT_EMAIL.")

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


async def run(dry_run: bool) -> int:
    if not os.environ.get("MISTRAL_API_KEY", "").strip():
        print("Missing required secret: MISTRAL_API_KEY")
        return 1

    base_url, api_key = _mistral_api_config()
    client = Mistral(api_key=api_key)
    now = datetime.now(UTC)

    state = await load_state(base_url, api_key)
    if state.last_run_at:
        try:
            since = datetime.fromisoformat(state.last_run_at)
        except ValueError:
            since = now
    else:
        since = now.fromtimestamp(now.timestamp() - 36 * 3600, tz=UTC)
    if since >= now:
        since = now.fromtimestamp(now.timestamp() - 36 * 3600, tz=UTC)
    print(f"Scanning inbox since {since.isoformat()}")

    emails = await fetch_emails(since.strftime("%Y/%m/%d"))
    print(f"Fetched {len(emails)} emails")

    email_analysis: EmailAnalysis = (
        (
            await with_retries(
                lambda: client.chat.parse_async(
                    EmailAnalysis,
                    model=EMAIL_MODEL,
                    messages=[
                        {
                            "role": "user",
                            "content": (
                                f"{EMAIL_ANALYST_INSTRUCTIONS}\n\n"
                                f"### Current open actions\n```json\n{json.dumps(state.open_actions, indent=2)}\n```\n\n"
                                f"### Recent emails (since {since.isoformat()})\n```json\n{json.dumps(emails, indent=2)}\n```"
                            ),
                        }
                    ],
                ),
                what="email analysis",
            )
        )
        .choices[0]
        .message.parsed
    )

    agent = await with_retries(
        lambda: client.beta.agents.create_async(
            model=MARKET_MODEL,
            name=f"reinsurance-market-analyst-{now.strftime('%Y%m%d%H%M%S')}",
            instructions=MARKET_ANALYST_INSTRUCTIONS,
            tools=[{"type": "web_search"}],
            completion_args={"temperature": 0.2},
        ),
        what="market agent",
    )
    try:
        conversation = await with_retries(
            lambda: client.beta.conversations.start_async(
                agent_id=agent.id,
                inputs=(
                    f"Today is {now.date().isoformat()}. Research since {since.date().isoformat()}. "
                    "Coverage countries: Kenya, Tanzania, Mauritius.\n\n"
                    f"### Current regulatory baselines\n```json\n{json.dumps(state.regulatory_baseline, indent=2)}\n```\n\n"
                    "### Already-reported events (do not repeat unless there is new development)\n"
                    f"```json\n{json.dumps(state.seen_events[-40:], indent=2)}\n```"
                ),
                store=False,
            ),
            what="market research",
        )
    finally:
        try:
            await client.beta.agents.delete_async(agent_id=agent.id)
        except Exception as cleanup_error:  # noqa: BLE001 - cleanup must not mask the run
            print(f"Agent cleanup skipped: {cleanup_error}")
    market_scan = _parse_json_block(last_output_text(conversation), MarketScan)

    strategic: StrategicAnalysis = (
        (
            await with_retries(
                lambda: client.chat.parse_async(
                    StrategicAnalysis,
                    model=STRATEGY_MODEL,
                    messages=[
                        {
                            "role": "user",
                            "content": (
                                f"{STRATEGIST_INSTRUCTIONS}\n\n"
                                f"### Email analysis\n```json\n{email_analysis.model_dump_json(indent=2)}\n```\n\n"
                                f"### Market scan\n```json\n{market_scan.model_dump_json(indent=2)}\n```\n\n"
                                f"### Current open actions\n```json\n{json.dumps(state.open_actions, indent=2)}\n```"
                            ),
                        }
                    ],
                ),
                what="strategic analysis",
            )
        )
        .choices[0]
        .message.parsed
    )

    report = _compose_report(now, email_analysis, market_scan, strategic)

    new_state = _merge_state(now, state, email_analysis, market_scan)
    new_state.last_run_at = now.isoformat()
    await save_state(base_url, api_key, new_state)

    output = os.environ.get("REPORT_OUTPUT", "").strip()
    if output:
        Path(output).write_text(report, encoding="utf-8")

    email_sent = False
    if not dry_run:
        email_sent = await send_email(
            subject=f"Daily Executive Briefing — {now.date().isoformat()}",
            body_text=report,
        )

    market_count = len(market_scan.all_updates()) + len(
        market_scan.all_regulatory_changes()
    )
    print(
        f"Done: {len(emails)} emails analysed, {len(new_state.open_actions)} open actions, "
        f"{market_count} market items, email_sent={email_sent}, dry_run={dry_run}"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the daily executive briefing.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Analyse and persist state, but do not send the email.",
    )
    args = parser.parse_args()
    return asyncio.run(run(args.dry_run))


if __name__ == "__main__":
    raise SystemExit(main())
