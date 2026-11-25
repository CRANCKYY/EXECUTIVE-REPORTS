# EXECUTIVE-REPORTS

Daily automated executive intelligence briefing covering:

1. **Email intelligence & executive action tracker** — reads the inbox, extracts actions (requests, commitments, deadlines), tracks their status day over day, and filters noise.
2. **Reinsurance market intelligence** — web research for **Kenya, Tanzania and Mauritius** (renewals, treaty placements, rates, capacity, M&A, cat events), deduplicated between runs.
3. **Regulatory change detection** — capital requirements and regulatory frameworks are stored once as a per-country baseline; the report only flags **changes** (new circulars, capital changes, levies, licensing), never repeats the baseline.
4. **Strategic analysis & decision matrix** — opportunities, risks, and a ranked table of key actions (impact × urgency × effort) with recommendations.

The briefing is delivered by email from your own Gmail account each morning.

## How it runs

A GitHub Actions scheduled job (`.github/workflows/daily-report.yml`) runs
`scripts/run_daily_report.py` daily at **06:00 East Africa Time** (03:00 UTC),
then emails you the report. A weekly keepalive workflow (`.github/workflows/keepalive.yml`)
keeps the schedule from being auto-disabled.

Day-over-day state (open actions, regulatory baselines, already-reported events) is
persisted in a **Mistral document library** named `reinsurance-intel-state` in your
workspace, created automatically on first run.

## Setup (one time)

Add these five **repository secrets** at *Settings → Secrets and variables → Actions*:

| Secret | Where to get it |
| --- | --- |
| `MISTRAL_API_KEY` | https://console.mistral.ai/api-keys/new |
| `GMAIL_CLIENT_ID` | Google Cloud Console → OAuth client (Desktop app), after enabling the Gmail API |
| `GMAIL_CLIENT_SECRET` | Same OAuth client |
| `GMAIL_REFRESH_TOKEN` | Via https://developers.google.com/oauthplayground with your own OAuth credentials and scopes `https://www.googleapis.com/auth/gmail.readonly`, `https://www.googleapis.com/auth/gmail.send`, `https://www.googleapis.com/auth/userinfo.email` |
| `RECIPIENT_EMAIL` | The address that receives the briefing (defaults to the Gmail account) |

### Test it manually

*Actions → Daily executive report → Run workflow* — leave **dry_run** enabled
(creates everything but does not send an email; the report is attached as an artifact).
Uncheck dry run, or wait for the 06:00 schedule, for real delivery.

The same workflow also exists as a Mistral Workflow (`apps/worker/`) for hosted
deployment under branch `vibe/daily-intel-0ba6ec`.

## Timezone

To change delivery time, edit the `cron` value in `.github/workflows/daily-report.yml`
(UTC). 06:00 EAT = 03:00 UTC; 09:00 EAT = 06:00 UTC, etc.
