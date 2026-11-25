from datetime import UTC, datetime

from worker.workflows.daily_intel import (
    EmailAnalysis,
    IntelState,
    MarketScan,
    StrategicAnalysis,
    _merge_state,
    _parse_json_block,
)


def _now() -> datetime:
    return datetime(2026, 11, 24, 6, 0, tzinfo=UTC)


def _market_scan() -> MarketScan:
    return MarketScan.model_validate(
        {
            "kenya": {
                "updates": [
                    {
                        "country": "Kenya",
                        "title": "IRA reviews minimum capital",
                        "date": "2026-11-23",
                        "summary": "Regulator opens consultation.",
                        "source_url": "https://example.com/a",
                    }
                ],
                "regulatory_changes": [
                    {
                        "country": "Kenya",
                        "what_changed": "Minimum capital raised",
                        "previous_state": "KES 250m",
                        "new_state": "KES 400m",
                        "source_url": "https://example.com/a",
                        "confidence": "confirmed",
                    }
                ],
                "baseline_current": "ok",
            },
            "tanzania": {
                "updates": [],
                "regulatory_changes": [],
                "baseline_current": "ok",
            },
            "mauritius": {
                "updates": [],
                "regulatory_changes": [
                    {
                        "country": "Mauritius",
                        "what_changed": "rumoured levy",
                        "new_state": "",
                        "confidence": "rumour",
                    }
                ],
                "baseline_current": "ok",
            },
        }
    )


def test_merge_state_tracks_actions_and_dedupes_rumours() -> None:
    state = IntelState(
        open_actions=[
            {
                "action": "Send updated treaty terms",
                "status": "open",
                "opened_at": "2026-11-20",
            }
        ]
    )
    email_analysis = EmailAnalysis.model_validate(
        {
            "new_actions": [
                {
                    "action": "Confirm reinsurance renewal budget",
                    "requester": "CFO",
                    "due_date": "2026-11-30",
                }
            ],
            "action_updates": [
                {
                    "matched_action": "Send updated treaty terms",
                    "status": "resolved",
                    "update": "Terms emailed on 21 Nov",
                }
            ],
        }
    )

    merged = _merge_state(_now(), state, email_analysis, _market_scan())

    assert len(merged.open_actions) == 1
    assert merged.open_actions[0]["action"] == "Confirm reinsurance renewal budget"

    assert "Kenya" in merged.regulatory_baseline
    assert merged.regulatory_baseline["Kenya"]["summary"] == "KES 400m"
    assert "Mauritius" not in merged.regulatory_baseline

    assert any(
        event["id"] == "Kenya:IRA reviews minimum capital"
        for event in merged.seen_events
    )
    assert merged.last_run_at == "2026-11-24T06:00:00+00:00"


def test_parse_json_block_parses_embedded_json() -> None:
    text = 'Here is the result:\n```json\n{"kenya": {"updates": []}, "tanzania": {"updates": []}, "mauritius": {"updates": []}}\n```'
    scan = _parse_json_block(text, MarketScan)
    assert scan.all_updates() == []


def test_strategic_analysis_defaults() -> None:
    analysis = StrategicAnalysis()
    assert analysis.decision_matrix == []
