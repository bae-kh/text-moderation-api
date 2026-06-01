import json
import logging

from app.core.logging import log_event


def test_log_event_outputs_json(caplog) -> None:
    with caplog.at_level(logging.INFO, logger="app"):
        log_event(
            event="detect_completed",
            request_id="test-request-id",
            action="block",
            stored=True,
            latency_ms=12.34,
        )

    assert len(caplog.records) == 1

    log_message = caplog.records[0].message
    payload = json.loads(log_message)

    assert payload["event"] == "detect_completed"
    assert payload["request_id"] == "test-request-id"
    assert payload["action"] == "block"
    assert payload["stored"] is True
    assert payload["latency_ms"] == 12.34
