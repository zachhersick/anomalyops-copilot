import json
import logging
import re
from types import SimpleNamespace

import pytest

import copilot.observability as observability
from copilot.observability import (
    MAX_TRACE_STRING_LENGTH,
    TRACE_LOGGER_NAME,
    emit_trace,
    get_request_id,
    reset_request_id,
    resolve_request_id,
    sanitize_trace_attributes,
    set_request_id,
    trace_span,
)


def trace_payloads(caplog) -> list[dict[str, object]]:
    return [
        json.loads(record.getMessage())
        for record in caplog.records
        if record.name == TRACE_LOGGER_NAME
    ]


@pytest.mark.parametrize(
    "request_id",
    [
        "request-123",
        "abc_DEF.123",
        "service:request:42",
    ],
)
def test_resolve_request_id_preserves_valid_value(
    request_id: str,
):
    assert resolve_request_id(request_id) == request_id


@pytest.mark.parametrize(
    "request_id",
    [
        None,
        "",
        "contains spaces",
        "contains/slash",
        "x" * 129,
    ],
)
def test_resolve_request_id_generates_value_for_invalid_input(
    request_id: str | None,
    monkeypatch,
):
    monkeypatch.setattr(
        observability,
        "uuid4",
        lambda: SimpleNamespace(
            hex="generated-request-id"
        ),
    )

    assert resolve_request_id(request_id) == (
        "generated-request-id"
    )


def test_set_and_reset_request_id_context():
    assert get_request_id() is None

    token = set_request_id("request-123")

    try:
        assert get_request_id() == "request-123"
    finally:
        reset_request_id(token)

    assert get_request_id() is None


def test_sanitize_trace_attributes_redacts_sensitive_values():
    sanitized = sanitize_trace_attributes(
        {
            "api_key": "secret-key",
            "authorization": "Bearer secret",
            "database_url": "postgresql://secret",
            "password": "password",
            "token": "token",
            "query": "private question",
            "prompt": "private prompt",
            "input": "private input",
            "output": "private output",
            "arguments": '{"secret": true}',
            "tool_output": '{"secret": true}',
            "operation": "grounded_answer",
            "input_count": 2,
        }
    )

    assert sanitized == {
        "api_key": "[REDACTED]",
        "authorization": "[REDACTED]",
        "database_url": "[REDACTED]",
        "password": "[REDACTED]",
        "token": "[REDACTED]",
        "query": "[REDACTED]",
        "prompt": "[REDACTED]",
        "input": "[REDACTED]",
        "output": "[REDACTED]",
        "arguments": "[REDACTED]",
        "tool_output": "[REDACTED]",
        "operation": "grounded_answer",
        "input_count": 2,
    }


def test_sanitize_trace_attributes_preserves_scalar_metadata():
    sanitized = sanitize_trace_attributes(
        {
            "provider": "openai",
            "model": "gpt-test",
            "count": 3,
            "duration_ms": 12.5,
            "success": True,
            "optional": None,
        }
    )

    assert sanitized == {
        "provider": "openai",
        "model": "gpt-test",
        "count": 3,
        "duration_ms": 12.5,
        "success": True,
        "optional": None,
    }


def test_sanitize_trace_attributes_truncates_long_strings():
    value = "x" * (
        MAX_TRACE_STRING_LENGTH + 100
    )

    sanitized = sanitize_trace_attributes(
        {
            "safe_value": value,
        }
    )

    assert sanitized["safe_value"] == (
        "x" * MAX_TRACE_STRING_LENGTH
    )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ({"secret": "value"}, "<dict>"),
        ([1, 2, 3], "<list>"),
        (object(), "<object>"),
    ],
)
def test_sanitize_trace_attributes_does_not_serialize_complex_values(
    value: object,
    expected: str,
):
    sanitized = sanitize_trace_attributes(
        {
            "metadata": value,
        }
    )

    assert sanitized["metadata"] == expected


def test_emit_trace_includes_event_and_request_id(
    caplog,
):
    token = set_request_id("request-123")

    try:
        with caplog.at_level(
            logging.INFO,
            logger=TRACE_LOGGER_NAME,
        ):
            emit_trace(
                "provider.request",
                provider="openai",
                model="gpt-test",
            )
    finally:
        reset_request_id(token)

    payload = trace_payloads(caplog)[-1]

    assert payload == {
        "event": "provider.request",
        "model": "gpt-test",
        "provider": "openai",
        "request_id": "request-123",
    }


def test_emit_trace_redacts_secrets(
    caplog,
):
    token = set_request_id("request-123")

    try:
        with caplog.at_level(
            logging.INFO,
            logger=TRACE_LOGGER_NAME,
        ):
            emit_trace(
                "test.event",
                api_key="super-secret-key",
                query="private query",
            )
    finally:
        reset_request_id(token)

    record_message = caplog.records[-1].getMessage()
    payload = json.loads(record_message)

    assert payload["api_key"] == "[REDACTED]"
    assert payload["query"] == "[REDACTED]"
    assert "super-secret-key" not in record_message
    assert "private query" not in record_message


def test_trace_span_logs_success_and_duration(
    caplog,
    monkeypatch,
):
    times = iter([10.0, 10.125])

    monkeypatch.setattr(
        observability,
        "perf_counter",
        lambda: next(times),
    )

    token = set_request_id("request-123")

    try:
        with caplog.at_level(
            logging.INFO,
            logger=TRACE_LOGGER_NAME,
        ):
            with trace_span(
                "provider.request",
                provider="openai",
            ):
                pass
    finally:
        reset_request_id(token)

    payload = trace_payloads(caplog)[-1]

    assert payload["event"] == "provider.request"
    assert payload["provider"] == "openai"
    assert payload["request_id"] == "request-123"
    assert payload["status"] == "ok"
    assert payload["duration_ms"] == 125.0


def test_trace_span_logs_error_type_without_message(
    caplog,
    monkeypatch,
):
    times = iter([20.0, 20.05])

    monkeypatch.setattr(
        observability,
        "perf_counter",
        lambda: next(times),
    )

    token = set_request_id("request-123")

    try:
        with caplog.at_level(
            logging.INFO,
            logger=TRACE_LOGGER_NAME,
        ):
            with pytest.raises(
                RuntimeError,
                match="sensitive failure detail",
            ):
                with trace_span(
                    "provider.request",
                    provider="openai",
                ):
                    raise RuntimeError(
                        "sensitive failure detail"
                    )
    finally:
        reset_request_id(token)

    record_message = caplog.records[-1].getMessage()
    payload = json.loads(record_message)

    assert payload["status"] == "error"
    assert payload["error_type"] == "RuntimeError"
    assert payload["duration_ms"] == 50.0
    assert "sensitive failure detail" not in record_message


def test_generated_request_id_format():
    request_id = resolve_request_id(None)

    assert re.fullmatch(
        r"[0-9a-f]{32}",
        request_id,
    )