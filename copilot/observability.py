import json
import logging
import re

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from time import perf_counter
from uuid import uuid4


TRACE_LOGGER_NAME = "anomalyops.trace"
MAX_TRACE_STRING_LENGTH = 256

_REQUEST_ID_PATTERN = re.compile(
    r"^[A-Za-z0-9._:-]{1,128}$"
)

_CONTENT_KEYS = {
    "arguments",
    "content",
    "context",
    "input",
    "output",
    "prompt",
    "query",
    "tool_output",
}

_SECRET_KEY_FRAGMENTS = {
    "api_key",
    "apikey",
    "authorization",
    "credential",
    "database_url",
    "password",
    "secret",
    "token",
}

_request_id_context: ContextVar[str | None] = ContextVar(
    "anomalyops_request_id",
    default=None,
)

logger = logging.getLogger(TRACE_LOGGER_NAME)


def resolve_request_id(
    supplied_request_id: str | None,
) -> str:
    if (
        supplied_request_id is not None
        and _REQUEST_ID_PATTERN.fullmatch(
            supplied_request_id
        )
    ):
        return supplied_request_id

    return uuid4().hex


def set_request_id(
    request_id: str,
) -> Token:
    return _request_id_context.set(request_id)


def reset_request_id(
    token: Token,
) -> None:
    _request_id_context.reset(token)


def get_request_id() -> str | None:
    return _request_id_context.get()


def _normalize_key(key: str) -> str:
    return re.sub(
        r"[^a-z0-9]+",
        "_",
        key.lower(),
    ).strip("_")


def _is_sensitive_key(key: str) -> bool:
    normalized = _normalize_key(key)

    if normalized in _CONTENT_KEYS:
        return True

    return any(
        fragment in normalized
        for fragment in _SECRET_KEY_FRAGMENTS
    )


def _sanitize_value(
    key: str,
    value: object,
) -> str | int | float | bool | None:
    if _is_sensitive_key(key):
        return "[REDACTED]"

    if value is None or isinstance(
        value,
        (int, float, bool),
    ):
        return value

    if isinstance(value, str):
        return value[:MAX_TRACE_STRING_LENGTH]

    return f"<{type(value).__name__}>"


def sanitize_trace_attributes(
    attributes: dict[str, object],
) -> dict[str, str | int | float | bool | None]:
    return {
        key: _sanitize_value(key, value)
        for key, value in attributes.items()
    }


def emit_trace(
    event: str,
    **attributes: object,
) -> None:
    payload = sanitize_trace_attributes(attributes)

    payload["event"] = event
    payload["request_id"] = get_request_id()

    logger.info(
        "%s",
        json.dumps(
            payload,
            sort_keys=True,
        ),
    )


@contextmanager
def trace_span(
    event: str,
    **attributes: object,
) -> Iterator[None]:
    started_at = perf_counter()

    try:
        yield
    except Exception as exc:
        emit_trace(
            event,
            **attributes,
            status="error",
            duration_ms=round(
                (
                    perf_counter()
                    - started_at
                )
                * 1000,
                3,
            ),
            error_type=type(exc).__name__,
        )
        raise
    else:
        emit_trace(
            event,
            **attributes,
            status="ok",
            duration_ms=round(
                (
                    perf_counter()
                    - started_at
                )
                * 1000,
                3,
            ),
        )