"""Normalize provider/SDK exceptions into a small, UI-friendly structure."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any


class ProviderFailureCategory(str, Enum):
    RATE_LIMIT = "rate_limit"
    OVERLOADED = "overloaded"
    AUTH = "auth"
    BAD_REQUEST = "bad_request"
    SERVER = "server"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"


def _parse_retry_after_header(raw: str | None) -> float | None:
    if not raw or not str(raw).strip():
        return None
    s = str(raw).strip()
    try:
        return float(s)
    except ValueError:
        return None


def _retry_after_from_response(response: Any) -> float | None:
    if response is None:
        return None
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    try:
        ra = headers.get("retry-after") or headers.get("Retry-After")
    except Exception:
        return None
    return _parse_retry_after_header(ra)


def _anthropic_error_type(body: object) -> str | None:
    if not isinstance(body, dict):
        return None
    err = body.get("error")
    if isinstance(err, dict):
        t = err.get("type")
        if isinstance(t, str):
            return t
    t2 = body.get("type")
    if isinstance(t2, str):
        return t2
    return None


def _one_line_from_body(body: object, *, max_len: int = 400) -> str:
    if body is None:
        return ""
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict):
            msg = err.get("message")
            if isinstance(msg, str) and msg.strip():
                return msg.strip()[:max_len]
        raw = json.dumps(body, default=str)[:max_len]
        return raw
    if isinstance(body, str):
        return body[:max_len]
    return str(body)[:max_len]


@dataclass(slots=True, frozen=True)
class ProviderFailure:
    """Unified, provider-agnostic description of a failed LLM call."""

    category: ProviderFailureCategory
    http_status: int | None
    retry_after_seconds: float | None
    public_message: str
    technical_detail: str


def _request_id_from_exc(exc: BaseException) -> str | None:
    rid = getattr(exc, "request_id", None)
    if isinstance(rid, str) and rid.strip():
        return rid.strip()
    response = getattr(exc, "response", None)
    if response is not None:
        h = getattr(response, "headers", None)
        if h is not None:
            try:
                for key in ("request-id", "x-request-id", "Request-Id"):
                    v = h.get(key)
                    if isinstance(v, str) and v.strip():
                        return v.strip()
            except Exception:
                pass
    return None


def infer_provider_failure(exc: BaseException) -> ProviderFailure | None:
    """Best-effort structured failure from a raw SDK/HTTP exception.

    Returns ``None`` when no useful structure is inferred (caller uses generic copy).
    """
    status: int | None = getattr(exc, "status_code", None)
    if not isinstance(status, int):
        response = getattr(exc, "response", None)
        if response is not None:
            sc = getattr(response, "status_code", None)
            if isinstance(sc, int):
                status = sc

    body = getattr(exc, "body", None)
    msg_lower = (str(exc) or "").lower()
    retry_after = None
    response = getattr(exc, "response", None)
    if response is not None:
        retry_after = _retry_after_from_response(response)

    # Anthropic connection / timeout (not APIStatusError)
    try:
        import anthropic

        if isinstance(exc, (anthropic.APITimeoutError, anthropic.APIConnectionError)):
            return ProviderFailure(
                category=ProviderFailureCategory.TIMEOUT,
                http_status=status,
                retry_after_seconds=None,
                public_message=(
                    "The request to the AI service timed out or the connection dropped. "
                    "Please try again."
                ),
                technical_detail=str(exc)[:2000],
            )
    except ImportError:
        pass

    # Anthropic APIStatusError (RateLimitError, BadRequestError, …)
    try:
        import anthropic

        if isinstance(exc, anthropic.APIStatusError):
            atype = _anthropic_error_type(body)
            rid = _request_id_from_exc(exc)
            tech = f"HTTP {status}"
            if atype:
                tech += f" · {atype}"
            line = _one_line_from_body(body)
            if line:
                tech += f" · {line[:300]}"
            if rid:
                tech += f" · request_id={rid}"

            if status == 429 or atype == "rate_limit_error":
                pub = (
                    "The AI service rate limit was reached (too many requests or input tokens "
                    "in a short time). The app may have retried automatically already."
                )
                if retry_after is not None:
                    secs = max(1, int(round(retry_after)))
                    pub += f" You can try again in about {secs} seconds."
                else:
                    pub += " Please wait a short time and try again."
                return ProviderFailure(
                    category=ProviderFailureCategory.RATE_LIMIT,
                    http_status=status,
                    retry_after_seconds=retry_after,
                    public_message=pub,
                    technical_detail=tech[:2000],
                )
            if status == 401:
                return ProviderFailure(
                    category=ProviderFailureCategory.AUTH,
                    http_status=status,
                    retry_after_seconds=None,
                    public_message="Authentication with the AI service failed. Check API keys or account access.",
                    technical_detail=tech[:2000],
                )
            if status == 403:
                return ProviderFailure(
                    category=ProviderFailureCategory.AUTH,
                    http_status=status,
                    retry_after_seconds=None,
                    public_message="The AI service refused this request (permission or billing).",
                    technical_detail=tech[:2000],
                )
            if status is not None and 500 <= status <= 599:
                return ProviderFailure(
                    category=ProviderFailureCategory.SERVER,
                    http_status=status,
                    retry_after_seconds=retry_after,
                    public_message=(
                        "The AI service had a temporary problem. Please try again in a moment."
                    ),
                    technical_detail=tech[:2000],
                )
            if status == 400:
                line = _one_line_from_body(body) or str(exc)[:300]
                return ProviderFailure(
                    category=ProviderFailureCategory.BAD_REQUEST,
                    http_status=status,
                    retry_after_seconds=None,
                    public_message=(
                        "The AI service rejected the request (invalid parameters or unsupported "
                        "options for this model)."
                    ),
                    technical_detail=(tech + (f" · {line}" if line else ""))[:2000],
                )
            return ProviderFailure(
                category=ProviderFailureCategory.UNKNOWN,
                http_status=status,
                retry_after_seconds=retry_after,
                public_message="The AI service returned an error. Please try again.",
                technical_detail=tech[:2000],
            )
    except ImportError:
        pass

    # OpenAI
    try:
        import openai

        if isinstance(exc, openai.RateLimitError):
            rid = _request_id_from_exc(exc)
            tech = f"HTTP {status or 429}"
            line = _one_line_from_body(body) if body is not None else str(exc)[:300]
            if line:
                tech += f" · {line[:300]}"
            if rid:
                tech += f" · request_id={rid}"
            pub = (
                "The AI service rate limit was reached (too many requests or tokens in a short time). "
                "The app may have retried automatically already."
            )
            if retry_after is not None:
                secs = max(1, int(round(retry_after)))
                pub += f" You can try again in about {secs} seconds."
            else:
                pub += " Please wait a short time and try again."
            return ProviderFailure(
                category=ProviderFailureCategory.RATE_LIMIT,
                http_status=status or 429,
                retry_after_seconds=retry_after,
                public_message=pub,
                technical_detail=tech[:2000],
            )
        if isinstance(exc, openai.AuthenticationError):
            return ProviderFailure(
                category=ProviderFailureCategory.AUTH,
                http_status=status or 401,
                retry_after_seconds=None,
                public_message="Authentication with the AI service failed. Check API keys.",
                technical_detail=str(exc)[:2000],
            )
        if isinstance(exc, openai.APIStatusError):
            tech = f"HTTP {status}" if status is not None else "API error"
            line = str(exc)[:400]
            if status == 429 or "rate" in msg_lower and "limit" in msg_lower:
                pub = (
                    "The AI service rate limit was reached. The app may have retried automatically already."
                )
                if retry_after is not None:
                    secs = max(1, int(round(retry_after)))
                    pub += f" You can try again in about {secs} seconds."
                else:
                    pub += " Please wait a short time and try again."
                return ProviderFailure(
                    category=ProviderFailureCategory.RATE_LIMIT,
                    http_status=status or 429,
                    retry_after_seconds=retry_after,
                    public_message=pub,
                    technical_detail=(tech + " · " + line)[:2000],
                )
            if status is not None and 500 <= status <= 599:
                return ProviderFailure(
                    category=ProviderFailureCategory.SERVER,
                    http_status=status,
                    retry_after_seconds=retry_after,
                    public_message="The AI service had a temporary problem. Please try again.",
                    technical_detail=(tech + " · " + line)[:2000],
                )
    except ImportError:
        pass

    # httpx
    try:
        import httpx

        if isinstance(exc, httpx.HTTPStatusError):
            resp = exc.response
            st = resp.status_code if resp is not None else status
            ra = _retry_after_from_response(resp) if resp is not None else None
            tech = f"HTTP {st}"
            if st == 429:
                pub = (
                    "The AI service rate limit was reached. The app may have retried automatically already."
                )
                if ra is not None:
                    secs = max(1, int(round(ra)))
                    pub += f" You can try again in about {secs} seconds."
                else:
                    pub += " Please wait a short time and try again."
                return ProviderFailure(
                    category=ProviderFailureCategory.RATE_LIMIT,
                    http_status=st,
                    retry_after_seconds=ra,
                    public_message=pub,
                    technical_detail=(tech + " · " + str(exc))[:2000],
                )
            if st is not None and 500 <= st <= 599:
                return ProviderFailure(
                    category=ProviderFailureCategory.SERVER,
                    http_status=st,
                    retry_after_seconds=ra,
                    public_message="The AI service had a temporary problem. Please try again.",
                    technical_detail=(tech + " · " + str(exc))[:2000],
                )
    except ImportError:
        pass

    # Heuristic fallback on message / status
    if status == 429:
        pub = (
            "The AI service rate limit was reached. The app may have retried automatically already. "
            "Please wait a short time and try again."
        )
        if retry_after is not None:
            secs = max(1, int(round(retry_after)))
            pub = (
                "The AI service rate limit was reached. The app may have retried automatically already. "
                f"You can try again in about {secs} seconds."
            )
        tech = (str(exc) or "").strip() or f"HTTP {status}"
        return ProviderFailure(
            category=ProviderFailureCategory.RATE_LIMIT,
            http_status=status,
            retry_after_seconds=retry_after,
            public_message=pub,
            technical_detail=tech[:2000],
        )
    if "rate limit" in msg_lower or "rate_limit" in msg_lower or "too many requests" in msg_lower:
        pub = (
            "The AI service rate limit was reached. The app may have retried automatically already. "
            "Please wait a short time and try again."
        )
        return ProviderFailure(
            category=ProviderFailureCategory.RATE_LIMIT,
            http_status=status,
            retry_after_seconds=retry_after,
            public_message=pub,
            technical_detail=str(exc)[:2000],
        )
    if "overloaded" in msg_lower or "capacity" in msg_lower:
        return ProviderFailure(
            category=ProviderFailureCategory.OVERLOADED,
            http_status=status,
            retry_after_seconds=retry_after,
            public_message="The AI service is busy. Please try again in a moment.",
            technical_detail=str(exc)[:2000],
        )
    if status is not None and 500 <= status <= 599:
        return ProviderFailure(
            category=ProviderFailureCategory.SERVER,
            http_status=status,
            retry_after_seconds=retry_after,
            public_message="The AI service had a temporary problem. Please try again.",
            technical_detail=str(exc)[:2000],
        )

    return None
