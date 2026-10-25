"""HTTP helpers: a configured session and a GET with retries on transient errors only."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

import requests

log = logging.getLogger(__name__)


class HttpError(RuntimeError):
    """A request that could not be completed (after retries, or not retryable)."""


def make_session(user_agent: str) -> requests.Session:
    """One session per run: shared headers, and connection reuse to the same host."""
    session = requests.Session()
    session.headers["User-Agent"] = user_agent
    return session


def _is_retryable_status(status: int) -> bool:
    return status == 429 or status >= 500


def http_get(
    session: requests.Session,
    url: str,
    *,
    timeout_s: float,
    max_retries: int,
    backoff_base_s: float = 2.0,
    sleep: Callable[[float], None] = time.sleep,
) -> requests.Response:
    """GET with exponential backoff on *transient* failures: timeouts, dropped
    connections, 429 and 5xx. Deterministic failures fail immediately:
    certificate errors and malformed requests raise HttpError at once, and other
    statuses (200, 404, ...) are returned for the caller to interpret."""
    last_error = ""
    for attempt in range(1, max_retries + 1):
        try:
            response = session.get(url, timeout=timeout_s)
        except requests.exceptions.SSLError as exc:
            # Must come before ConnectionError, of which SSLError is a subclass.
            raise HttpError(
                f"TLS/certificate error for {url} (not retried, see README "
                f"'Troubleshooting'): {exc}"
            ) from exc
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as exc:
            last_error = str(exc)
        except requests.exceptions.RequestException as exc:
            raise HttpError(f"Request error for {url}: {exc}") from exc
        else:
            if not _is_retryable_status(response.status_code):
                return response
            last_error = f"HTTP {response.status_code}"

        if attempt < max_retries:
            wait = backoff_base_s**attempt
            log.warning("retry %d/%d in %.0fs (%s)", attempt, max_retries - 1, wait, last_error)
            sleep(wait)
    raise HttpError(f"GET {url} failed after {max_retries} attempts ({last_error})")
