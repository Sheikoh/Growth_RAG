import pytest
import requests

from conftest import FakeResponse, FakeSession
from growth_rag.utils.http import HttpError, http_get

URL = "https://example.org/file.txt"


def get(session, max_retries=3):
    waits = []
    response = http_get(session, URL, timeout_s=1, max_retries=max_retries, sleep=waits.append)
    return response, waits


def test_retries_transient_errors_then_succeeds():
    session = FakeSession({URL: [FakeResponse(503), requests.exceptions.ConnectionError("reset"),
                                 FakeResponse(200, b"ok")]})
    response, waits = get(session)
    assert response.status_code == 200
    assert waits == [2.0, 4.0]  # exponential backoff
    assert len(session.calls) == 3


def test_returns_non_retryable_status_immediately():
    session = FakeSession({URL: FakeResponse(404)})
    response, waits = get(session)
    assert response.status_code == 404 and waits == [] and len(session.calls) == 1


def test_certificate_errors_are_not_retried():
    session = FakeSession({URL: requests.exceptions.SSLError("CERTIFICATE_VERIFY_FAILED")})
    with pytest.raises(HttpError, match="TLS/certificate"):
        get(session)
    assert len(session.calls) == 1


def test_gives_up_after_max_retries():
    session = FakeSession({URL: FakeResponse(500)})
    with pytest.raises(HttpError, match="after 3 attempts"):
        get(session)
    assert len(session.calls) == 3
