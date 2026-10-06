"""Tests for the Tinfoil backend's transient-error retry (v0.9.2).

The Tinfoil SDK does a network fetch at client construction
(GET https://atc.tinfoil.sh/routers) and again for the completion call.
On hosts with flaky DNS, a single transient lookup failure used to
hard-fail confidential summarization.  _summarize_tinfoil now retries
transient network/DNS errors with backoff, while failing fast on real
auth/model errors.
"""
from __future__ import annotations

import socket
import sys
import types
import urllib.error

import pytest

import millet.summarize as sm
from millet.summarize import SummaryConfig, _is_transient_network_error

# ── classifier ──────────────────────────────────────────────────────────────


def test_classifier_flags_dns_gaierror():
    assert _is_transient_network_error(socket.gaierror(-2, "Name or service not known"))


def test_classifier_flags_sdk_wrapped_router_error():
    # The SDK wraps a URLError in ValueError("Failed to fetch router addresses…").
    inner = urllib.error.URLError("[Errno -2] Name or service not known")
    outer = ValueError(f"Failed to fetch router addresses: {inner}")
    outer.__cause__ = inner
    assert _is_transient_network_error(outer)


def test_classifier_rejects_auth_error():
    assert not _is_transient_network_error(RuntimeError("401 Unauthorized: bad api key"))
    assert not _is_transient_network_error(ValueError("model 'x' not found"))


# ── retry harness ────────────────────────────────────────────────────────────


def _install_fake_tinfoil(monkeypatch, behavior):
    """Install a fake ``tinfoil`` module whose TinfoilAI(...) runs ``behavior``
    (a callable taking the 1-based attempt number) and returns a fake
    completion response when it doesn't raise."""
    state = {"attempt": 0}

    class _Msg:
        content = "## Meeting Overview\n\nAll good."

    class _Choice:
        message = _Msg()

    class _Resp:
        choices = [_Choice()]

    class _Completions:
        def create(self, **kwargs):
            state["create_kwargs"] = kwargs
            return _Resp()

    class _Chat:
        completions = _Completions()

    class FakeTinfoilAI:
        def __init__(self, api_key=None):
            state["attempt"] += 1
            behavior(state["attempt"])  # may raise
            self.chat = _Chat()

    mod = types.ModuleType("tinfoil")
    mod.TinfoilAI = FakeTinfoilAI
    monkeypatch.setitem(sys.modules, "tinfoil", mod)
    # Ensure an API key is "present" so we reach the network path.
    monkeypatch.setattr(sm, "_resolve_tinfoil_api_key", lambda: "tk_fake")
    # _summarize_tinfoil does ``import time`` then time.sleep(...) for
    # backoff — patch the stdlib time.sleep so tests don't actually wait.
    import time as _t
    monkeypatch.setattr(_t, "sleep", lambda *_a, **_k: None)
    return state


def test_retries_transient_then_succeeds(monkeypatch):
    def behavior(attempt):
        if attempt == 1:
            # First client init: simulate the SDK router-discovery DNS failure.
            raise ValueError(
                "Failed to fetch router addresses: <urlopen error "
                "[Errno -2] Name or service not known>"
            )
        # Second attempt succeeds.

    state = _install_fake_tinfoil(monkeypatch, behavior)
    cfg = SummaryConfig(backend="tinfoil", model="glm-5-3-flash")
    result = sm._summarize_tinfoil("sys", "user", cfg)
    assert state["attempt"] == 2  # retried once
    assert "Meeting Overview" in result.markdown
    assert result.backend == "tinfoil"


def test_persistent_transient_fails_after_max_attempts(monkeypatch):
    def behavior(attempt):
        raise socket.gaierror(-2, "Name or service not known")

    state = _install_fake_tinfoil(monkeypatch, behavior)
    cfg = SummaryConfig(backend="tinfoil", model="glm-5-3-flash")
    with pytest.raises(RuntimeError, match="unreachable after"):
        sm._summarize_tinfoil("sys", "user", cfg)
    assert state["attempt"] == sm._TINFOIL_MAX_ATTEMPTS  # all attempts used


def test_completion_call_passes_timeout(monkeypatch):
    """Regression (0.13.0): the Tinfoil completion call was the only
    backend call without a timeout — a stalled TLS connection to the
    enclave hung the pipeline forever, worst for the no-fallback
    `confidential` preset."""
    state = _install_fake_tinfoil(monkeypatch, lambda attempt: None)
    cfg = SummaryConfig(backend="tinfoil", model="glm-5-3-flash")
    sm._summarize_tinfoil("sys", "user", cfg)
    assert state["create_kwargs"].get("timeout") == cfg.timeout
    assert cfg.timeout and cfg.timeout > 0


def test_auth_error_fails_fast_no_retry(monkeypatch):
    def behavior(attempt):
        raise RuntimeError("401 Unauthorized: invalid API key")

    state = _install_fake_tinfoil(monkeypatch, behavior)
    cfg = SummaryConfig(backend="tinfoil", model="glm-5-3-flash")
    with pytest.raises(RuntimeError, match="Tinfoil TEE API error"):
        sm._summarize_tinfoil("sys", "user", cfg)
    assert state["attempt"] == 1  # no retry on a real auth error


# ── hard wall-clock deadline (incident 2026-09-15) ──────────────────────────
#
# The SDK's custom httpx transport can block in ssl.read() with no effective
# timeout: millet's per-request timeout=600 never fired and a summary attempt
# pinned a vezir job in 'transcribing' for 30+ minutes.  Each attempt now runs
# under an external wall-clock deadline; a stalled attempt raises
# _TinfoilAttemptStuck (a TimeoutError subclass -> transient -> retried), and
# the blocked daemon thread is abandoned to die with the process.


def test_stuck_exception_is_transient_timeout():
    assert _is_transient_network_error(sm._TinfoilAttemptStuck("stalled"))


def test_deadline_helper_returns_result_and_propagates_errors():
    assert sm._run_under_deadline(lambda: 42, seconds=5) == 42
    with pytest.raises(ValueError, match="boom"):
        sm._run_under_deadline(lambda: (_ for _ in ()).throw(ValueError("boom")), seconds=5)


def test_deadline_helper_raises_stuck_on_expiry():
    import threading

    never = threading.Event()
    with pytest.raises(sm._TinfoilAttemptStuck, match="wall-clock deadline"):
        sm._run_under_deadline(lambda: never.wait(timeout=30), seconds=0.1)


def test_stalled_attempt_aborts_and_retries_then_fails_loud(monkeypatch):
    """A create() that never returns must not hang the pipeline: the
    attempt aborts at its deadline, the ladder retries, and after the full
    budget the job fails loudly instead of pinning 'transcribing' forever."""
    import threading

    calls = {"n": 0}
    never = threading.Event()  # never set: create() blocks "forever"

    class _Msg:
        content = "## Meeting Overview"

    class _Choice:
        message = _Msg()

    class _Resp:
        choices = [_Choice()]

    class _Completions:
        def create(self, **kwargs):
            calls["n"] += 1
            never.wait(timeout=30)  # abandoned thread gives up eventually
            return _Resp()

    class _Chat:
        completions = _Completions()

    class FakeTinfoilAI:
        def __init__(self, api_key=None):
            self.chat = _Chat()

    import sys as _sys
    import types as _types

    mod = _types.ModuleType("tinfoil")
    mod.TinfoilAI = FakeTinfoilAI
    monkeypatch.setitem(_sys.modules, "tinfoil", mod)
    monkeypatch.setattr(sm, "_resolve_tinfoil_api_key", lambda: "tk_fake")
    import time as _t

    monkeypatch.setattr(_t, "sleep", lambda *_a, **_k: None)
    monkeypatch.setattr(sm, "_TINFOIL_ATTEMPT_GRACE_SECONDS", 0.1)

    cfg = SummaryConfig(backend="tinfoil", model="glm-5-3-flash", timeout=0)
    with pytest.raises(RuntimeError, match="unreachable after"):
        sm._summarize_tinfoil("sys", "user", cfg)
    # Every attempt hit the wall-clock deadline (never the per-request one).
    assert calls["n"] == sm._TINFOIL_MAX_ATTEMPTS


def test_stall_on_first_attempt_only_still_succeeds(monkeypatch):
    """A one-off stall recovers: attempt 1 hits the deadline, attempt 2
    answers normally — the summary is produced, no error surfaces."""
    import threading

    state = {"n": 0}

    class _Msg:
        content = "## Meeting Overview\n\nRecovered."

    class _Choice:
        message = _Msg()

    class _Resp:
        choices = [_Choice()]

    class _Completions:
        def create(self, **kwargs):
            state["n"] += 1
            if state["n"] == 1:
                threading.Event().wait(timeout=30)
            return _Resp()

    class _Chat:
        completions = _Completions()

    class FakeTinfoilAI:
        def __init__(self, api_key=None):
            self.chat = _Chat()

    import sys as _sys
    import types as _types

    mod = _types.ModuleType("tinfoil")
    mod.TinfoilAI = FakeTinfoilAI
    monkeypatch.setitem(_sys.modules, "tinfoil", mod)
    monkeypatch.setattr(sm, "_resolve_tinfoil_api_key", lambda: "tk_fake")
    import time as _t

    monkeypatch.setattr(_t, "sleep", lambda *_a, **_k: None)
    monkeypatch.setattr(sm, "_TINFOIL_ATTEMPT_GRACE_SECONDS", 0.1)

    cfg = SummaryConfig(backend="tinfoil", model="glm-5-3-flash", timeout=0)
    result = sm._summarize_tinfoil("sys", "user", cfg)
    assert state["n"] == 2
    assert "Recovered" in result.markdown
    assert result.backend == "tinfoil"


# ── capacity (503) classification + sibling TEE fallback (v0.18.1) ───────────
#
# Tinfoil retired glm-5-2 with no notice: the model vanished from /v1/models
# and every request started returning 503 "engine is currently overloaded".
# That is not an auth error and not a DNS error, so it previously failed the
# job instantly — on the one preset that by design never falls back.

_OVERLOADED = (
    "Error code: 503 - {'error': {'message': 'The engine is currently "
    "overloaded, please try again later.', 'type': 'server_error'}}"
)
_NO_MODEL = (
    "Error code: 404 - {'error': {'message': 'The model does not exist.', "
    "'type': 'invalid_request_error'}}"
)


def test_classifier_flags_503_overloaded():
    assert _is_transient_network_error(RuntimeError(_OVERLOADED))


def test_classifier_rejects_404_missing_model():
    """A retired model will never come back — retrying it is pointless."""
    assert not _is_transient_network_error(RuntimeError(_NO_MODEL))


def test_pool_error_classifier_distinguishes_model_from_service():
    assert sm._is_model_pool_error(RuntimeError(_OVERLOADED))
    assert sm._is_model_pool_error(RuntimeError(_NO_MODEL))
    # Service-wide connectivity failure: a sibling model would fail too.
    assert not sm._is_model_pool_error(
        ValueError("Failed to fetch router addresses: <urlopen error>")
    )


def test_persistent_503_falls_back_to_sibling_tee_model(monkeypatch):
    """A drained pool on the primary must not kill the job: retry the
    sibling TEE model, which keeps the confidential contract intact."""
    seen: list[str] = []

    def behavior(attempt):
        pass  # client construction always succeeds

    _install_fake_tinfoil(monkeypatch, behavior)

    import sys as _sys

    fake = _sys.modules["tinfoil"]

    class _Msg:
        content = "## Meeting Overview\n\nFrom the sibling."

    class _Choice:
        message = _Msg()

    class _Resp:
        choices = [_Choice()]

    class _Completions:
        def create(self, **kwargs):
            seen.append(kwargs["model"])
            if kwargs["model"] == sm.DEFAULT_TINFOIL_MODEL:
                raise RuntimeError(_OVERLOADED)
            return _Resp()

    class _Chat:
        completions = _Completions()

    class FakeAI:
        def __init__(self, api_key=None):
            self.chat = _Chat()

    fake.TinfoilAI = FakeAI

    cfg = SummaryConfig(backend="tinfoil", model=sm.DEFAULT_TINFOIL_MODEL)
    result = sm._summarize_tinfoil("sys", "user", cfg)

    # Primary exhausted its retry budget, then the sibling was tried.
    assert seen.count(sm.DEFAULT_TINFOIL_MODEL) == sm._TINFOIL_MAX_ATTEMPTS
    assert sm.DEFAULT_TINFOIL_FALLBACK_MODEL in seen
    # Still a TEE model, and the switch is recorded — never silent.
    assert result.backend == "tinfoil"
    assert result.fallback_used is True
    assert sm.DEFAULT_TINFOIL_FALLBACK_MODEL in result.model
    assert "(TEE)" in result.model


def test_dns_failure_does_not_try_sibling(monkeypatch):
    """Router-discovery failure means Tinfoil itself is unreachable —
    a sibling model would fail identically, so don't waste the round trip."""

    def behavior(attempt):
        raise socket.gaierror(-2, "Name or service not known")

    state = _install_fake_tinfoil(monkeypatch, behavior)
    cfg = SummaryConfig(backend="tinfoil", model="glm-5-3-flash")
    with pytest.raises(RuntimeError, match="unreachable after"):
        sm._summarize_tinfoil("sys", "user", cfg)
    assert state["attempt"] == sm._TINFOIL_MAX_ATTEMPTS


def test_sibling_differs_from_primary_family():
    """The sibling must not share the primary's model family, or a single
    drained pool (exactly what killed glm-5-2) takes out both."""
    primary = sm.DEFAULT_TINFOIL_MODEL.split("-")[0]
    sibling = sm.DEFAULT_TINFOIL_FALLBACK_MODEL.split("-")[0]
    assert primary != sibling


# ── catalog pre-flight probe (v0.18.1) ──────────────────────────────────────


def _fake_catalog(monkeypatch, data, *, key="tk_fake"):
    monkeypatch.setattr(sm, "_resolve_tinfoil_api_key", lambda: key)
    sm._TINFOIL_CATALOG_CHECKED.clear()

    class _Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"data": data}

    monkeypatch.setattr(sm.requests, "get", lambda *a, **k: _Resp())


def test_probe_flags_retired_model(monkeypatch):
    _fake_catalog(monkeypatch, [{"id": "glm-5-3-flash"}])
    problem = sm.verify_tinfoil_model("glm-5-2")
    assert problem and "not in the catalog" in problem


def test_probe_flags_deprecated_model(monkeypatch):
    _fake_catalog(
        monkeypatch,
        [{"id": "deepseek-v4-flash", "deprecated": True, "deprecationDate": "2026-09-15"}],
    )
    problem = sm.verify_tinfoil_model("deepseek-v4-flash")
    assert problem and "2026-09-15" in problem


def test_probe_silent_for_healthy_model(monkeypatch):
    _fake_catalog(monkeypatch, [{"id": "glm-5-3-flash"}])
    assert sm.verify_tinfoil_model("glm-5-3-flash") is None


def test_probe_never_raises_on_network_failure(monkeypatch):
    monkeypatch.setattr(sm, "_resolve_tinfoil_api_key", lambda: "tk_fake")
    sm._TINFOIL_CATALOG_CHECKED.clear()

    def _boom(*a, **k):
        raise OSError("network down")

    monkeypatch.setattr(sm.requests, "get", _boom)
    # Advisory only: a probe failure must never block summarization.
    assert sm.verify_tinfoil_model("glm-5-3-flash") is None


def test_probe_checks_each_model_once(monkeypatch):
    calls = {"n": 0}
    monkeypatch.setattr(sm, "_resolve_tinfoil_api_key", lambda: "tk_fake")
    sm._TINFOIL_CATALOG_CHECKED.clear()

    class _Resp:
        def raise_for_status(self):
            pass

        def json(self):
            calls["n"] += 1
            return {"data": [{"id": "glm-5-3-flash"}]}

    monkeypatch.setattr(sm.requests, "get", lambda *a, **k: _Resp())
    sm.verify_tinfoil_model("glm-5-3-flash")
    sm.verify_tinfoil_model("glm-5-3-flash")
    assert calls["n"] == 1
