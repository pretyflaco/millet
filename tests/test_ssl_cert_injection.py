"""Tests for the F3 SSL_CERT_FILE injection in meet/__init__.py.

Reported by @peggy in the M8 retrospective: ``meet download <lang>``
failed with ``CERTIFICATE_VERIFY_FAILED`` on a python.org Python build
on macOS, because torchaudio's alignment-model fetcher uses raw urllib
which inherits the (empty) default SSL context. The fix injects
``certifi.where()`` as ``SSL_CERT_FILE`` at package import time, unless
the env var is already set.
"""

from __future__ import annotations

import os
import subprocess
import sys


def test_ssl_cert_file_is_set_after_import():
    """After importing meet, SSL_CERT_FILE points at an existing file.

    Uses a subprocess to get a clean import with no prior module state,
    avoiding test-ordering issues with sys.modules manipulation.
    """
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import millet, os, json; "
            "print(json.dumps({"
            "'set': 'SSL_CERT_FILE' in os.environ, "
            "'path': os.environ.get('SSL_CERT_FILE', ''), "
            "'exists': os.path.exists(os.environ.get('SSL_CERT_FILE', ''))}))",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        env={k: v for k, v in os.environ.items() if k != "SSL_CERT_FILE"},
    )
    assert result.returncode == 0, f"subprocess failed: {result.stderr}"
    import json

    data = json.loads(result.stdout.strip())
    assert data["set"], (
        "meet/__init__.py should inject SSL_CERT_FILE on import"
    )
    assert data["exists"], (
        f"SSL_CERT_FILE={data['path']!r} should point to an existing file "
        "(certifi's bundled CA store)"
    )


def test_ssl_cert_file_preserves_user_override(tmp_path):
    """If SSL_CERT_FILE is set before import, we do not overwrite it."""
    custom_cert = tmp_path / "custom-ca-bundle.pem"
    custom_cert.write_text("# placeholder CA bundle for the test\n")

    env = os.environ.copy()
    env["SSL_CERT_FILE"] = str(custom_cert)

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import millet, os; print(os.environ.get('SSL_CERT_FILE', ''))",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        env=env,
    )
    assert result.returncode == 0, f"subprocess failed: {result.stderr}"
    # The user's explicit choice must win.
    assert result.stdout.strip() == str(custom_cert)
