from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = REPO_ROOT / "test-results" / "e2e"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_streamlit(port: int, timeout_seconds: float = 30.0) -> None:
    health_url = f"http://127.0.0.1:{port}/_stcore/health"
    deadline = time.monotonic() + timeout_seconds
    last_error: Exception | None = None

    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(health_url, timeout=1.0) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
        time.sleep(0.25)

    raise RuntimeError(f"Streamlit did not become ready at {health_url}: {last_error}")


@pytest.fixture(scope="session")
def streamlit_server() -> str:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    port = _free_port()
    log_path = ARTIFACT_DIR / "streamlit.log"
    log_file = log_path.open("w", encoding="utf-8")

    env = os.environ.copy()
    env.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")

    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "app.py",
            "--server.headless=true",
            "--server.address=127.0.0.1",
            f"--server.port={port}",
            "--browser.gatherUsageStats=false",
            "--server.runOnSave=false",
        ],
        cwd=REPO_ROOT,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        env=env,
        text=True,
    )

    try:
        _wait_for_streamlit(port)
        yield f"http://127.0.0.1:{port}"
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        log_file.close()


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    setattr(item, f"rep_{report.when}", report)


@pytest.fixture
def e2e_page(page, context, request, streamlit_server):
    test_name = request.node.name.replace("/", "_")
    context.tracing.start(screenshots=True, snapshots=True, sources=True)
    page.goto(streamlit_server, wait_until="domcontentloaded")

    yield page

    failed = getattr(request.node, "rep_call", None) and request.node.rep_call.failed
    if failed:
        screenshot_path = ARTIFACT_DIR / f"{test_name}.png"
        trace_path = ARTIFACT_DIR / f"{test_name}.zip"
        page.screenshot(path=screenshot_path, full_page=True)
        context.tracing.stop(path=trace_path)
    else:
        context.tracing.stop()
