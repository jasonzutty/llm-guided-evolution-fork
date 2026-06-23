import atexit
import os
import shutil
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HOSTNAME_FILE = ROOT / "hostname.log"
PORT = int(os.getenv("LLMGE_SERVER_PORT", "8137"))
READY_TIMEOUT = int(os.getenv("LLM_SERVER_READY_TIMEOUT", "3600"))
READY_CHECK_INTERVAL = int(os.getenv("LLM_SERVER_READY_CHECK_INTERVAL", "10"))

_local_server_process = None


def _server_url(hostname):
    return f"http://{hostname}:{PORT}/"


def _hostname_from_file():
    try:
        hostname = HOSTNAME_FILE.read_text().strip()
    except OSError:
        return None
    return hostname or None


def _server_is_ready(hostname):
    if not hostname:
        return False
    try:
        with urllib.request.urlopen(_server_url(hostname), timeout=5) as response:
            return 200 <= response.status < 300
    except (TimeoutError, urllib.error.URLError, OSError):
        return False


def _wait_for_server(timeout=READY_TIMEOUT):
    start_time = time.time()
    last_hostname = None

    while time.time() - start_time <= timeout:
        if _local_server_process is not None and _local_server_process.poll() is not None:
            raise RuntimeError(
                "Local LLM test server exited before becoming ready "
                f"(exit code {_local_server_process.returncode})."
            )

        hostname = _hostname_from_file()
        if hostname:
            last_hostname = hostname
        if _server_is_ready(hostname):
            print(f"LLM test server is ready at {_server_url(hostname)}", flush=True)
            return True

        elapsed = round(time.time() - start_time)
        print(
            f"Waiting for LLM test server ({elapsed}s/{timeout}s). "
            f"Host file: {HOSTNAME_FILE}",
            flush=True,
        )
        time.sleep(READY_CHECK_INTERVAL)

    if last_hostname:
        print(f"Timed out waiting for LLM test server at {_server_url(last_hostname)}", flush=True)
    else:
        print(f"Timed out waiting for LLM test server; {HOSTNAME_FILE} was not written", flush=True)
    return False


def _start_slurm_server():
    print("Submitting LLM test server job with sbatch server.sh", flush=True)
    result = subprocess.run(
        [
            "sbatch",
            "--parsable",
            "--export=ALL,SUBMIT_ISLAND_CONTROLLER=0",
            "server.sh",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Failed to submit LLM server job with sbatch.\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    job_id = result.stdout.strip()
    print(f"Submitted LLM test server job {job_id}", flush=True)


def _start_local_server():
    global _local_server_process

    hostname = socket.gethostname()
    HOSTNAME_FILE.write_text(f"{hostname}\n")

    env = os.environ.copy()
    env["SERVER_HOSTNAME"] = hostname
    env.setdefault("CUDA_VISIBLE_DEVICES", "0,1")
    env.setdefault("UV_CACHE_DIR", f"/tmp/uv-cache-llmge-pytest-{os.getpid()}")

    print(f"Starting local LLM test server on {hostname}:{PORT}", flush=True)
    _local_server_process = subprocess.Popen(
        [
            "uv",
            "run",
            "uvicorn",
            "server:app",
            "--host",
            hostname,
            "--port",
            str(PORT),
            "--workers",
            "1",
        ],
        cwd=ROOT,
        env=env,
    )
    atexit.register(_stop_local_server)


def _stop_local_server():
    if _local_server_process is None or _local_server_process.poll() is not None:
        return
    _local_server_process.terminate()
    try:
        _local_server_process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        _local_server_process.kill()
        _local_server_process.wait()


def pytest_sessionstart(session):
    if os.getenv("LLMGE_AUTO_START_SERVER", "1") == "0":
        return

    hostname = _hostname_from_file()
    if _server_is_ready(hostname):
        print(f"Using existing LLM test server at {_server_url(hostname)}", flush=True)
        return

    if shutil.which("sbatch") and os.getenv("LLMGE_TEST_SERVER_MODE", "slurm") != "local":
        _start_slurm_server()
    else:
        _start_local_server()

    if not _wait_for_server():
        raise RuntimeError(
            "LLM test server did not become ready. "
            "Set LLMGE_AUTO_START_SERVER=0 to skip auto-starting it."
        )
