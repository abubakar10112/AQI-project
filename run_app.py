import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def start_process(command, name):
    """Launch a subprocess in a new console window on Windows."""
    if sys.platform.startswith("win"):
        return subprocess.Popen(
            command,
            cwd=str(ROOT),
            creationflags=subprocess.CREATE_NEW_CONSOLE,
            shell=False,
        )
    return subprocess.Popen(command, cwd=str(ROOT), shell=False)


def main():
    print("Starting AQI app stack...")
    backend = start_process([sys.executable, "app/flask_api.py"], "Flask API")
    time.sleep(2)
    dashboard = start_process([
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "app/streamlit_app.py",
        "--server.headless",
        "true",
        "--server.address",
        "0.0.0.0",
    ], "Streamlit Dashboard")

    print("Backend PID:", backend.pid)
    print("Dashboard PID:", dashboard.pid)
    print("API: http://127.0.0.1:5000/api/health")
    print("Dashboard: http://127.0.0.1:8501")
    print("Press Ctrl+C to stop both services.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping AQI app stack...")
        backend.terminate()
        dashboard.terminate()
        backend.wait(timeout=10)
        dashboard.wait(timeout=10)


if __name__ == "__main__":
    main()
