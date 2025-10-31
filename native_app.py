import os
import sys
import time
import signal
import subprocess
import threading
from contextlib import suppress


def _wait_for_server(url: str, timeout_seconds: int = 30) -> bool:
    import urllib.request

    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        with suppress(Exception):
            with urllib.request.urlopen(url, timeout=2) as resp:
                if 200 <= resp.getcode() < 500:
                    return True
        time.sleep(0.5)
    return False


def _start_streamlit(app_path: str, port: int) -> subprocess.Popen:
    env = os.environ.copy()
    env.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
    env.setdefault("STREAMLIT_SERVER_FILE_WATCHER_TYPE", "none")

    args = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        app_path,
        "--server.headless",
        "true",
        "--server.port",
        str(port),
        "--browser.serverAddress",
        "localhost",
        "--server.address",
        "127.0.0.1",
    ]

    return subprocess.Popen(
        args,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
        creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0,
    )


def main():
    if getattr(sys, "frozen", False):
        base_dir = os.path.dirname(sys.executable)
    else:
        base_dir = os.path.dirname(os.path.abspath(__file__))

    app_path = os.path.join(base_dir, "pc_early_detection.py")
    port = 8501
    url = f"http://127.0.0.1:{port}"

    proc = _start_streamlit(app_path, port)

    if not _wait_for_server(url, timeout_seconds=40):
        with suppress(Exception):
            proc.terminate()
        raise SystemExit("Streamlit server did not start.")

    import webview

    def _on_closed():
        # give streamlit a graceful shutdown
        with suppress(Exception):
            if os.name == "nt":
                proc.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                proc.terminate()
        # hard kill after a short delay
        def _force_kill():
            time.sleep(2)
            with suppress(Exception):
                proc.kill()
        threading.Thread(target=_force_kill, daemon=True).start()

    window = webview.create_window(
        title="Prostate Cancer Early Detection",
        url=url,
        width=1000,
        height=720,
        resizable=True,
        confirm_close=False,
        text_select=True,
    )
    webview.start(func=None, debug=False, http_server=False, gui=None, private_mode=False, on_closed=_on_closed)


if __name__ == "__main__":
    main()



