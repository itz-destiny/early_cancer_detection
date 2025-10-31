import os
import sys


def main():
    # Determine path to the Streamlit script next to the executable or this file
    if getattr(sys, "frozen", False):
        base_dir = os.path.dirname(sys.executable)
    else:
        base_dir = os.path.dirname(os.path.abspath(__file__))

    script_path = os.path.join(base_dir, "pc_early_detection.py")

    # Tweak Streamlit environment for packaged apps
    os.environ.setdefault("STREAMLIT_SERVER_FILE_WATCHER_TYPE", "none")
    os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")

    # Build args and invoke Streamlit runner programmatically
    from streamlit.web import cli as stcli

    sys.argv = [
        "streamlit",
        "run",
        script_path,
        "--server.headless=false",
    ]
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()



