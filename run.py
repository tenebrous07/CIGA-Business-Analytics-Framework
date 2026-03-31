"""
CIGA — Customer Intent Gap Analyzer
====================================
Startup Script — Run this file to launch everything.
Course: Big Data Computing for Business Analytics (MGT1062)
Team: Thejesh - 23MIA1033

Usage:
    python run.py              # Download data, process, start web server
    python run.py --no-web     # Download & analyse only, no web server
    python run.py --force      # Force re-download even if cache exists
"""

import argparse
import logging
import os
import sys
import time
import webbrowser
import threading

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

BANNER = r"""
╔══════════════════════════════════════════════════════════════╗
║   ██████╗ ██╗ ██████╗  █████╗                               ║
║  ██╔════╝ ██║██╔════╝ ██╔══██╗                              ║
║  ██║      ██║██║  ███╗███████║                              ║
║  ██║      ██║██║   ██║██╔══██║                              ║
║  ╚██████╗ ██║╚██████╔╝██║  ██║                              ║
║   ╚═════╝ ╚═╝ ╚═════╝ ╚═╝  ╚═╝                              ║
║                                                              ║
║   Customer Intent Gap Analyzer                               ║
║   Course : MGT1062 — Big Data Computing for Business         ║
║   Team   : Thejesh · 23MIA1033                               ║
╚══════════════════════════════════════════════════════════════╝
"""

def check_deps() -> bool:
    """Verify required packages are installed."""
    required = ["flask", "pandas", "numpy", "kagglehub", "tqdm", "sklearn"]
    missing  = []
    for pkg in required:
        try:
            __import__(pkg.replace("-", "_"))
        except ImportError:
            missing.append(pkg)
    if missing:
        logger.error("Missing packages: %s", ", ".join(missing))
        logger.error("Run:  pip install -r requirements.txt")
        return False
    return True


def open_browser(port: int, delay: float = 2.0):
    """Open the dashboard in the default browser after a short delay."""
    def _open():
        time.sleep(delay)
        url = f"http://127.0.0.1:{port}"
        logger.info("Opening dashboard: %s", url)
        webbrowser.open(url)
    threading.Thread(target=_open, daemon=True).start()


def main():
    print(BANNER)
    parser = argparse.ArgumentParser(description="CIGA Launcher")
    parser.add_argument("--no-web",   action="store_true", help="Skip web server")
    parser.add_argument("--force",    action="store_true", help="Force re-download data")
    parser.add_argument("--port",     type=int, default=8080,  help="Web server port")
    parser.add_argument("--no-browser", action="store_true", help="Don't auto-open browser")
    args = parser.parse_args()

    # ── Dependency check ──────────────────────────────────────────────────
    if not check_deps():
        sys.exit(1)

    # ── Step 1: Data download & preprocessing ────────────────────────────
    logger.info("STEP 1 — Loading / downloading dataset …")
    from data_loader import get_data, print_structured_summary

    t0 = time.time()
    df = get_data(force_reload=args.force)
    logger.info("Data ready in %.1f s — %d rows", time.time() - t0, len(df))
    print_structured_summary(df)

    # ── Step 2: Intent analysis (console output) ─────────────────────────
    logger.info("STEP 2 — Running intent analysis …")
    from intent_analyzer import CustomerIntentGapAnalyzer, print_analysis_report
    analyzer = CustomerIntentGapAnalyzer(df)
    print_analysis_report(analyzer)

    # ── Step 3: Web server ────────────────────────────────────────────────
    if args.no_web:
        logger.info("--no-web flag set. Exiting without starting server.")
        return

    logger.info("STEP 3 — Starting CIGA web dashboard on port %d …", args.port)
    if not args.no_browser:
        open_browser(args.port, delay=2.5)

    # Import and run Flask app (bootstrap will reuse the already-loaded data)
    os.environ["CIGA_DF_LOADED"] = "1"          # flag so app.py skips reload
    from app import app, bootstrap as flask_boot
    flask_boot()   # uses cache so instant
    print(f"\n  ✓  Dashboard running at  http://127.0.0.1:{args.port}\n"
          f"  ✓  Press CTRL+C to stop\n")
    app.run(host="0.0.0.0", port=args.port, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
