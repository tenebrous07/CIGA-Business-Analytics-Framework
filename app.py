"""
CIGA - Customer Intent Gap Analyzer
====================================
Flask Web Application — API + Dashboard Server
Course: Big Data Computing for Business Analytics (MGT1062)
Team: Thejesh - 23MIA1033

Run:
    python app.py
Then open  http://127.0.0.1:8080
"""

import json
import logging
import os
import sys
import time
from functools import wraps

from flask import Flask, jsonify, render_template, request

# ── Import CIGA modules ───────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from data_loader import get_data
from intent_analyzer import CustomerIntentGapAnalyzer, print_analysis_report

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False

# ── Globals ───────────────────────────────────────────────────────────────────
_analyzer: CustomerIntentGapAnalyzer | None = None
_load_error: str = ""


# ── Bootstrap ─────────────────────────────────────────────────────────────────
def bootstrap():
    """Load data and build the analyzer once at startup."""
    global _analyzer, _load_error
    try:
        logger.info("Bootstrapping CIGA …")
        t0 = time.time()
        df = get_data()
        _analyzer = CustomerIntentGapAnalyzer(df)
        logger.info("CIGA ready in %.1f s", time.time() - t0)

        # Print structured console report
        print_analysis_report(_analyzer)
    except Exception as exc:
        _load_error = str(exc)
        logger.error("Bootstrap failed: %s", exc, exc_info=True)


# ── Helper: require analyzer ──────────────────────────────────────────────────
def require_analyzer(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if _analyzer is None:
            return jsonify({
                "error": _load_error or "Data not yet loaded. Run data_loader.py first."
            }), 503
        return fn(*args, **kwargs)
    return wrapper


# ── Simple JSON cache ─────────────────────────────────────────────────────────
_cache: dict = {}
CACHE_TTL = 300  # seconds

def cached(key: str):
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            now = time.time()
            if key in _cache and now - _cache[key]["ts"] < CACHE_TTL:
                return jsonify(_cache[key]["data"])
            result = fn(*args, **kwargs).get_json()
            _cache[key] = {"data": result, "ts": now}
            return jsonify(result)
        return wrapper
    return decorator


# ═══════════════════════════════════════════════════════════════════════════════
#  ROUTES — Pages
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({
        "status": "ok" if _analyzer else "loading",
        "error":  _load_error or None,
    })


# ═══════════════════════════════════════════════════════════════════════════════
#  ROUTES — API  (all prefixed /api/)
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/api/status")
def api_status():
    return jsonify({
        "ready":   _analyzer is not None,
        "error":   _load_error or None,
        "rows":    len(_analyzer.df) if _analyzer else 0,
    })


@app.route("/api/overview")
@require_analyzer
def api_overview():
    return jsonify(_analyzer.get_overview())


@app.route("/api/funnel")
@require_analyzer
def api_funnel():
    return jsonify(_analyzer.get_intent_funnel())


@app.route("/api/revenue-leakage")
@require_analyzer
def api_leakage():
    return jsonify(_analyzer.get_revenue_leakage())


@app.route("/api/high-intent-users")
@require_analyzer
def api_high_intent():
    return jsonify(_analyzer.get_high_intent_users())


@app.route("/api/failure-points")
@require_analyzer
def api_failures():
    return jsonify(_analyzer.get_failure_points())


@app.route("/api/alerts")
@require_analyzer
def api_alerts():
    return jsonify(_analyzer.get_alerts())


@app.route("/api/category-analysis")
@require_analyzer
def api_categories():
    return jsonify(_analyzer.get_category_analysis())


# ── Combined summary for the Overview page ────────────────────────────────────
@app.route("/api/all")
@require_analyzer
def api_all():
    """Return all data in a single call — used by dashboard on load."""
    return jsonify({
        "overview":        _analyzer.get_overview(),
        "funnel":          _analyzer.get_intent_funnel(),
        "revenue_leakage": _analyzer.get_revenue_leakage(),
        "high_intent":     _analyzer.get_high_intent_users(),
        "failure_points":  _analyzer.get_failure_points(),
        "alerts":          _analyzer.get_alerts(),
        "categories":      _analyzer.get_category_analysis(),
    })


# ── Reload data endpoint ──────────────────────────────────────────────────────
@app.route("/api/reload", methods=["POST"])
def api_reload():
    _cache.clear()
    bootstrap()
    return jsonify({"status": "reloaded"})


# ═══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    bootstrap()
    port = int(os.environ.get("PORT", 8080))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    logger.info("Starting CIGA server on http://127.0.0.1:%d", port)
    app.run(host="0.0.0.0", port=port, debug=debug)
