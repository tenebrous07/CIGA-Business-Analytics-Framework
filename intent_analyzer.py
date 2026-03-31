"""
CIGA - Customer Intent Gap Analyzer
====================================
Core Intent Analysis Engine
Course: Big Data Computing for Business Analytics (MGT1062)
Team: Thejesh - 23MIA1033

Steps:
  1. Collect & preprocess behavioural data  (data_loader.py)
  2. Detect intent signals                  ← this module
  3. Identify failure points
  4. Generate business output (funnel, leakage, recommendations)
"""

from __future__ import annotations
import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
#  CustomerIntentGapAnalyzer
# ═══════════════════════════════════════════════════════════════════════════════
class CustomerIntentGapAnalyzer:
    """
    Analyses ecommerce clickstream data to surface:
      • High-intent users who failed to convert (revenue leakage)
      • Where in the journey intent broke down
      • Actionable recovery recommendations
    """

    # Intent scoring weights
    WEIGHTS = dict(
        view=1,
        repeat_view=3,    # same product viewed 2+ times
        cart_add=5,
        cart_remove=2,
        session=2,
        comparison=3,     # multiple products same category
    )

    FAILURE_SCORE_THRESHOLD = 40  # 0–100 normalised intent score

    # ── Constructor ──────────────────────────────────────────────────────────
    def __init__(self, df: pd.DataFrame):
        self.df = df
        logger.info("CIGA initialised with %d events.", len(df))
        self._precompute()

    # ── Pre-computation (vectorised for performance) ─────────────────────────
    def _precompute(self) -> None:
        """Build all derived tables once so every API call is O(1)."""
        df = self.df

        self.views_df    = df[df["event_type"] == "view"].copy()
        self.cart_df     = df[df["event_type"] == "cart"].copy()
        self.purchase_df = df[df["event_type"] == "purchase"].copy()
        self.remove_df   = df[df["event_type"] == "remove_from_cart"].copy()

        # ── Per-user aggregate ───────────────────────────────────────────────
        user_agg = df.groupby("user_id").agg(
            total_views    =("event_type", lambda x: (x == "view").sum()),
            cart_adds      =("event_type", lambda x: (x == "cart").sum()),
            purchases      =("event_type", lambda x: (x == "purchase").sum()),
            sessions       =("user_session", "nunique"),
        ).reset_index()

        # Max times same product viewed per user
        rpt = (
            self.views_df
            .groupby(["user_id", "product_id"]).size()
            .reset_index(name="vc")
            .groupby("user_id")["vc"].max()
            .reset_index(name="max_repeat_views")
        )
        user_agg = user_agg.merge(rpt, on="user_id", how="left")
        user_agg["max_repeat_views"] = user_agg["max_repeat_views"].fillna(1)

        # Comparison shopping: distinct products in same category
        comp = (
            self.views_df
            .groupby(["user_id", "category_main"])["product_id"].nunique()
            .reset_index(name="distinct_products")
        )
        comp_flag = (
            comp[comp["distinct_products"] >= 3]
            .groupby("user_id").size()
            .reset_index(name="comparison_count")
        )
        user_agg = user_agg.merge(comp_flag, on="user_id", how="left")
        user_agg["comparison_count"] = user_agg["comparison_count"].fillna(0)

        # Cart value per user
        cv = (
            self.cart_df.groupby("user_id")["price"].sum()
            .reset_index(name="total_cart_value")
        )
        user_agg = user_agg.merge(cv, on="user_id", how="left")
        user_agg["total_cart_value"] = user_agg["total_cart_value"].fillna(0)

        # ── Intent score (raw) ───────────────────────────────────────────────
        W = self.WEIGHTS
        user_agg["intent_raw"] = (
            user_agg["total_views"]       * W["view"]      +
            user_agg["max_repeat_views"]  * W["repeat_view"] +
            user_agg["cart_adds"]         * W["cart_add"]  +
            user_agg["comparison_count"]  * W["comparison"]+
            user_agg["sessions"]          * W["session"]
        )
        max_raw = user_agg["intent_raw"].max()
        user_agg["intent_score"] = (
            (user_agg["intent_raw"] / max_raw * 100).round(1)
            if max_raw > 0 else 0
        )

        # ── User segment ─────────────────────────────────────────────────────
        def segment(row):
            if row["cart_adds"] > 0 and row["purchases"] == 0:
                return "Cart Abandoner"
            elif row["max_repeat_views"] >= 3 and row["purchases"] == 0:
                return "Comparison Shopper"
            elif row["total_views"] > 10 and row["purchases"] == 0:
                return "Window Shopper"
            elif row["sessions"] >= 3 and row["purchases"] == 0:
                return "Repeat Visitor"
            elif row["purchases"] > 0:
                return "Converter"
            return "Browser"

        user_agg["segment"] = user_agg.apply(segment, axis=1)

        self.user_stats = user_agg
        logger.info("Pre-computation complete. %d users scored.", len(user_agg))

    # ═══════════════════════════════════════════════════════════════════════
    #  API Methods (called by Flask routes)
    # ═══════════════════════════════════════════════════════════════════════

    # ── 1. Overview KPIs ─────────────────────────────────────────────────────
    def get_overview(self) -> Dict[str, Any]:
        df = self.df
        ev = df["event_type"].value_counts()
        views     = int(ev.get("view", 0))
        carts     = int(ev.get("cart", 0))
        purchases = int(ev.get("purchase", 0))

        view_to_cart   = round(carts / views * 100, 2)       if views  > 0 else 0
        cart_to_buy    = round(purchases / carts * 100, 2)   if carts  > 0 else 0
        overall_conv   = round(purchases / views * 100, 2)   if views  > 0 else 0

        total_rev   = round(float(self.purchase_df["price"].sum()), 2)
        avg_cart    = float(self.cart_df["price"].mean()) if len(self.cart_df) > 0 else 0
        leaked_rev  = round((carts - purchases) * avg_cart, 2)

        high_intent = int(
            (self.user_stats["intent_score"] >= self.FAILURE_SCORE_THRESHOLD).sum()
        )
        high_intent_no_buy = int(
            (
                (self.user_stats["intent_score"] >= self.FAILURE_SCORE_THRESHOLD) &
                (self.user_stats["purchases"] == 0)
            ).sum()
        )

        # Daily event trend (last 30 days)
        daily = (
            df.groupby(["date", "event_type"]).size()
            .unstack(fill_value=0).reset_index()
        )
        daily["date"] = daily["date"].astype(str)
        daily = daily.tail(30)

        return {
            "kpis": {
                "total_events":          int(len(df)),
                "total_users":           int(df["user_id"].nunique()),
                "total_sessions":        int(df["user_session"].nunique()),
                "total_products":        int(df["product_id"].nunique()),
                "views":                 views,
                "cart_adds":             carts,
                "purchases":             purchases,
                "view_to_cart_rate":     view_to_cart,
                "cart_to_buy_rate":      cart_to_buy,
                "overall_conversion":    overall_conv,
                "total_revenue":         total_rev,
                "revenue_leaked":        leaked_rev,
                "recovery_potential":    round(leaked_rev * 0.28, 2),
                "high_intent_users":     high_intent,
                "high_intent_no_buy":    high_intent_no_buy,
            },
            "daily_trend": {
                "dates":     daily["date"].tolist(),
                "views":     daily.get("view",    pd.Series(dtype=int)).tolist(),
                "carts":     daily.get("cart",    pd.Series(dtype=int)).tolist(),
                "purchases": daily.get("purchase",pd.Series(dtype=int)).tolist(),
            },
            "hourly_pattern": self._hourly_pattern(),
        }

    # ── 2. Intent Loss Funnel ────────────────────────────────────────────────
    def get_intent_funnel(self) -> Dict[str, Any]:
        ev = self.df["event_type"].value_counts()
        views     = int(ev.get("view", 0))
        carts     = int(ev.get("cart", 0))
        purchases = int(ev.get("purchase", 0))

        # High-intent cohort: users with intent_score ≥ threshold
        hi_users  = set(
            self.user_stats[
                self.user_stats["intent_score"] >= self.FAILURE_SCORE_THRESHOLD
            ]["user_id"]
        )
        hi_views  = int(self.views_df[self.views_df["user_id"].isin(hi_users)]["user_id"].nunique())
        hi_carts  = int(self.cart_df[self.cart_df["user_id"].isin(hi_users)]["user_id"].nunique())
        hi_buy    = int(self.purchase_df[self.purchase_df["user_id"].isin(hi_users)]["user_id"].nunique())

        def drop(a, b):
            return round((1 - b / a) * 100, 1) if a > 0 else 0

        return {
            "standard": {
                "stages": ["All Views", "Cart Adds", "Purchases"],
                "values": [views, carts, purchases],
                "drops":  [0, drop(views, carts), drop(carts, purchases)],
            },
            "intent_weighted": {
                "stages": ["High-Intent Viewers", "→ Added to Cart", "→ Purchased"],
                "values": [hi_views, hi_carts, hi_buy],
                "drops":  [0, drop(hi_views, hi_carts), drop(hi_carts, hi_buy)],
            },
            "intent_gap": {
                "non_converting_high_intent": hi_views - hi_buy,
                "gap_pct": round((hi_views - hi_buy) / hi_views * 100, 1) if hi_views > 0 else 0,
            },
        }

    # ── 3. Revenue Leakage ───────────────────────────────────────────────────
    def get_revenue_leakage(self) -> Dict[str, Any]:
        cart  = self.cart_df.copy()
        purch = self.purchase_df[["user_id", "product_id"]].copy()
        purch["_bought"] = True

        cart["_key"] = cart["user_id"] + "_" + cart["product_id"]
        purch["_key"] = purch["user_id"] + "_" + purch["product_id"]
        bought_keys   = set(purch["_key"])
        abandoned     = cart[~cart["_key"].isin(bought_keys)].copy()

        total_rev    = round(float(self.purchase_df["price"].sum()), 2)
        total_leaked = round(float(abandoned["price"].sum()), 2)
        leak_rate    = round(total_leaked / (total_rev + total_leaked) * 100, 1) \
                       if (total_rev + total_leaked) > 0 else 0

        # By category
        by_cat = (
            abandoned.groupby("category_main")
            .agg(abandoned_count=("product_id", "count"),
                 revenue_leaked =("price", "sum"),
                 avg_price      =("price", "mean"))
            .reset_index()
            .sort_values("revenue_leaked", ascending=False)
            .head(10)
            .round(2)
        )

        # By brand (top 10)
        by_brand = (
            abandoned.groupby("brand")
            .agg(abandoned_count=("product_id", "count"),
                 revenue_leaked =("price", "sum"))
            .reset_index()
            .sort_values("revenue_leaked", ascending=False)
            .head(10)
            .round(2)
        )

        # By price tier
        by_price = (
            abandoned.groupby("price_tier")
            .agg(abandoned_count=("product_id", "count"),
                 revenue_leaked =("price", "sum"))
            .reset_index()
            .round(2)
        )

        # Daily leakage trend
        abandoned["date"] = pd.to_datetime(abandoned["event_time"]).dt.date
        daily = (
            abandoned.groupby("date")["price"].sum()
            .reset_index()
            .tail(30)
        )
        daily["date"] = daily["date"].astype(str)

        return {
            "summary": {
                "total_revenue":     total_rev,
                "total_leaked":      total_leaked,
                "leakage_rate_pct":  leak_rate,
                "recoverable":       round(total_leaked * 0.28, 2),
            },
            "by_category":  by_cat.to_dict("records"),
            "by_brand":     by_brand.fillna("unknown").to_dict("records"),
            "by_price_tier":by_price.to_dict("records"),
            "daily_trend":  daily.rename(columns={"price": "revenue_leaked"})
                                 .to_dict("records"),
        }

    # ── 4. High-Intent Users ─────────────────────────────────────────────────
    def get_high_intent_users(self) -> Dict[str, Any]:
        us = self.user_stats
        hi = us[
            (us["intent_score"] >= self.FAILURE_SCORE_THRESHOLD) &
            (us["purchases"] == 0)
        ].sort_values("intent_score", ascending=False)

        seg_counts = us[us["purchases"] == 0]["segment"].value_counts().to_dict()

        # Score distribution (10 bins)
        hist, edges = np.histogram(us["intent_score"], bins=10, range=(0, 100))
        score_dist = {
            "labels": [f"{int(edges[i])}–{int(edges[i+1])}" for i in range(len(hist))],
            "values": hist.tolist(),
        }

        top_users = (
            hi.head(20)[
                ["user_id", "intent_score", "total_views",
                 "cart_adds", "sessions", "total_cart_value", "segment"]
            ]
            .round({"intent_score": 1, "total_cart_value": 2})
            .to_dict("records")
        )

        return {
            "summary": {
                "total_high_intent_no_buy": int(len(hi)),
                "avg_intent_score":         round(float(hi["intent_score"].mean()), 1),
                "potential_revenue":        round(float(hi["total_cart_value"].sum()), 2),
                "recoverable_revenue":      round(float(hi["total_cart_value"].sum()) * 0.28, 2),
            },
            "segments":         seg_counts,
            "score_distribution": score_dist,
            "top_users":        top_users,
        }

    # ── 5. Failure Points ────────────────────────────────────────────────────
    def get_failure_points(self) -> Dict[str, Any]:
        df = self.df
        us = self.user_stats
        non_buying_ids = set(us[us["purchases"] == 0]["user_id"])

        # Cart abandonment by category
        cart  = self.cart_df.copy()
        purch_keys = set(
            self.purchase_df["user_id"] + "_" + self.purchase_df["product_id"]
        )
        cart["_key"]  = cart["user_id"] + "_" + cart["product_id"]
        cart["_abandoned"] = ~cart["_key"].isin(purch_keys)

        aband_cat = (
            cart.groupby("category_main")
            .agg(total_carts    =("product_id", "count"),
                 abandoned_carts=("_abandoned", "sum"),
                 avg_price      =("price", "mean"))
            .reset_index()
        )
        aband_cat["abandonment_rate"] = (
            aband_cat["abandoned_carts"] / aband_cat["total_carts"] * 100
        ).round(1)
        aband_cat["revenue_at_risk"] = (
            aband_cat["abandoned_carts"] * aband_cat["avg_price"]
        ).round(2)
        aband_cat = aband_cat.sort_values("revenue_at_risk", ascending=False).head(10)

        # Price sensitivity: conversion rate by price tier
        price_conv = (
            df.groupby("price_tier")
            .agg(views    =("event_type", lambda x: (x == "view").sum()),
                 purchases=("event_type", lambda x: (x == "purchase").sum()))
            .reset_index()
        )
        price_conv["conv_rate"] = (
            price_conv["purchases"] / price_conv["views"].clip(lower=1) * 100
        ).round(2)

        # Failure reason cards
        n_price_sensitive = int(
            df[
                (df["event_type"] == "view") &
                (df["price"] > df["price"].quantile(0.75)) &
                (df["user_id"].isin(non_buying_ids))
            ]["user_id"].nunique()
        )
        n_cart_abandoners = int(us[
            (us["cart_adds"] > 0) & (us["purchases"] == 0)
        ].shape[0])
        n_comparison = int(us[
            (us["comparison_count"] > 0) & (us["purchases"] == 0)
        ].shape[0])
        n_ux_friction = int(
            df.groupby("user_session")
            .filter(lambda x: len(x) <= 2 and (x["event_type"] == "purchase").sum() == 0)
            ["user_id"].nunique()
        )

        failure_cards = [
            {
                "reason":       "Price Sensitivity",
                "description":  "Users repeatedly viewing high-priced items without converting",
                "affected":     n_price_sensitive,
                "severity":     "HIGH",
                "icon":         "💰",
                "action":       "Send targeted discount or instalment-plan offer",
            },
            {
                "reason":       "Cart Abandonment",
                "description":  "Users who added items to cart but never purchased",
                "affected":     n_cart_abandoners,
                "severity":     "CRITICAL",
                "icon":         "🛒",
                "action":       "Trigger abandoned-cart email within 1 hour",
            },
            {
                "reason":       "Comparison Friction",
                "description":  "Users browsing ≥3 products in same category without deciding",
                "affected":     n_comparison,
                "severity":     "MEDIUM",
                "icon":         "🔄",
                "action":       "Show side-by-side comparison widget with top choice",
            },
            {
                "reason":       "UX / Navigation Friction",
                "description":  "Sessions with ≤2 events and zero purchases (bounce)",
                "affected":     n_ux_friction,
                "severity":     "MEDIUM",
                "icon":         "⚠️",
                "action":       "A/B test landing page layout and search results quality",
            },
        ]

        return {
            "failure_cards":       failure_cards,
            "abandonment_by_cat":  aband_cat.round(2).to_dict("records"),
            "price_sensitivity":   price_conv.to_dict("records"),
        }

    # ── 6. Real-Time Alerts ──────────────────────────────────────────────────
    def get_alerts(self) -> List[Dict[str, Any]]:
        """Simulate real-time high-intent-about-to-drop alerts."""
        session_stats = (
            self.df.groupby("user_session")
            .agg(
                has_cart    =("event_type", lambda x: (x == "cart").any()),
                has_purchase=("event_type", lambda x: (x == "purchase").any()),
                event_count =("event_type", "count"),
                cart_value  =("price",
                              lambda x: x[self.df.loc[x.index, "event_type"] == "cart"].sum()),
                last_event  =("event_time", "max"),
                user_id     =("user_id", "first"),
                last_cat    =("category_main", "last"),
                last_brand  =("brand", "last"),
            )
            .reset_index()
        )

        at_risk = (
            session_stats[
                session_stats["has_cart"] &
                ~session_stats["has_purchase"] &
                (session_stats["cart_value"] > 0)
            ]
            .sort_values("cart_value", ascending=False)
            .head(15)
        )

        alerts = []
        for _, row in at_risk.iterrows():
            val   = float(row["cart_value"])
            level = "CRITICAL" if val > 200 else "HIGH" if val > 80 else "MEDIUM"
            recs  = {
                "CRITICAL": f"Flash 10% off — saves ≈${val*0.1:.0f}. Send push + email NOW.",
                "HIGH":     "Retargeting email with 5-star reviews for cart item.",
                "MEDIUM":   "Show free-shipping threshold or similar-product nudge.",
            }
            alerts.append({
                "user_id":        str(row["user_id"])[-8:],
                "session_id":     str(row["user_session"])[:14] + "…",
                "cart_value":     round(val, 2),
                "event_count":    int(row["event_count"]),
                "intent_level":   level,
                "category":       str(row["last_cat"]),
                "brand":          str(row["last_brand"]),
                "recommendation": recs[level],
                "last_activity":  str(row["last_event"])[:16],
            })
        return alerts

    # ── 7. Category Deep Dive ────────────────────────────────────────────────
    def get_category_analysis(self) -> List[Dict[str, Any]]:
        df = self.df
        cat = (
            df.groupby("category_main")
            .agg(
                views    =("event_type", lambda x: (x == "view").sum()),
                carts    =("event_type", lambda x: (x == "cart").sum()),
                purchases=("event_type", lambda x: (x == "purchase").sum()),
                revenue  =("price",
                           lambda x: x[df.loc[x.index, "event_type"] == "purchase"].sum()),
                users    =("user_id", "nunique"),
            )
            .reset_index()
        )
        cat["cart_rate"]    = (cat["carts"]     / cat["views"].clip(lower=1) * 100).round(2)
        cat["conv_rate"]    = (cat["purchases"] / cat["views"].clip(lower=1) * 100).round(2)
        cat["cart_conv"]    = (cat["purchases"] / cat["carts"].clip(lower=1) * 100).round(2)
        cat["avg_revenue"]  = (cat["revenue"]   / cat["purchases"].clip(lower=1)).round(2)
        return cat.sort_values("views", ascending=False).head(15).round(2).to_dict("records")

    # ── Private helpers ──────────────────────────────────────────────────────
    def _hourly_pattern(self) -> Dict[str, Any]:
        hourly = (
            self.df.groupby(["hour", "event_type"]).size()
            .unstack(fill_value=0).reset_index()
        )
        hours = list(range(24))
        def h(col):
            s = hourly.set_index("hour").reindex(hours, fill_value=0)
            return s[col].tolist() if col in s.columns else [0] * 24

        return {
            "hours":     hours,
            "views":     h("view"),
            "carts":     h("cart"),
            "purchases": h("purchase"),
        }


# ── Structured Console Report ─────────────────────────────────────────────────
def print_analysis_report(analyzer: CustomerIntentGapAnalyzer) -> None:
    """Print a full structured analysis report to stdout."""
    SEP = "─" * 62
    BIG = "═" * 62

    print(f"\n{BIG}")
    print("  CIGA — CUSTOMER INTENT GAP ANALYZER")
    print("  Structured Analysis Report")
    print(f"  Course: MGT1062  |  Team: Thejesh - 23MIA1033")
    print(BIG)

    # Overview
    ov = analyzer.get_overview()["kpis"]
    print(f"\n{SEP}")
    print("  STEP 1 — OVERVIEW METRICS")
    print(SEP)
    for k, v in ov.items():
        label = k.replace("_", " ").title()
        val   = f"${v:,.2f}" if "revenue" in k or "potential" in k else f"{v:,}"
        print(f"  {label:<35}: {val:>14}")

    # Intent Funnel
    fn = analyzer.get_intent_funnel()
    print(f"\n{SEP}")
    print("  STEP 2 — INTENT LOSS FUNNEL")
    print(SEP)
    for stage, val, drop in zip(
        fn["standard"]["stages"],
        fn["standard"]["values"],
        fn["standard"]["drops"],
    ):
        bar = "▓" * int(drop // 5) if drop > 0 else ""
        print(f"  {stage:<22}: {val:>10,}   drop {drop:5.1f}%  {bar}")
    print(f"\n  [Intent Gap] High-intent non-converters : "
          f"{fn['intent_gap']['non_converting_high_intent']:,}  "
          f"({fn['intent_gap']['gap_pct']}%)")

    # Revenue Leakage
    rl = analyzer.get_revenue_leakage()
    s  = rl["summary"]
    print(f"\n{SEP}")
    print("  STEP 3 — REVENUE LEAKAGE SUMMARY")
    print(SEP)
    print(f"  Revenue Earned          : ${s['total_revenue']:>14,.2f}")
    print(f"  Revenue Leaked          : ${s['total_leaked']:>14,.2f}")
    print(f"  Leakage Rate            : {s['leakage_rate_pct']:>14.1f}%")
    print(f"  Recoverable (28% est.)  : ${s['recoverable']:>14,.2f}")
    print(f"\n  Top 5 Leakage Categories:")
    for r in rl["by_category"][:5]:
        print(f"    {r['category_main']:<22}: ${r['revenue_leaked']:>10,.2f}  "
              f"({r['abandoned_count']} abandoned)")

    # High-Intent Users
    hi = analyzer.get_high_intent_users()
    print(f"\n{SEP}")
    print("  STEP 4 — HIGH-INTENT NON-CONVERTING USERS")
    print(SEP)
    s2 = hi["summary"]
    print(f"  Total High-Intent Users (no purchase) : {s2['total_high_intent_no_buy']:,}")
    print(f"  Average Intent Score                  : {s2['avg_intent_score']}")
    print(f"  Potential Revenue at Risk             : ${s2['potential_revenue']:,.2f}")
    print(f"  Recoverable Revenue (28%)             : ${s2['recoverable_revenue']:,.2f}")
    print(f"\n  User Segments:")
    for seg, cnt in hi["segments"].items():
        print(f"    {seg:<25}: {cnt:,}")

    # Failure Points
    fp = analyzer.get_failure_points()
    print(f"\n{SEP}")
    print("  STEP 5 — FAILURE POINT DIAGNOSIS")
    print(SEP)
    for card in fp["failure_cards"]:
        print(f"  {card['icon']}  [{card['severity']}] {card['reason']}")
        print(f"      Affected users : {card['affected']:,}")
        print(f"      Action         : {card['action']}")

    print(f"\n{BIG}\n")
