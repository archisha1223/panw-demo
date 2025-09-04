from fastapi import FastAPI, Request, Body
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from datetime import date, timedelta
import io, csv
import pandas as pd
import numpy as np

from typing import List, Dict
from pydantic import BaseModel
from sklearn.linear_model import LinearRegression

import re
from collections import Counter

from sklearn.linear_model import TheilSenRegressor
from sklearn.ensemble import IsolationForest


app = FastAPI(title="Smart Financial Coach")

# Static & templates
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

# data for now 
FAKE_USER = {
    "name": "Archisha",
    "balance_cents": 1248659  
}

# Helpers
def cents_to_dollars(cents: int) -> str:
    return f"${cents/100:,.2f}"

class GoalInput(BaseModel):
    goal_amount: float      # e.g., 3000
    months: int             # e.g., 10
    monthly_income: float   # e.g., 5200

FEE_KEYWORDS = [
    "fee","service","processing","surcharge","convenience","foreign","intl","overdraft","nsf"
]

MIN_GROCERIES = 200.0

def _normalize_merchant(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def scan_subscriptions(df: pd.DataFrame):
    """Heuristic detector for subscriptions, trials->paid, and 'gray charges'."""
    if df.empty:
        return {"subscriptions": [], "trials": [], "gray": [], "summary": {}}

    data = df.copy()
    data["date"] = pd.to_datetime(data["date"])
    data["amount"] = data["amount"].astype(float)
    data["merchant_norm"] = data["merchant"].apply(_normalize_merchant)
    spend = data[data["amount"] < 0].sort_values("date").copy()

    subs, trials, grays = [], [], []

    for merchant, g in spend.groupby("merchant_norm"):
        g = g.sort_values("date")
        # excude rent
        dom_cat = g["category"].mode().iloc[0] if "category" in g and not g["category"].isna().all() else None
        if dom_cat == "Rent" or re.search(r"\b(landlord|rent|mortgage|lease|apartment|hoa)\b", merchant):
            continue

        amts = -g["amount"].values  # positive numbers
        dates = g["date"].values

        if len(g) >= 2:
            deltas = np.diff(dates).astype("timedelta64[D]").astype(int)
        else:
            deltas = np.array([])

        # cadence guess
        cadence = None
        if len(deltas):
            med = int(np.median(deltas))
            if 26 <= med <= 33: cadence = "Monthly (~30d)"
            elif 6 <= med <= 8: cadence = "Weekly (~7d)"
            elif 83 <= med <= 98: cadence = "Quarterly (~90d)"

        # subscription test: >=3 charges, stable amount, and a cadence
        amt_cv = (np.std(amts)/ (np.mean(amts) + 1e-6)) if len(amts) >= 2 else 1.0
        looks_like_sub = (len(g) >= 3) and (cadence is not None) and (amt_cv <= 0.25)

        # trial->paid: first small (<= $5) followed by larger within 7–30 days
        trial_flag = False
        if len(g) >= 2:
            first_small = amts[0] <= 5.0
            if first_small:
                # find first larger within 7–30 days
                for j in range(1, len(g)):
                    days = (g["date"].iloc[j] - g["date"].iloc[0]).days
                    if 7 <= days <= 30 and amts[j] >= 2 * max(1.0, amts[0]):
                        trial_flag = True
                        trial_amt = float(amts[j])
                        trial_date = g["date"].iloc[j]
                        break

        # gray charges: fee-like keywords or tiny repeated amounts
        descs = (g["description"].fillna("").str.lower() + " " + g["merchant"].fillna("").str.lower())
        is_fee = any(any(k in d for k in FEE_KEYWORDS) for d in descs)
        tiny_repeat = (np.mean(amts) < 6.0 and len(g) >= 2)

        if looks_like_sub:
            avg_amt = float(np.mean(amts))
            last_date = g["date"].iloc[-1]
            # estimate next bill date from cadence
            if "Monthly" in cadence:
                next_date = last_date + pd.DateOffset(days=30)
            elif "Weekly" in cadence:
                next_date = last_date + pd.DateOffset(days=7)
            elif "Quarterly" in cadence:
                next_date = last_date + pd.DateOffset(days=90)
            else:
                next_date = last_date + pd.DateOffset(days=int(np.median(deltas)) if len(deltas) else 30)

            subs.append({
                "merchant": g["merchant"].iloc[-1],     # show original casing from last txn
                "avg_monthly": round(avg_amt, 2),
                "occurrences": int(len(g)),
                "cadence": cadence or "Recurring",
                "first_seen": g["date"].iloc[0].strftime("%Y-%m-%d"),
                "last_seen": last_date.strftime("%Y-%m-%d"),
                "next_estimated": next_date.strftime("%Y-%m-%d"),
                "confidence": 0.9 if amt_cv < 0.15 else 0.75
            })

        if trial_flag:
            trials.append({
                "merchant": g["merchant"].iloc[-1],
                "trial_indicator": "Small first charge then upgrade",
                "upgrade_amount": round(trial_amt, 2),
                "upgrade_date": trial_date.strftime("%Y-%m-%d"),
                "confidence": 0.7
            })

        if is_fee or tiny_repeat:
            # avoid double-listing true subs; only include if not already flagged
            if not looks_like_sub:
                grays.append({
                    "merchant": g["merchant"].iloc[-1],
                    "avg_amount": round(float(np.mean(amts)), 2),
                    "count": int(len(g)),
                    "why": "Fee keyword" if is_fee else "Small repeated charge",
                    "last_seen": g["date"].iloc[-1].strftime("%Y-%m-%d")
                })

    # summary
    total_monthly = float(sum(s["avg_monthly"] for s in subs))
    summary = {
        "subscriptions_count": len(subs),
        "trials_count": len(trials),
        "gray_count": len(grays),
        "estimated_monthly_spend_on_subs": round(total_monthly, 2)
    }
    return {"subscriptions": subs, "trials": trials, "gray": grays, "summary": summary}

# Routes
@app.get("/")
def home(request: Request):
    balance_display = cents_to_dollars(FAKE_USER["balance_cents"])
    pages = [
        {"href": "/insights", "label": "AI-Powered Spending Insights"},
        {"href": "/goals", "label": "Personalized Goal Forecasting"},
        {"href": "/subscriptions", "label": "Subscriptions & Gray Charges"},
        {"href": "/credit", "label": "Credit Score Spending Tracker"},
    ]
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "name": FAKE_USER["name"],
            "balance": balance_display,
            "pages": pages
        }
    )

# INSIGHTS FEATURE 
@app.get("/insights")
def insights_page(request: Request):
    return templates.TemplateResponse("insights.html", {"request": request})

def synth_transactions(seed=7, months=12, daily_prob=0.45):
    rng = np.random.default_rng(seed)
    today = date.today().replace(day=1)
    start = (today.replace(day=1) - pd.DateOffset(months=months-1)).date()

    cats = {
        "Groceries": (25, 90),
        "Dining Out": (8, 32),
        "Coffee": (4, 12),
        "Transport": (10, 40),
        "Shopping": (15, 80),
        "Utilities": (80, 180),
        "Entertainment": (8, 40),
        "Rent": (2125, 3375)  # we will control this month-by-month
    }

    rows = []
    d = start
    end = today + pd.DateOffset(months=1)

    def month_start(dt): return dt.replace(day=1)

    while d < end.date():
        # stochastic daily
        if rng.random() < daily_prob:
            for _ in range(rng.integers(1, 4)):
                cat = rng.choice(["Groceries","Dining Out","Coffee","Transport","Shopping","Entertainment"])
                amt = -float(rng.integers(*cats[cat]))
                merch = {"Coffee":"Cafe","Groceries":"Market","Dining Out":"Restaurant",
                         "Transport":"Transit","Shopping":"Store","Entertainment":"Tickets"}[cat]
                rows.append([d.isoformat(), f"{cat} purchase", merch, amt, cat])

        d += timedelta(days=1)

    # rent
    # Build a month iterator that stops at the LAST day of the current month
    months_iter = pd.period_range(
        start=pd.to_datetime(start),
        end=end - pd.DateOffset(days=1),   # end-exclusive: prevents adding next month
        freq="M"
    )

    # Rent on the first day of each month in range
    for per in months_iter:
        m_start = per.to_timestamp()  # first day of that month
        rent_amt = -2125.0 if m_start < pd.Timestamp("2025-09-01") else -3375.0
        rows.append([m_start.date().isoformat(), "Monthly Rent", "Your Landlord", rent_amt, "Rent"])

    # Boost coffee for demo
    this_m = today.isoformat()
    coffees = [r for r in rows if r[0].startswith(this_m) and r[4]=="Coffee"]
    while len(coffees) < 6:
        rows.append([this_m, "Coffee purchase", "Cafe", -float(np.random.randint(8,14)), "Coffee"])
        coffees.append(rows[-1])
    
    months_iter = pd.period_range(start=pd.to_datetime(start), end=end - pd.DateOffset(days=1), freq="M")
    for per in months_iter:
        m_start = per.to_timestamp()  # month start (first day)
        # ATT phone bill (monthly)
        rows.append([m_start.date().isoformat(), "AT&T Wireless Bill", "AT&T", -85.00, "Utilities"])
        # Spotify (monthly)
        rows.append([m_start.date().isoformat(), "Spotify Premium", "Spotify", -9.99, "Entertainment"])
        # Amazon Prime (monthly)
        rows.append([m_start.date().isoformat(), "Amazon Prime Membership", "Amazon Prime", -14.99, "Entertainment"])

        # International calling fee from AT&T (gray fee — a few months only)
        if np.random.default_rng(seed + per.month).random() < 0.5:
            fee_day = (m_start + pd.DateOffset(days=15)).date().isoformat()
            rows.append([fee_day, "AT&T International Calling Fee", "AT&T", -7.49, "Utilities"])

    # HBO Max free trial → paid (7 days)
    # Put it in the most recent month so it shows up clearly
    recent_month_start = months_iter[-1].to_timestamp().date()
    trial_day = recent_month_start.isoformat()
    upgrade_day = (recent_month_start + timedelta(days=7)).isoformat()
    rows.append([trial_day, "HBO Max Trial", "HBO Max", -0.99, "Entertainment"])   # small first charge
    rows.append([upgrade_day, "HBO Max Subscription", "HBO Max", -15.99, "Entertainment"])

    return pd.DataFrame(rows, columns=["date","description","merchant","amount","category"])

def forecast_spend_ml(df: pd.DataFrame, horizon_months: int = 10):
    """
    Train a per-category LinearRegression on monthly spend and predict the next N months.
    Returns:
      pred_by_month: [{month:'YYYY-MM', total: float, by_cat:{cat:float,...}}, ...]
      avg_monthly_spend_pred: float
      by_cat_recent_avg: pd.Series (recent 3-mo avg per category)
    """
    tmp = df.copy()
    tmp["date"] = pd.to_datetime(tmp["date"])
    tmp["month"] = tmp["date"].dt.to_period("M").dt.to_timestamp()
    tmp["amount"] = tmp["amount"].astype(float)

    spend = tmp[tmp["amount"] < 0].copy()
    spend["spend"] = -spend["amount"]
    monthly = spend.groupby(["month","category"])["spend"].sum().reset_index()
    months_sorted = sorted(monthly["month"].unique())

    if not months_sorted:
        base_month = pd.Timestamp(date.today().replace(day=1))
        out = []
        for k in range(horizon_months):
            m = (base_month + pd.DateOffset(months=k)).strftime("%Y-%m")
            out.append({"month": m, "total": 0.0, "by_cat": {}})
        return out, 0.0, pd.Series(dtype=float)

    month_to_x = {m: i for i, m in enumerate(months_sorted)}
    cats = sorted(monthly["category"].unique())

    # Recent 3-month average 
    last_m = months_sorted[-1]
    last3_cutoff = last_m - pd.DateOffset(months=2)
    recent = monthly[monthly["month"] >= last3_cutoff]
    by_cat_recent_avg = (recent.groupby("category")["spend"].sum() / 3.0).sort_values(ascending=False)

    preds_by_cat = {c: [] for c in cats}
    for cat in cats:
        sub = monthly[monthly["category"] == cat].sort_values("month")
        xs = np.array([month_to_x[m] for m in sub["month"].values]).reshape(-1, 1)
        ys = sub["spend"].values.astype(float)

        # SPECIAL RULE: Rent should be constant at the latest observed value 
        if cat == "Rent" and len(ys) > 0:
            latest = float(ys[-1])         # e.g., 3375 after the Sept move
            yhat = np.full(horizon_months, latest)
            preds_by_cat[cat] = yhat.tolist()
            continue

        # Otherwise, use ML (or mean fallback)
        if len(ys) >= 3 and np.var(xs) > 0:
            model = LinearRegression()
            model.fit(xs, ys)
            start_x = len(months_sorted)
            future_x = np.arange(start_x, start_x + horizon_months).reshape(-1, 1)
            yhat = model.predict(future_x)
        else:
            yhat = np.full(horizon_months, ys.mean() if len(ys) else 0.0)

        yhat = np.clip(yhat, 0, None)      # no negative spend
        preds_by_cat[cat] = yhat.tolist()

    

    base_month = months_sorted[-1] + pd.offsets.MonthBegin(1)
    pred_by_month = []
    for k in range(horizon_months):
        m = (base_month + pd.DateOffset(months=k)).strftime("%Y-%m")
        by_cat = {cat: float(preds_by_cat[cat][k]) for cat in cats}
        total = float(sum(by_cat.values()))
        pred_by_month.append({"month": m, "total": round(total, 2), "by_cat": by_cat})

    avg_monthly_spend_pred = float(np.mean([p["total"] for p in pred_by_month])) if pred_by_month else 0.0
    return pred_by_month, avg_monthly_spend_pred, by_cat_recent_avg


def analyze(df: pd.DataFrame):
    df["date"] = pd.to_datetime(df["date"])
    df["amount"] = df["amount"].astype(float)  # negative spend
    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()

    agg = df.groupby(["month","category"])["amount"].sum().reset_index()
    agg["spend"] = -agg["amount"]

    current_month = df["month"].max()
    this_month = agg[agg["month"] == current_month].sort_values("spend", ascending=False)

    # color-coded bar 
    palette = {
        "Rent":"#8b5cf6", "Utilities":"#60a5fa", "Groceries":"#22c55e",
        "Dining Out":"#f59e0b", "Coffee":"#38bdf8", "Transport":"#a78bfa",
        "Shopping":"#ec4899", "Entertainment":"#eab308"
    }
    top_cats = this_month.head(6)
    top_dict = {
        "labels": top_cats["category"].tolist(),
        "values": [round(v,2) for v in top_cats["spend"].tolist()],
        "colors": [palette.get(c,"#94a3b8") for c in top_cats["category"].tolist()]
    }

    # focus category: highest non-Rent 
    non_rent = this_month[this_month["category"] != "Rent"]
    focus_cat = non_rent["category"].iloc[0] if not non_rent.empty else (top_cats["category"].iloc[0] if not top_cats.empty else "Groceries")
    trend_focus = agg[agg["category"]==focus_cat].sort_values("month")
    trend_out = {
        "labels": trend_focus["month"].dt.strftime("%Y-%m").tolist(),
        "values": [round(v,2) for v in trend_focus["spend"].tolist()]
    }

    # rent-shift detector 
    rent_shift = {"present": False, "labels": [], "values": []}
    rent_series = agg[agg["category"]=="Rent"].sort_values("month")
    if not rent_series.empty:
        vals = rent_series["spend"].values
        labels = rent_series["month"].dt.strftime("%Y-%m").tolist()
        if len(vals) >= 2:
            prev = np.median(vals[:-1])  # robust baseline
            last = vals[-1]
            if last >= prev * 1.25 or (last - prev) >= 500:  # big jump
                rent_shift = {
                    "present": True,
                    "labels": labels,
                    "values": [round(v,2) for v in vals],
                    "from": float(prev),
                    "to": float(last)
                }

    # ML: robust trend (Theil–Sen) and IsolationForest outlier detection
    insights = []
    anomalies = []  # will be filled by IsolationForest
    trends = []     # (cat, slope_ts)

    cat_features = []  # rows: one per category (for IF)
    cat_index    = []  # index -> category name

    for cat, sub in agg.groupby("category"):
        if cat == "Rent":
            continue

        ser = sub.sort_values("month")["spend"].values.astype(float)
        n = len(ser)
        if n < 4:
            continue

        # Robust slope on full history (less sensitive than polyfit)
        X = np.arange(n).reshape(-1, 1)
        ts = TheilSenRegressor(random_state=0)
        ts.fit(X, ser)
        slope_ts = float(ts.coef_[0])
        if slope_ts > 25:         # keep your old threshold
            trends.append((cat, slope_ts))

        # Build recent-window features for IsolationForest
        w = min(6, n)             # up to last 6 months
        recent = ser[-w:]
        mean_r = float(np.mean(recent[:-1])) if w >= 2 else float(np.mean(recent))
        std_r  = float(np.std(recent[:-1])) + 1e-6
        last   = float(recent[-1])
        ratio  = last / (mean_r + 1e-6)

        xs2 = np.arange(len(recent)).reshape(-1, 1)
        ts2 = TheilSenRegressor(random_state=0)
        try:
            ts2.fit(xs2, recent)
            slope_recent = float(ts2.coef_[0])
        except Exception:
            slope_recent = 0.0

        cat_features.append([last, mean_r, std_r, ratio, slope_recent])
        cat_index.append(cat)

    # IsolationForest across categories for THIS month
    if cat_features:
        Xf = np.array(cat_features)
        iso = IsolationForest(random_state=0, contamination=0.2, n_estimators=100)
        iso.fit(Xf)
        pred   = iso.predict(Xf)             # -1 = outlier
        scores = iso.decision_function(Xf)   # lower is "more unusual"

        # Turn outliers into anomaly tuples
        for cat, p, s, feats in zip(cat_index, pred, scores, cat_features):
            if p == -1:
                last, mean_r, std_r, ratio, slope_recent = feats
                anomalies.append((cat, last, -s, ratio))  # use -s so bigger = stronger

    
    # recent rising streak detector (exclude Rent)
    streak_hits = []
    for cat, sub in agg.groupby("category"):
        if cat == "Rent":
            continue
        series = sub.sort_values("month")["spend"].values
        if len(series) >= 5:
            last6 = series[-6:]  # focus on recent behavior
            diffs = np.diff(last6)
            increases = int((diffs > 0).sum())
            current = float(last6[-1])
            baseline = float(np.mean(last6[:-1]))  # recent avg (excluding current)
            pct_up = (current - baseline) / (baseline + 1e-6)

            # Rule: at least 3 increases in last 5 steps AND decent current level
            if increases >= 3 and current >= 300:
                severity = "High" if pct_up >= 0.30 else "Medium"
                streak_hits.append((cat, current, pct_up, increases, severity))

    # Convert streaks to insights (friendly, actionable)
    for cat, current, pct_up, increases, severity in streak_hits:
        tip = "Consider meal planning or setting a monthly cap." if cat == "Groceries" \
            else "Set a soft limit and review recent purchases."
        insights.append({
            "severity": severity,
            "icon": "🛒" if cat == "Groceries" else "📈",
            "message": (
                f"{cat} has rose {increases}/5 months and is now ${int(current)} "
                f"(~{int(pct_up*100)}% above your recent average). {tip}"
            ),
        })

    coffee_added = False

    # coffee nudge (dedup with anomalies: if coffee is anomalous => single HIGH message)
    coffee_spend = this_month[this_month["category"]=="Coffee"]["spend"]
    coffee_anom = next((a for a in anomalies if a[0]=="Coffee"), None)
    if not coffee_spend.empty:
        c = coffee_spend.iloc[0]
        yearly = int(c * 12)
        if coffee_anom:  # one strong message
            insights.append({"severity":"High","icon":"☕","message":f"Coffee spend spiked to ${int(c)} this month. Brewing at home 3×/week could save ≈ ${yearly:,}/yr."})
            anomalies = [a for a in anomalies if a[0]!="Coffee"]
            coffee_added = True
        elif c >= 80:
            insights.append({"severity":"Medium","icon":"☕","message":f"You've spent ${int(c)} on coffee this month. Try home-brew days; could save ≈ ${yearly:,}/yr."})
            coffee_added = True

    # If a category's current month is >25% and >$100 above its recent (last 3 mo) average,
    # add a gentle, non-judgmental suggestion. Skip anything already flagged.
    flagged_now = {m.get("category") for m in []}  # placeholder if you later add category to insights
    recent_cutoff = current_month - pd.DateOffset(months=2)
    recent3 = agg[agg["month"] >= recent_cutoff].groupby("category")["spend"].mean()

    tip_map = {
        "Dining Out": "Try 1–2 home-cooked meals this week.",
        "Shopping": "Pause impulse buys with a 24-hour rule.",
        "Transport": "Compare ride-hail vs transit for routine trips.",
        "Groceries": "Switch a few staples to store brands.",
        "Entertainment": "Look for free events or family plans.",
        "Utilities": "Check autopay/plan; reduce phantom usage."
    }

    for cat in this_month["category"]:
        if cat == "Rent" or (cat == "Coffee" and coffee_added): 
            continue
        # skip if already flagged by other rules
        if cat in {a[0] for a in anomalies} or cat in {t[0] for t in trends} or any("Groceries" in i.get("message","") for i in insights):
            pass  # still eligible for a nudge if not Coffee special-cased
        cur = float(this_month[this_month["category"]==cat]["spend"].iloc[0])
        base = float(recent3.get(cat, 0.0))
        if base <= 0:
            continue
        uplift = cur - base
        if cur >= 100 and uplift/base >= 0.25:  # >25% and >$100 higher than recent average
            tip = tip_map.get(cat, "Set a soft limit and review recent purchases.")
            insights.append({
                "severity": "Medium",
                "icon": "💡",
                "message": f"{cat} is ${int(uplift)} above your recent average (${int(base)} → ${int(cur)}). {tip}"
            })
    if anomalies:
            # anomalies → insights (IsolationForest)
            for a_cat, amount, strength, ratio in sorted(anomalies, key=lambda x: (-x[2], -x[3]))[:5]:
                if a_cat == "Coffee":   # coffee handled separately
                    continue
                # skip if already mentioned in streaks/trends/nudges
                if any(a_cat in i.get("message", "") for i in insights):
                    continue
                insights.append({
                    "severity": "High",
                    "icon": "🔎",
                    "message": f"{a_cat}: unusually high at ${int(amount)} "
                            f"(≈{int((ratio-1)*100)}% above recent). Consider a 10% cap next month."
                })


    # trends → insights
    for cat, slope in sorted(trends, key=lambda x: -x[1])[:5]:
        insights.append({"severity":"Medium","icon":"📈","message":f"{cat} is trending up by about ${int(slope)}/month. Set a soft limit or find cheaper swaps."})

    # rent shift insight
    if rent_shift["present"]:
        insights.insert(0, {
            "severity": "Info",           
            "icon": "🏠",
            "message": f"Rent change detected: ~${int(rent_shift['from'])} → ${int(rent_shift['to'])}. We’ll keep Rent out of ‘focus category’ trends."
        })

    # collect categories already flagged elsewhere
    flagged_cats = set()

    # red anomalies
    flagged_cats |= {a[0] for a in anomalies}

    # yellow linear trends
    flagged_cats |= {cat for cat, _ in trends}

    # rising-streak detector hits
    flagged_cats |= {cat for cat, *_ in streak_hits}

    # coffee message (either anomalous or medium nudge)
    if "Coffee" in agg["category"].unique():
        coffee_flagged = False
        if 'coffee_anom' in locals() and coffee_anom:
            coffee_flagged = True
        elif not coffee_spend.empty and coffee_spend.iloc[0] >= 80:
            coffee_flagged = True
        if coffee_flagged:
            flagged_cats.add("Coffee")

    # Rent is handled by the blue info pill only; never mark it stable
    flagged_cats.add("Rent")

    # stable categories summary 
    stable_cats = []
    for cat, sub in agg.groupby("category"):
        if cat == "Rent":  # skip rent
            continue
        if len(sub) >= 4:
            xs = np.arange(len(sub))
            ys = sub.sort_values("month")["spend"].values
            slope = np.polyfit(xs, ys, 1)[0]
            # "stable" = slope is small and category not flagged elsewhere
            if -20 <= slope <= 20 and cat not in flagged_cats:   # <-- use flagged_cats here
                stable_cats.append(cat)

    if stable_cats:
        cats_str = ", ".join(stable_cats)
        insights.append({
            "severity": "Low",   # green pill
            "icon": "✅",
            "message": f"Stable spending detected in: {cats_str}."
        })

    if not insights:
        insights.append({"severity":"Low","icon":"✅","message":"Spending looks stable this month. Nice work staying on track! 🎉"})

    sev_order = {"Info": 0, "High": 1, "Medium": 2, "Low": 3}
    insights.sort(key=lambda x: sev_order.get(x["severity"], 99))

    return {
        "insights": insights,
        "top_categories": top_dict,
        "focus_category": focus_cat,
        "trend": trend_out,
        "rent_shift": rent_shift
    }

DATA_CACHE = synth_transactions(seed=7, months=12)

def get_data():
    # return a copy so analyses don’t mutate the cache
    return DATA_CACHE.copy()

@app.get("/api/insights/sample")
def sample_insights():
    df = get_data()
    return analyze(df)

@app.post("/api/insights/analyze_csv")
def analyze_csv(raw: str = Body(..., media_type="text/plain")):
    # Expect CSV: date,description,merchant,amount,category
    reader = csv.DictReader(io.StringIO(raw))
    rows = list(reader)
    if not rows:
        return {"insights":[{"severity":"Low","message":"No rows found in CSV."}],
                "top_categories":{"labels":[],"values":[]},
                "focus_category":"N/A","trend":{"labels":[],"values":[]}}
    df = pd.DataFrame(rows)
    return analyze(df)

@app.post("/api/goals/forecast")
def forecast_goal(payload: GoalInput):
    # Use your synthetic transaction generator for history
    df = get_data()

     # this-month by category (to match Insights) 
    df2 = df.copy()
    df2["date"] = pd.to_datetime(df2["date"])
    df2["month"] = df2["date"].dt.to_period("M").dt.to_timestamp()
    df2["amount"] = df2["amount"].astype(float)
    spend2 = df2[df2["amount"] < 0].copy()
    spend2["spend"] = -spend2["amount"]
    last_m = spend2["month"].max()
    this_month_by_cat = (
        spend2[spend2["month"] == last_m]
        .groupby("category")["spend"].sum()
        .sort_values(ascending=False)
    )

    goal_total = float(payload.goal_amount)
    months = int(payload.months)
    income = float(payload.monthly_income)
    horizon = max(1, months)

    # ML forecast of future spend by month
    pred_by_month, avg_monthly_spend_pred, by_cat_recent_avg = forecast_spend_ml(df, horizon_months=horizon)

    # Baseline: income - predicted average spend
    baseline_net = income - avg_monthly_spend_pred
    baseline_total_save = baseline_net * months

    on_track = baseline_total_save >= goal_total
    gap_total = max(0.0, goal_total - baseline_total_save)
    gap_per_month = gap_total / months if months > 0 else 0.0

    # Suggestions: greedy cuts (exclude Rent), up to 20% per category
    reducible = by_cat_recent_avg[by_cat_recent_avg.index != "Rent"]
    suggestions: List[Dict] = []
    remaining = gap_per_month
    for cat, cat_monthly in reducible.items():
        if remaining <= 0: break
        max_cut = 0.20 * cat_monthly
        cut = min(max_cut, remaining)
        if cut > 5:
            pct = (cut / cat_monthly) if cat_monthly > 0 else 0
            suggestions.append({
                "category": cat,
                "current_per_month": round(cat_monthly, 2),
                "suggested_cut_per_month": round(cut, 2),
                "suggested_cut_percent": round(100*pct, 1)
            })
            remaining -= cut

    suggested_net = baseline_net + (gap_per_month - remaining if not on_track else 0.0)
    projected_total_with_cuts = suggested_net * months

    # build stacked-by-category data + dotted goal line threshold 
    months_list = [p["month"] for p in pred_by_month]
    # collect all cats that appear in predictions
    all_cats = sorted({k for p in pred_by_month for k in p["by_cat"].keys()})
    # matrix: one array per category, ordered by months_list
    stack_values = [[round(p["by_cat"].get(cat, 0.0), 2) for p in pred_by_month] for cat in all_cats]

    # dotted goal line = max affordable spend per month to still hit goal
    allowed_spend_per_month = income - (goal_total / months if months > 0 else 0.0)

    return {
        "inputs": {"goal_total": goal_total, "months": months, "monthly_income": income},
        "predicted_avg_monthly_spend": round(avg_monthly_spend_pred, 2),
        "baseline_net_per_month": round(baseline_net, 2),
        "baseline_total_save": round(baseline_total_save, 2),
        "on_track": on_track,
        "gap_total": round(gap_total, 2),
        "gap_per_month": round(gap_per_month, 2),
        "suggestions": suggestions,
        "projected_total_with_cuts": round(projected_total_with_cuts, 2),

        # single-line forecast (kept for compatibility)
        "forecast": {
            "months": months_list,
            "totals": [p["total"] for p in pred_by_month]
        },

        # NEW: stacked forecast by category + dotted goal line
        "allowed_spend_per_month": round(allowed_spend_per_month, 2),
        "forecast_by_cat": {
            "months": months_list,
            "cats": all_cats,
            "values": stack_values        # shape: [numCats][numMonths]
        },

        # bars for the sidebar
        "categories": {
            "labels": list(by_cat_recent_avg.index),
            "values": [round(v,2) for v in by_cat_recent_avg.values]
        },
        "categories_this_month": {
            "labels": list(this_month_by_cat.index),
            "values": [round(v,2) for v in this_month_by_cat.values]
        }
    }

# INSIGHTS FEATURE 

@app.get("/goals")
def goals_page(request: Request):
    return templates.TemplateResponse("goals.html", {"request": request})

@app.get("/api/subscriptions/scan")
def api_subs_scan():
    df = get_data()              
    return scan_subscriptions(df)

@app.get("/subscriptions")
def subscriptions_page(request: Request):
    return templates.TemplateResponse("subscriptions.html", {"request": request})

@app.get("/credit")
def credit(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "name": FAKE_USER["name"],
        "balance": cents_to_dollars(FAKE_USER["balance_cents"]),
        "pages": [],
        "placeholder": "Credit Score Spending Tracker (Soon)"
    })