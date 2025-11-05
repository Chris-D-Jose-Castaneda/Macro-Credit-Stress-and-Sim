# app.py
# Macro–Credit Stressboard — Streamlit
# - Reads keys from .env
# - Flow: Macro/Credit → Stress → Markets → Macro-linked Option Scenarios
# - Robust to missing data/keys (no crashes)

from __future__ import annotations

import os
from math import erf, sqrt, exp
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

# Optional deps (guarded usage)
FRED_AVAILABLE = True
try:
    from fredapi import Fred
except Exception:
    FRED_AVAILABLE = False

YF_AVAILABLE = True
try:
    import yfinance as yf
except Exception:
    YF_AVAILABLE = False


# ──────────────────────────────────────────────────────────────────────────────
# Page / Keys
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Macro–Credit Stressboard", page_icon="📊", layout="wide")

load_dotenv()
FRED_API_KEY = os.getenv("FRED_API_KEY", "").strip()

st.title("📊 Macro–Credit Stressboard")
st.caption("Macro, credit, housing stress & option scenarios (no vendor lock-in).")
st.markdown(
    "Macro & credit conditions feed a **composite stress score**, which informs **market context** "
    "and a **macro-linked Monte Carlo option engine**. Adjust inputs and every chart responds."
)

fred = None
if FRED_AVAILABLE and FRED_API_KEY:
    fred = Fred(api_key=FRED_API_KEY)
elif FRED_AVAILABLE and not FRED_API_KEY:
    # fredapi requires a key; we continue without FRED features.
    fred = None


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def fetch_fred(series_ids: List[str], start: str, end: str) -> pd.DataFrame:
    if fred is None or not series_ids:
        return pd.DataFrame()
    frames = []
    for sid in series_ids:
        try:
            s = fred.get_series(sid, observation_start=start, observation_end=end)
            frames.append(pd.Series(s, name=sid))
        except Exception:
            # Missing series or key—skip but keep going
            continue
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, axis=1)
    df.index = pd.to_datetime(df.index)
    return df.sort_index()

@st.cache_data(ttl=1800)
def fetch_yf_prices(ticker_map: Dict[str, str], names: List[str], start: str, end: str, interval: str) -> pd.DataFrame:
    """
    ticker_map: display_name -> yfinance ticker
    names: list of display_names to fetch
    interval: '1d', '1wk', '1mo'
    """
    if not YF_AVAILABLE or not names:
        return pd.DataFrame()
    req = [ticker_map[n] for n in names if n in ticker_map]
    if not req:
        return pd.DataFrame()
    data = yf.download(req, start=start, end=end, interval=interval, auto_adjust=False, progress=False)
    if data is None or data.empty:
        return pd.DataFrame()
    # yfinance returns MultiIndex columns (OHLC etc.)
    if isinstance(data.columns, pd.MultiIndex):
        if ("Close" in data.columns.get_level_values(0)):
            close = data["Close"].copy()
        else:
            # Fallback to Adj Close if Close not present
            close = data["Adj Close"].copy() if ("Adj Close" in data.columns.get_level_values(0)) else data.iloc[:, 0:0]
    else:
        # Single ticker returns a Series or 1D DataFrame; harmonize to DataFrame
        close = data.to_frame(name=req[0]) if isinstance(data, pd.Series) else data

    close.index = pd.to_datetime(close.index)
    close = close.sort_index()

    # Map tickers -> display labels
    inv = {v: k for k, v in ticker_map.items()}
    close = close.rename(columns=inv)
    # Keep only requested display names (and in same order)
    keep = [n for n in names if n in close.columns]
    return close.loc[:, keep]

def resample(df: pd.DataFrame, freq: str, agg: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    if freq == "D":
        return df.copy()
    if agg == "Last":
        return df.resample(freq).last()
    return df.resample(freq).mean()

def idx100(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    base = df.iloc[0:1, :]
    base = base.replace(0, np.nan)
    out = (df / base.values) * 100.0
    return out

def yoy_pct(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    return df.pct_change(12) * 100.0

def mom_pct(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    return df.pct_change() * 100.0

def latest_delta(series: pd.Series) -> Tuple[float, float]:
    s = pd.Series(series).dropna()
    if s.shape[0] == 0:
        return (float("nan"), float("nan"))
    latest = float(s.iloc[-1])
    prev   = float(s.iloc[-2]) if s.shape[0] >= 2 else float("nan")
    return latest, (latest - prev) if not np.isnan(prev) else float("nan")

def zscore(s: pd.Series) -> pd.Series:
    x = s.dropna()
    if x.empty:
        return s * 0.0
    mu = x.mean()
    sd = x.std(ddof=0)
    sd = sd if sd and sd != 0 else 1.0
    return (s - mu) / sd

def norm_cdf(x: np.ndarray | float) -> np.ndarray | float:
    return 0.5 * (1.0 + erf(np.array(x) / sqrt(2.0)))

def gbm_paths(S0: float, r: float, sigma: float, days: int, n_paths: int, seed: int) -> np.ndarray:
    days = max(int(days), 1)
    n_paths = max(int(n_paths), 1)
    np.random.seed(int(seed))
    Z = np.random.standard_normal((days, n_paths))
    dt = 1.0 / 252.0
    drift = (r - 0.5 * sigma**2) * dt
    vol   = sigma * sqrt(dt)
    P = np.empty_like(Z, dtype=float)
    P[0, :] = S0 * np.exp(drift + vol * Z[0, :])
    for t in range(1, days):
        P[t, :] = P[t-1, :] * np.exp(drift + vol * Z[t, :])
    return P

def disc(x: float, r: float, T_years: float) -> float:
    return x * exp(-r * T_years)


# ──────────────────────────────────────────────────────────────────────────────
# Global Settings
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("## Global Settings")
gs1, gs2, gs3, gs4 = st.columns([1,1,1,1])
start_date = gs1.date_input("Start date", pd.to_datetime("2010-01-01")).strftime("%Y-%m-%d")
end_date   = gs2.date_input("End date",   pd.Timestamp.today()).strftime("%Y-%m-%d")
freq_choice = gs3.selectbox("Frequency", ["Daily", "Weekly (Fri)", "Monthly"], index=2)
agg_choice  = gs4.selectbox("Aggregation", ["Last", "Mean"], index=0)
freq_map = {"Daily": "D", "Weekly (Fri)": "W-FRI", "Monthly": "M"}
freq_code = freq_map[freq_choice]
yf_interval = {"D": "1d", "W-FRI": "1wk", "M": "1mo"}[freq_code]


# ──────────────────────────────────────────────────────────────────────────────
# Macro & Credit
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("## Macro & Credit")

mc1, mc2, mc3 = st.columns([2,1,1])
macro_pick = mc1.multiselect(
    "Select macro/credit series",
    ["Unemployment", "Mortgage Delinquency", "Policy Rate", "10Y Yield", "10Y–3M Spread", "HY OAS", "IG OAS", "30Y Mortgage Rate", "Volatility Index"],
    default=["Unemployment", "Policy Rate", "10Y–3M Spread", "HY OAS", "Volatility Index"]
)
macro_tf   = mc2.selectbox("Transform", ["Level", "MoM %", "YoY %"], index=0)
macro_idx  = mc3.checkbox("Index (if Level)", value=False)

FRED_SERIES: Dict[str, str] = {
    "Unemployment": "UNRATE",
    "Mortgage Delinquency": "DRSFRMACBS",
    "Policy Rate": "FEDFUNDS",
    "10Y Yield": "DGS10",
    "10Y–3M Spread": "T10Y3M",
    "HY OAS": "BAMLH0A0HYM2",
    "IG OAS": "BAMLC0A0CM",
    "30Y Mortgage Rate": "MORTGAGE30US",
    "Volatility Index": "VIXCLS",  # will fallback to yfinance ^VIX if FRED unavailable
}

fred_ids = [FRED_SERIES[n] for n in macro_pick if n in FRED_SERIES and n != "Volatility Index"]
fred_raw = fetch_fred(fred_ids, start_date, end_date) if fred_ids else pd.DataFrame()

# Volatility Index special-case: prefer FRED VIXCLS if available, else yfinance ^VIX
vix_fred = fetch_fred(["VIXCLS"], start_date, end_date) if ("Volatility Index" in macro_pick) else pd.DataFrame()
if vix_fred.empty and "Volatility Index" in macro_pick and YF_AVAILABLE:
    vix_yf = fetch_yf_prices({"Volatility Index": "^VIX"}, ["Volatility Index"], start_date, end_date, interval=yf_interval)
    if not vix_yf.empty:
        vix_yf.columns = ["VIXCLS"]  # harmonize name to FRED code
        vix_fred = vix_yf

macro_raw = fred_raw.copy()
if "Volatility Index" in macro_pick and not vix_fred.empty:
    macro_raw = macro_raw.join(vix_fred, how="outer")

macro_rs  = resample(macro_raw, freq_code, agg_choice)

if macro_tf == "Level":
    macro_df = macro_rs.copy()
    if macro_idx and not macro_df.empty:
        macro_df = idx100(macro_df.dropna())
else:
    base = macro_rs if freq_code == "M" else macro_rs.resample("M").last()
    macro_df = mom_pct(base) if macro_tf == "MoM %" else yoy_pct(base)

if not macro_df.empty:
    fig_macro = px.line(
        macro_df.dropna(how="all"),
        labels={"index": "Date", "value": "Indexed" if (macro_tf=="Level" and macro_idx) else macro_tf}
    )
    fig_macro.update_layout(height=420, hovermode="x unified", legend_title_text="")
    st.plotly_chart(fig_macro, use_container_width=True)
else:
    st.info("No macro data to plot. Add a FRED API key in `.env` or select series available via FRED; VIX will fallback to Yahoo Finance if possible.")


# ──────────────────────────────────────────────────────────────────────────────
# Stress Engine
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("## Stress Engine")

se1, se2, se3, se4 = st.columns(4)
w_credit = se1.slider("Weight: Credit (HY/IG OAS)", 0.0, 1.0, 0.35, 0.05)
w_curve  = se2.slider("Weight: Curve (−10Y–3M)",     0.0, 1.0, 0.15, 0.05)
w_vol    = se3.slider("Weight: Equity Vol (VIX)",    0.0, 1.0, 0.25, 0.05)
w_labor  = se4.slider("Weight: Labor (Unemployment)",0.0, 1.0, 0.25, 0.05)

need: Dict[str, pd.Series] = {}
if "BAMLH0A0HYM2" in macro_rs.columns: need["HY OAS"] = macro_rs["BAMLH0A0HYM2"]
if "BAMLC0A0CM"   in macro_rs.columns: need["IG OAS"] = macro_rs["BAMLC0A0CM"]
if "T10Y3M"       in macro_rs.columns: need["Curve Inversion"] = -macro_rs["T10Y3M"]
if "UNRATE"       in macro_rs.columns: need["Unemployment"] = macro_rs["UNRATE"]
if "VIXCLS"       in macro_rs.columns: need["Volatility"] = macro_rs["VIXCLS"]

panel = pd.DataFrame(need).dropna(how="all")
Z = panel.apply(zscore) if not panel.empty else pd.DataFrame()

weights = {
    "HY OAS": w_credit/2,
    "IG OAS": w_credit/2,
    "Curve Inversion": w_curve,
    "Volatility": w_vol,
    "Unemployment": w_labor,
}
common = [c for c in Z.columns if c in weights] if not Z.empty else []
composite = (Z[common] * pd.Series(weights)[common]).sum(axis=1) if common else pd.Series(dtype=float)

risk_z = float(composite.iloc[-1]) if composite.shape[0] else 0.0
risk_score = float(norm_cdf(risk_z) * 100.0)

g1, g2, g3 = st.columns(3)
g1.plotly_chart(
    go.Figure(go.Indicator(mode="gauge+number",
        value=risk_score,
        gauge={"axis":{"range":[0,100]},
               "steps":[{"range":[0,35],"color":"#2ca02c"},
                        {"range":[35,65],"color":"#ffbf00"},
                        {"range":[65,100],"color":"#d62728"}]},
        title={"text":"Composite Stress (0–100)"}
    )).update_layout(height=220, margin=dict(l=10,r=10,t=40,b=10)),
    use_container_width=True
)

vix_latest = float(panel["Volatility"].iloc[-1]) if "Volatility" in panel.columns and not panel.empty else float("nan")
g2.metric("Latest VIX", f"{vix_latest:.2f}" if not np.isnan(vix_latest) else "—")
g3.metric("Stress z", f"{risk_z:+.2f}")

# Persist for later sections
st.session_state["risk_z"] = risk_z
st.session_state["vix_series"] = panel["Volatility"] if "Volatility" in panel.columns else pd.Series(dtype=float)


# ──────────────────────────────────────────────────────────────────────────────
# Markets
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("## Markets")

ASSETS_YF: Dict[str, str] = {
    "MBS": "MBB",                         # iShares MBS ETF
    "High Yield Credit": "HYG",
    "Investment Grade Credit": "LQD",
    "10Y Treasury": "^TNX",               # 10y yield * 10; we'll scale later if shown
    "SPY": "SPY",
}

mk1, mk2, mk3 = st.columns([2,1,1])
asset_show = mk1.multiselect(
    "Select market series",
    list(ASSETS_YF.keys()),
    default=["MBS", "High Yield Credit", "Investment Grade Credit", "SPY"]
)
normalize_levels = mk2.checkbox("Index to 100", value=True)
use_log_y        = mk3.checkbox("Log scale", value=False)

markets_raw = fetch_yf_prices(ASSETS_YF, asset_show, start_date, end_date, interval=yf_interval)
if not markets_raw.empty and "10Y Treasury" in markets_raw.columns:
    # ^TNX is quoted in yield*10
    markets_raw["10Y Treasury"] = markets_raw["10Y Treasury"] / 10.0

mkts_rs = resample(markets_raw, freq_code, agg_choice)

if not mkts_rs.empty:
    st.markdown("#### Key Risk Pulse")
    kcols = st.columns(4)
    if "SPY" in mkts_rs:
        v, d = latest_delta(mkts_rs["SPY"])
        kcols[0].metric("SPY (close)", f"{v:.2f}" if not np.isnan(v) else "—",
                        f"{d:+.02f}" if not np.isnan(d) else "—")
    if "High Yield Credit" in mkts_rs:
        hv = mkts_rs["High Yield Credit"].pct_change().dropna().tail(20).mean() * 100 if mkts_rs["High Yield Credit"].shape[0] > 1 else np.nan
        kcols[1].metric("HY 1M avg %", f"{hv:.2f}%" if not np.isnan(hv) else "—")
    if "Investment Grade Credit" in mkts_rs:
        iv = mkts_rs["Investment Grade Credit"].pct_change().dropna().tail(20).mean() * 100 if mkts_rs["Investment Grade Credit"].shape[0] > 1 else np.nan
        kcols[2].metric("IG 1M avg %", f"{iv:.2f}%" if not np.isnan(iv) else "—")
    if "MBS" in mkts_rs:
        mv = mkts_rs["MBS"].pct_change().dropna().tail(20).mean() * 100 if mkts_rs["MBS"].shape[0] > 1 else np.nan
        kcols[3].metric("MBS 1M avg %", f"{mv:.2f}%" if not np.isnan(mv) else "—")

    m_level = idx100(mkts_rs) if normalize_levels else mkts_rs
    if not m_level.empty:
        fig_m = px.line(m_level.dropna(how="all"),
                        labels={"index":"Date","value": "Indexed" if normalize_levels else "Level"})
        fig_m.update_layout(height=420, hovermode="x unified", legend_title_text="")
        fig_m.update_yaxes(type="log" if use_log_y else "linear")
        st.plotly_chart(fig_m, use_container_width=True)
    else:
        st.info("Selected market series have no data after transformations.")
else:
    st.info("No market data to display. Ensure internet access for Yahoo Finance.")


# ──────────────────────────────────────────────────────────────────────────────
# Option Scenarios (Macro-linked)
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("## Option Scenarios (Macro-linked)")
st.markdown(
    "Call/put payoffs via **risk-neutral Monte Carlo**, with **volatility scaled by the live stress score**:\n"
    r"$\sigma_{\text{macro}} = \sigma_{\text{hist}} \cdot e^{\kappa_\sigma \cdot z}$"
)

if not mkts_rs.empty:
    os1, os2, os3, os4, os5 = st.columns(5)
    underlying_choices = [k for k in asset_show if k in mkts_rs.columns]
    underlying = os1.selectbox("Underlying (from Markets)", underlying_choices, index=0) if underlying_choices else None
    days_to_expiry = os2.slider("Days to expiry", 7, 365, 90, step=7)
    strike_pct     = os3.slider("Strike as % of spot", 50, 150, 100, step=5)
    rf_basis       = os4.selectbox("Rate basis", ["Policy Rate", "10Y Yield"], index=0)
    k_sigma        = os5.slider("Stress→σ sensitivity κσ", 0.00, 1.00, 0.35, 0.05)

    if underlying is not None:
        hist_u = mkts_rs[[underlying]].dropna()
        if not hist_u.empty:
            spot = float(hist_u.iloc[-1, 0])
            strike = spot * (strike_pct / 100.0)
            rets_u = np.log(hist_u[underlying]).diff().dropna()
            sigma_hist = float(rets_u.tail(252).std() * sqrt(252.0)) if not rets_u.empty else 0.20

            # Rate from FRED if possible; else fallback to 10y from markets if selected
            if rf_basis == "Policy Rate":
                rf_df = fetch_fred(["FEDFUNDS"], start_date, end_date)
            else:
                rf_df = fetch_fred(["DGS10"], start_date, end_date)
            rf_series = resample(rf_df, freq_code, agg_choice).dropna()
            if rf_series.empty and "10Y Treasury" in mkts_rs.columns:
                # as a last resort, derive a rough proxy from market chart (already in %)
                rf_series = mkts_rs[["10Y Treasury"]].copy()

            r = float(rf_series.iloc[-1, 0] / 100.0) if not rf_series.empty else 0.01

            # Macro-linked σ
            z = float(st.session_state.get("risk_z", 0.0))
            sigma_macro = float(sigma_hist * exp(k_sigma * z))

            sim1, sim2, sim3 = st.columns([1,1,1])
            n_paths  = sim1.slider("Number of paths (run)", 500, 20000, 5000, step=500)
            plot_n   = sim2.slider("Lines to plot", 10, 400, 100, step=10)
            seed_val = sim3.number_input("Random seed", value=42, step=1)

            paths = gbm_paths(spot, r, sigma_macro, days_to_expiry, n_paths, int(seed_val))
            T = days_to_expiry / 365.0
            call_payoff = np.maximum(paths[-1, :] - strike, 0.0)
            put_payoff  = np.maximum(strike - paths[-1, :], 0.0)
            mc_call = float(disc(np.mean(call_payoff), r, T))
            mc_put  = float(disc(np.mean(put_payoff),  r, T))

            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Spot", f"{spot:,.2f}")
            m2.metric("Strike", f"{strike:,.2f}")
            m3.metric("Rate (ann.)", f"{r*100:.2f}%")
            m4.metric("σ (hist → macro)", f"{sigma_hist*100:.2f}% → {sigma_macro*100:.2f}%")
            m5.metric("T (yrs)", f"{T:.3f}")

            c1, c2, c3 = st.columns(3)
            c1.metric("Scenario Call (MC)", f"{mc_call:,.2f}")
            c2.metric("Scenario Put (MC)",  f"{mc_put:,.2f}")
            c3.metric("Paths (run)", f"{n_paths:,}")

            # Path plot
            plot_n_eff = int(min(plot_n, paths.shape[1]))
            if plot_n_eff > 0:
                idx = np.linspace(0, paths.shape[1]-1, plot_n_eff, dtype=int)
                df_sim = pd.DataFrame(paths[:, idx], index=pd.date_range(pd.Timestamp.today().normalize(), periods=days_to_expiry, freq="B"))
                fig_sim = px.line(df_sim)
                fig_sim.update_layout(height=420, title=f"{underlying} — Macro-linked Monte Carlo Paths")
                st.plotly_chart(fig_sim, use_container_width=True)

            # 3D call surface
            st.markdown("#### 3D Scenario Surface (Call value)")
            k_grid = np.linspace(0.8, 1.2, 25)
            t_grid = np.linspace(7, 365, 24) / 365.0
            Zsurf = np.zeros((t_grid.size, k_grid.size))
            for i, T_ in enumerate(t_grid):
                inner = gbm_paths(spot, r, sigma_macro, int(max(1, round(T_ * 365))), 600, 123)
                for j, k_ in enumerate(k_grid):
                    K_ = spot * k_
                    payoff = np.maximum(inner[-1, :] - K_, 0.0)
                    Zsurf[i, j] = disc(np.mean(payoff), r, T_)
            fig_surf = go.Figure(data=[go.Surface(
                x=k_grid*100.0, y=t_grid*365.0, z=Zsurf, colorbar=dict(title="Call Value")
            )])
            fig_surf.update_layout(
                scene=dict(xaxis_title="Strike (% of Spot)", yaxis_title="Days to Expiry", zaxis_title="Value"),
                height=560
            )
            st.plotly_chart(fig_surf, use_container_width=True)
        else:
            st.info("Not enough history for the selected underlying.")
    else:
        st.info("Select at least one market series to drive scenarios.")
else:
    st.info("Load at least one market series to enable option scenarios.")


# ──────────────────────────────────────────────────────────────────────────────
# Correlation & Data Export
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("## Correlation & Data Export")

cm1, cm2 = st.columns([1,1])
corr_basis = cm1.selectbox("Correlation basis", ["Levels (indexed)", "1M changes (%)"], index=1)
show_preview = cm2.checkbox("Show data preview", value=True)

panel_all = pd.DataFrame()
if not mkts_rs.empty:
    panel_all = mkts_rs.add_prefix("M: ")
if not macro_rs.empty:
    panel_all = panel_all.join(macro_rs.add_prefix("F: "), how="outer") if not panel_all.empty else macro_rs.add_prefix("F: ")

panel_all_m = resample(panel_all, "M", "Last").dropna(how="all") if not panel_all.empty else pd.DataFrame()
if not panel_all_m.empty:
    X = idx100(panel_all_m.dropna()) if corr_basis == "Levels (indexed)" else (panel_all_m.pct_change() * 100.0)
    C = X.corr().replace([np.inf, -np.inf], np.nan).dropna(how="all").dropna(axis=1, how="all")
    if not C.empty:
        fig_hm = go.Figure(data=go.Heatmap(z=C.values, x=C.columns, y=C.index, zmin=-1, zmax=1, colorbar=dict(title="ρ")))
        fig_hm.update_layout(height=560)
        st.plotly_chart(fig_hm, use_container_width=True)
    else:
        st.info("Correlation matrix is empty after transformations.")
else:
    st.info("No combined panel to compute correlations.")

if show_preview and not panel_all.empty:
    st.markdown("#### Data (tail) & Export")
    st.dataframe(panel_all.tail(20))
    csv = panel_all.to_csv(index=True).encode("utf-8")
    st.download_button("⬇️ Download CSV", data=csv, file_name="macro_credit_panel.csv", mime="text/csv")
else:
    if show_preview:
        st.info("No data available to preview or export.")
