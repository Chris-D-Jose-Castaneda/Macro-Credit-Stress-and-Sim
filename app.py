# app.py
# Macro–Credit Stressboard — Streamlit 
# - Reads keys from .env
# - Cohesive flow: Macro/Credit → Stress → Markets → Macro-linked Option Scenarios
# - Parameters live above each section.

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
from fredapi import Fred
import eikon as ek


# Page / Keys (kept vendor-agnostic in the UI)
st.set_page_config(page_title="Macro–Credit Stressboard", page_icon="📊", layout="wide")

load_dotenv()
FRED_API_KEY = os.getenv("FRED_API_KEY", "")
EIKON_APP_KEY = os.getenv("EIKON_APP_KEY", "")

st.title("📊 Macro–Credit Stressboard")
st.caption("Macro, credit, housing stress & option scenarios.")
st.markdown(
    "One cohesive engine: macro & credit conditions feed a **stress score**, which drives **market context** "
    "and a **macro-linked scenario model** for options. Tune inputs and every chart reacts, including a 3D surface."
)

if FRED_API_KEY == "" or EIKON_APP_KEY == "":
    st.error("Required data-access keys are missing. Add them to your `.env` and restart.")
    st.stop()

fred = Fred(api_key=FRED_API_KEY)
ek.set_app_key(EIKON_APP_KEY)


# Helpers
@st.cache_data(ttl=3600)
def fetch_fred(series_ids: List[str], start: str, end: str) -> pd.DataFrame:
    frames = []
    for sid in series_ids:
        s = fred.get_series(sid, observation_start=start, observation_end=end)
        frames.append(pd.Series(s, name=sid))
    df = pd.concat(frames, axis=1)
    df.index = pd.to_datetime(df.index)
    return df.sort_index()

@st.cache_data(ttl=1800)
def fetch_market_timeseries(code_map: Dict[str, str], names: List[str], start: str, end: str, interval: str) -> pd.DataFrame:
    rics = [code_map[n] for n in names if n in code_map]
    df = ek.get_timeseries(rics, fields="CLOSE", start_date=start, end_date=end, interval=interval)
    if isinstance(df.columns, pd.MultiIndex) and "CLOSE" in df.columns.get_level_values(-1):
        df = df.xs("CLOSE", axis=1, level=-1)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    inv = {v: k for k, v in code_map.items()}
    df = df.rename(columns=inv)
    return df

def resample(df: pd.DataFrame, freq: str, agg: str) -> pd.DataFrame:
    if freq == "D":
        return df
    if agg == "Last":
        return df.resample(freq).last()
    return df.resample(freq).mean()

def idx100(df: pd.DataFrame) -> pd.DataFrame:
    base = df.iloc[0:1, :].values
    return (df / base) * 100.0

def yoy_pct(df: pd.DataFrame) -> pd.DataFrame:
    return df.pct_change(12) * 100.0

def mom_pct(df: pd.DataFrame) -> pd.DataFrame:
    return df.pct_change() * 100.0

def latest_delta(series: pd.Series) -> Tuple[float, float]:
    s = series.dropna()
    latest = float(s.iloc[-1])
    prev   = float(s.iloc[-2]) if s.shape[0] >= 2 else np.nan
    return latest, latest - prev

def zscore(s: pd.Series) -> pd.Series:
    x = s.dropna()
    mu = x.mean()
    sd = x.std(ddof=0)
    return (s - mu) / (sd if sd != 0 else 1.0)

# Normal CDF (no SciPy)
def norm_cdf(x: np.ndarray | float) -> np.ndarray | float:
    return 0.5 * (1.0 + erf(np.array(x) / sqrt(2.0)))

# GBM paths (risk-neutral drift r; σ may be macro-linked)
def gbm_paths(S0: float, r: float, sigma: float, days: int, n_paths: int, seed: int) -> np.ndarray:
    np.random.seed(int(seed))
    Z = np.random.standard_normal((days, n_paths))
    dt = 1.0 / 252.0
    P = np.zeros_like(Z)
    drift = (r - 0.5 * sigma**2) * dt
    vol   = sigma * sqrt(dt)
    P[0, :] = S0 * np.exp(drift + vol * Z[0, :])
    for t in range(1, days):
        P[t, :] = P[t-1, :] * np.exp(drift + vol * Z[t, :])
    return P

def disc(x: float, r: float, T_years: float) -> float:
    return x * exp(-r * T_years)



# GLOBAL SETTINGS
st.markdown("## Global Settings")
gs1, gs2, gs3, gs4 = st.columns([1,1,1,1])
start_date = gs1.date_input("Start date", pd.to_datetime("2010-01-01")).strftime("%Y-%m-%d")
end_date   = gs2.date_input("End date",   pd.Timestamp.today()).strftime("%Y-%m-%d")
freq_choice = gs3.selectbox("Frequency", ["Daily", "Weekly (Fri)", "Monthly"], index=2)
agg_choice  = gs4.selectbox("Aggregation", ["Last", "Mean"], index=0)
freq_map = {"Daily": "D", "Weekly (Fri)": "W-FRI", "Monthly": "M"}
freq_code = freq_map[freq_choice]


# MACRO & CREDIT (inputs to the stress engine)
st.markdown("## Macro & Credit")

mc1, mc2, mc3 = st.columns([2,1,1])
macro_pick = mc1.multiselect(
    "Select macro/credit series",
    ["Unemployment", "Mortgage Delinquency", "Policy Rate", "10Y Yield", "10Y–3M Spread", "HY OAS", "IG OAS", "30Y Mortgage Rate", "Volatility Index"],
    default=["Unemployment", "Mortgage Delinquency", "Policy Rate", "10Y–3M Spread", "HY OAS", "Volatility Index"]
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
    "Volatility Index": "VIXCLS",     # <- FRED VIX (no Eikon permission needed)
}
fred_raw = fetch_fred([FRED_SERIES[n] for n in macro_pick], start_date, end_date)
fred_rs  = resample(fred_raw, freq_code, agg_choice)

if macro_tf == "Level":
    macro_df = fred_rs.copy()
    if macro_idx:
        macro_df = idx100(macro_df.dropna())
else:
    base = fred_rs if freq_code == "M" else fred_rs.resample("M").last()
    macro_df = mom_pct(base) if macro_tf == "MoM %" else yoy_pct(base)

fig_macro = px.line(macro_df.dropna(how="all"),
                    labels={"index": "Date", "value": "Indexed" if (macro_tf=="Level" and macro_idx) else macro_tf})
fig_macro.update_layout(height=420, hovermode="x unified", legend_title_text="")
st.plotly_chart(fig_macro, use_container_width=True)


# STRESS ENGINE (single scalar that drives scenarios)
st.markdown("## Stress Engine")

se1, se2, se3, se4 = st.columns(4)
w_credit = se1.slider("Weight: Credit (HY/IG OAS)", 0.0, 1.0, 0.35, 0.05)
w_curve  = se2.slider("Weight: Curve (−10Y–3M)",     0.0, 1.0, 0.15, 0.05)
w_vol    = se3.slider("Weight: Equity Vol (VIX)",    0.0, 1.0, 0.25, 0.05)
w_labor  = se4.slider("Weight: Labor (Unemployment)",0.0, 1.0, 0.25, 0.05)

need: Dict[str, pd.Series] = {}
if "BAMLH0A0HYM2" in fred_rs: need["HY OAS"] = fred_rs["BAMLH0A0HYM2"]
if "BAMLC0A0CM"   in fred_rs: need["IG OAS"] = fred_rs["BAMLC0A0CM"]
if "T10Y3M"       in fred_rs: need["Curve Inversion"] = -fred_rs["T10Y3M"]
if "UNRATE"       in fred_rs: need["Unemployment"] = fred_rs["UNRATE"]

# Volatility from FRED VIX (no Eikon .VIX call)
vix_df = fetch_fred(["VIXCLS"], start_date, end_date)
vix_rs = resample(vix_df, freq_code, agg_choice)["VIXCLS"]
need["Volatility"] = vix_rs

panel = pd.DataFrame(need).dropna(how="all")
Z = panel.apply(zscore)

weights = {
    "HY OAS": w_credit/2,
    "IG OAS": w_credit/2,
    "Curve Inversion": w_curve,
    "Volatility": w_vol,
    "Unemployment": w_labor,
}
common = [c for c in Z.columns if c in weights]
composite = (Z[common] * pd.Series(weights)[common]).sum(axis=1)

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
g2.metric("Latest VIX (FRED)", f"{float(vix_rs.iloc[-1]):.2f}")
g3.metric("Stress z", f"{risk_z:+.2f}")

# Persist for later sections
st.session_state["risk_z"] = risk_z
st.session_state["vix_series"] = vix_rs



# MARKETS (context feeds the scenario engine) 
st.markdown("## Markets")

ASSETS: Dict[str, str] = {
    "MBS": "MBB.O",
    "High Yield Credit": "HYG",
    "Investment Grade Credit": "LQD",
    "10Y Treasury Yield": "US10YT=RR",
    "SPY": "SPY",
}
int_map = {"D": "daily", "W-FRI": "weekly", "M": "monthly"}

mk1, mk2, mk3 = st.columns([2,1,1])
asset_show = mk1.multiselect(
    "Select market series",
    list(ASSETS.keys()),
    default=["MBS", "High Yield Credit", "Investment Grade Credit", "SPY"]
)
normalize_levels = mk2.checkbox("Index to 100", value=True)
use_log_y        = mk3.checkbox("Log scale", value=False)

markets_raw = fetch_market_timeseries(ASSETS, asset_show, start_date, end_date, interval=int_map[freq_code])
mkts_rs = resample(markets_raw, freq_code, agg_choice)
m_level = idx100(mkts_rs) if normalize_levels else mkts_rs

st.markdown("#### Key Risk Pulse")
kcols = st.columns(4)
if "SPY" in mkts_rs:
    v, d = latest_delta(mkts_rs["SPY"])
    kcols[0].metric("SPY (close)", f"{v:.2f}", f"{d:+.02f}")
if "High Yield Credit" in mkts_rs:
    hv = mkts_rs["High Yield Credit"].pct_change().dropna().tail(20).mean() * 100
    kcols[1].metric("HY 1M avg %", f"{hv:.2f}%")
if "Investment Grade Credit" in mkts_rs:
    iv = mkts_rs["Investment Grade Credit"].pct_change().dropna().tail(20).mean() * 100
    kcols[2].metric("IG 1M avg %", f"{iv:.2f}%")
if "MBS" in mkts_rs:
    mv = mkts_rs["MBS"].pct_change().dropna().tail(20).mean() * 100
    kcols[3].metric("MBS 1M avg %", f"{mv:.2f}%")

fig_m = px.line(m_level.dropna(how="all"),
                labels={"index":"Date","value": "Indexed" if normalize_levels else "Level"})
fig_m.update_layout(height=420, hovermode="x unified", legend_title_text="")
fig_m.update_yaxes(type="log" if use_log_y else "linear")
st.plotly_chart(fig_m, use_container_width=True)


# OPTION SCENARIOS (Macro-linked
st.markdown("## Option Scenarios (Macro-linked)")
st.markdown(
    "We value simple call/put payoffs by **risk-neutral Monte Carlo**, with **volatility scaled by the live stress score**. "
    "This ties the option paths directly to macro/credit conditions."
)
st.markdown(
    r"""
**Engine formulas.**  
- **Volatility link:** $\sigma_{\text{macro}} = \sigma_{\text{hist}} \cdot \exp(\kappa_\sigma \cdot z)$ where $z$ is the composite stress score.  
- **GBM under risk-neutral:** $S_{t+\Delta} = S_t \exp\!\big((r-\tfrac{1}{2}\sigma^2)\Delta + \sigma \sqrt{\Delta}\,\varepsilon\big)$.
"""
)

os1, os2, os3, os4, os5 = st.columns(5)
underlying = os1.selectbox("Underlying (from Markets)", [k for k in asset_show if k in mkts_rs.columns], index=0)
days_to_expiry = os2.slider("Days to expiry", 7, 365, 90, step=7)
strike_pct     = os3.slider("Strike as % of spot", 50, 150, 100, step=5)
rf_basis       = os4.selectbox("Rate basis", ["Policy Rate", "10Y Yield"], index=0)
k_sigma        = os5.slider("Stress→σ sensitivity κσ", 0.00, 1.00, 0.35, 0.05)

# Spot & hist-vol from the same market feed
hist_u = mkts_rs[[underlying]].dropna()
spot = float(hist_u.iloc[-1, 0]) if hist_u.shape[0] else np.nan
strike = spot * (strike_pct / 100.0)
rets_u = np.log(hist_u[underlying]).diff().dropna()
sigma_hist = float(rets_u.tail(252).std() * sqrt(252.0)) if rets_u.shape[0] else 0.20

# Rate from FRED
rate_src = "FEDFUNDS" if rf_basis == "Policy Rate" else "DGS10"
rf_df = fetch_fred([rate_src], start_date, end_date)
rf_series = resample(rf_df, freq_code, agg_choice).dropna()
r = float(rf_series.iloc[-1, 0] / 100.0) if rf_series.shape[0] else 0.01

# Macro-linked σ
z = float(st.session_state.get("risk_z", 0.0))
sigma_macro = float(sigma_hist * exp(k_sigma * z))

# Simulations
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

plot_idx = np.random.choice(paths.shape[1], size=min(plot_n, paths.shape[1]), replace=False)
df_sim = pd.DataFrame(paths[:, plot_idx], index=pd.date_range(pd.Timestamp.today(), periods=days_to_expiry, freq="B"))
fig_sim = px.line(df_sim)
fig_sim.update_layout(height=420, title=f"{underlying} — Macro-linked Monte Carlo Paths")
st.plotly_chart(fig_sim, use_container_width=True)

# 3D scenario surface (value vs strike% vs expiry) with the same macro-linked σ
st.markdown("#### 3D Scenario Surface")
k_grid = np.linspace(0.8, 1.2, 25)
t_grid = np.linspace(7, 365, 24) / 365.0
Zsurf = np.zeros((t_grid.size, k_grid.size))
for i, T_ in enumerate(t_grid):
    inner = gbm_paths(spot, r, sigma_macro, int(T_*365), 600, 123)
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


# CORRELATION & EXPORT
st.markdown("## Correlation & Data Export")

cm1, cm2 = st.columns([1,1])
corr_basis = cm1.selectbox("Correlation basis", ["Levels (indexed)", "1M changes (%)"], index=1)
show_preview = cm2.checkbox("Show data preview", value=True)

panel_all = pd.DataFrame(index=pd.to_datetime([]))
if mkts_rs.shape[1] > 0: panel_all = mkts_rs.add_prefix("M: ")
if fred_rs.shape[1] > 0: panel_all = panel_all.join(fred_rs.add_prefix("F: "), how="outer")

panel_all_m = resample(panel_all, "M", "Last").dropna(how="all")
X = idx100(panel_all_m.dropna()) if corr_basis == "Levels (indexed)" else (panel_all_m.pct_change() * 100.0)

C = X.corr().replace([np.inf, -np.inf], np.nan).dropna(how="all").dropna(axis=1, how="all")
fig_hm = go.Figure(data=go.Heatmap(z=C.values, x=C.columns, y=C.index, zmin=-1, zmax=1, colorbar=dict(title="ρ")))
fig_hm.update_layout(height=560)
st.plotly_chart(fig_hm, use_container_width=True)

if show_preview:
    st.markdown("#### Data (tail) & Export")
    st.dataframe(panel_all.tail(20))
    csv = panel_all.to_csv(index=True).encode("utf-8")
    st.download_button("⬇️ Download CSV", data=csv, file_name="macro_credit_panel.csv", mime="text/csv")
