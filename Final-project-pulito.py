import numpy as np
import streamlit as st
import pandas as pd
from datetime import date
import yfinance as yf
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import minimize

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG  — must be the very first Streamlit call
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="UPO Analytics",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
#  GLOBAL PALETTE  — one source of truth for all charts
# ══════════════════════════════════════════════════════════════════════════════

PAL = {
    "meu":      "#2166AC",
    "mvp":      "#1A9641",
    "sharpe":   "#D6604D",
    "bench":    "#7B3294",
    "frontier": "#4393C3",
    "gold":     "#C9A84C",
}

HIST_PAL = {
    "meu":    "#7EB8F7",
    "mvp":    "#85E89D",
    "sharpe": "#FFD580",
}

NORM_PAL = {
    "meu":    "#FF1744",
    "mvp":    "#00B0FF",
    "sharpe": "#AA00FF",
}

TRADING_DAYS = 252

# ══════════════════════════════════════════════════════════════════════════════
#  GLOBAL CSS
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
[data-baseweb="tag"] {
    background-color: #42A5F5 !important;
    color: #FFFFFF !important;
}
.upo-footer {
    margin-top: 3rem;
    border-top: 2px solid #1A1A1A;
    padding-top: 1rem;
    font-family: Georgia, "Times New Roman", serif;
    font-size: 0.78rem;
    color: #4A4A4A;
    line-height: 1.7;
}
.upo-footer strong { color: #1A1A1A; font-size: 0.82rem; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  HEADER BANNER  (SVG inline — no external image needed)
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<svg viewBox="0 0 900 160" xmlns="http://www.w3.org/2000/svg"
     width="100%" style="display:block;margin-bottom:0.5rem;">
  <defs>
    <linearGradient id="bgGrad" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%"   stop-color="#0D2137"/>
      <stop offset="100%" stop-color="#1565C0"/>
    </linearGradient>
    <linearGradient id="chartGrad" x1="0%" y1="100%" x2="0%" y2="0%">
      <stop offset="0%"   stop-color="#1976D2" stop-opacity="0.0"/>
      <stop offset="100%" stop-color="#42A5F5" stop-opacity="0.35"/>
    </linearGradient>
  </defs>
  <rect width="900" height="160" fill="url(#bgGrad)"/>
  <rect x="0" y="0" width="900" height="3" fill="#C9A84C"/>
  <polygon
    points="520,145 560,100 600,115 640,75 680,90 720,55 760,70 800,40 840,60 880,30 900,42 900,145"
    fill="url(#chartGrad)"/>
  <polyline
    points="520,145 560,100 600,115 640,75 680,90 720,55 760,70 800,40 840,60 880,30 900,42"
    fill="none" stroke="#42A5F5" stroke-width="2.5" stroke-linejoin="round"/>
  <circle cx="640" cy="75"  r="3.5" fill="#FFFFFF" opacity="0.7"/>
  <circle cx="720" cy="55"  r="3.5" fill="#FFFFFF" opacity="0.7"/>
  <circle cx="800" cy="40"  r="3.5" fill="#FFFFFF" opacity="0.7"/>
  <circle cx="880" cy="30"  r="5"   fill="#C9A84C"/>
  <text x="38" y="88" font-family="Georgia,'Times New Roman',serif"
        font-size="54" font-weight="700" fill="#FFFFFF" letter-spacing="1">
    UPO Analytics</text>
  <text x="40" y="118" font-family="Georgia,'Times New Roman',serif"
        font-size="14" fill="#A8C8F0" letter-spacing="3.5">
    PORTFOLIO &amp; MARKET INTELLIGENCE PLATFORM</text>
  <rect x="38" y="130" width="415" height="1.2" fill="#C9A84C" opacity="0.85"/>
</svg>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  LANDING DESCRIPTION
# ══════════════════════════════════════════════════════════════════════════════

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("""
**Stock Analysis**

Deep-dive into a single asset: historical prices, moving averages (MA50/MA200),
log-return or simple-return distributions, descriptive statistics,
Normal overlay and Beta vs a chosen benchmark.
""")
with c2:
    st.markdown("""
**Correlation Analysis**

Explore co-movement across your asset universe: covariance matrix, correlation
matrix, interactive heatmap and a normalised price chart (base 100).
""")
with c3:
    st.markdown("""
**Portfolio Analysis**

MEU, GMV and Max-Sharpe optimisation, efficient frontier, cumulative wealth
simulation, Sharpe, Beta, Max Drawdown, and both bootstrap and
parametric VaR/ES at multiple horizons.
""")

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def validate_ticker(ticker: str) -> bool:
    try:
        data = yf.download(ticker, period="5d", progress=False, auto_adjust=True)
        return not data.empty
    except Exception:
        return False


@st.cache_data(show_spinner=False)
def get_ticker_currency(ticker: str) -> str:
    try:
        return yf.Ticker(ticker).info.get("currency", "N/A")
    except Exception:
        return "N/A"


@st.cache_data(show_spinner=False)
def get_full_history(ticker: str):
    """Download the full available history for a ticker (cached)."""
    try:
        data = yf.download(ticker, period="max", progress=False, auto_adjust=True)
        return data if not data.empty else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def get_first_available_date(ticker: str):
    data = get_full_history(ticker)
    return data.index.min().date() if not data.empty else None


def fetch_risk_free_rate() -> float:
    try:
        rf = yf.download("^IRX", period="5d", progress=False, auto_adjust=True)
        return float(rf["Close"].iloc[-1].item()) / 100
    except Exception:
        return 0.05


# ── Portfolio optimisation ─────────────────────────────────────────────────────

def portfolio_stats(weights, mu, cov):
    w = np.array(weights)
    return float(w @ mu) * TRADING_DAYS, float(np.sqrt(w @ cov @ w)) * np.sqrt(TRADING_DAYS)


def _opt(fun, n, cov=None, extra_constraints=None):
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    if extra_constraints:
        constraints += extra_constraints
    res = minimize(fun, np.ones(n) / n, method="SLSQP",
                   bounds=[(0, 1)] * n, constraints=constraints,
                   options={"ftol": 1e-12, "maxiter": 1000})
    return res.x if res.success else np.ones(n) / n


def meu_portfolio(mu, cov, lam):
    return _opt(lambda w: -(w @ mu - lam * (w @ cov @ w)), len(mu))


def mvp_portfolio(cov):
    return _opt(lambda w: w @ cov @ w, cov.shape[0])


def max_sharpe_portfolio(mu, cov, rf_daily=0.0):
    def neg_sharpe(w):
        r = w @ mu * TRADING_DAYS
        v = np.sqrt(w @ cov @ w) * np.sqrt(TRADING_DAYS)
        return -(r - rf_daily * TRADING_DAYS) / (v + 1e-10)
    return _opt(neg_sharpe, len(mu))


def efficient_frontier(mu, cov, n_points=80):
    n = len(mu)
    w_gmv   = mvp_portfolio(cov)
    gmv_ret = float(w_gmv @ mu) * TRADING_DAYS
    front_vols, front_rets = [], []
    for tr in np.linspace(gmv_ret, float(mu.max()) * TRADING_DAYS, n_points):
        res = minimize(lambda w: w @ cov @ w, np.ones(n) / n, method="SLSQP",
                       bounds=[(0, 1)] * n,
                       constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1},
                                    {"type": "eq", "fun": lambda w, tr=tr: w @ mu * TRADING_DAYS - tr}],
                       options={"ftol": 1e-12, "maxiter": 1000})
        if res.success:
            front_vols.append(np.sqrt(res.fun) * np.sqrt(TRADING_DAYS))
            front_rets.append(tr)
    return np.array(front_vols), np.array(front_rets)


# ── Risk metrics ───────────────────────────────────────────────────────────────

def var_es_bootstrap(port_returns: np.ndarray, horizon_days: int,
                     confidence: float, n_sims: int = 1000) -> tuple:
    """
    Bootstrap VaR and ES for a given horizon.

    For each simulation, `horizon_days` daily returns are drawn with replacement
    from the empirical distribution and summed to produce a multi-period return.
    VaR and ES are read off the resulting distribution — no normality assumed.

    Parameters
    ----------
    port_returns : daily log-returns of the portfolio
    horizon_days : number of trading days (1, 21, 63, 126, 252)
    confidence   : e.g. 0.95 → 5th-percentile loss
    n_sims       : number of bootstrap paths (default 1 000)

    Returns
    -------
    (var, es) both as log-return fractions (negative = loss)
    """
    port_returns = np.asarray(port_returns, dtype=float)
    alpha = 1.0 - confidence

    # Draw n_sims paths of horizon_days daily returns
    paths = np.random.choice(port_returns, size=(n_sims, horizon_days), replace=True)
    cum_returns = paths.sum(axis=1)          # sum of log-returns ≡ log of cumulative return

    var = float(np.percentile(cum_returns, alpha * 100))
    tail = cum_returns[cum_returns <= var]
    es   = float(tail.mean()) if len(tail) > 0 else var

    return var, es


def var_es_parametric(port_returns: np.ndarray, confidence: float = 0.95) -> tuple:
    """Daily parametric (Gaussian) VaR and ES."""
    mu_p  = float(np.mean(port_returns))
    sig_p = float(np.std(port_returns, ddof=1))
    alpha = 1.0 - confidence
    z     = stats.norm.ppf(alpha)
    var   = mu_p + z * sig_p
    es    = mu_p - sig_p * stats.norm.pdf(z) / alpha
    return var, es


def data_quality_banner(n_obs: int) -> tuple:
    """
    Returns (streamlit_method_name, message) based on the number of observations.
    Drives the coloured banner shown above the VaR table.
    """
    if n_obs < 60:
        return ("error",
                "**Insufficient data** (< 60 observations). "
                "All VaR/ES estimates are statistically unreliable — "
                "interpret results with extreme caution.")
    elif n_obs < 252:
        return ("warning",
                f"**Limited data** ({n_obs} obs, < 1 year). "
                "Only the **daily** VaR is reasonably reliable. "
                "Longer horizons are simulated by chaining short bootstrap paths — "
                "treat them as rough approximations.")
    elif n_obs < 756:
        return ("warning",
                f"**Moderate data** ({n_obs} obs, 1–3 years). "
                "Daily and monthly VaR are reliable. "
                "3 M / 6 M / 1 Y horizons carry higher estimation uncertainty.")
    elif n_obs < 2520:
        return ("info",
                f"**Good data** ({n_obs} obs, 3–10 years). "
                "All horizons computed via bootstrap path simulation. "
                "The 1-year tail estimate still has material sampling error.")
    else:
        return ("success",
                f"**Excellent data** ({n_obs} obs, > 10 years). "
                "All horizons fully reliable via bootstrap.")


def max_drawdown(port_returns: np.ndarray) -> float:
    cum         = np.exp(np.cumsum(port_returns))
    running_max = np.maximum.accumulate(cum)
    return float((cum / running_max - 1).min())


def sharpe_ratio(ret_ann, vol_ann, rf_ann):
    return (ret_ann - rf_ann) / (vol_ann + 1e-10)


def sortino_ratio(ret_ann, port_daily_returns, rf_ann):
    downside = port_daily_returns[port_daily_returns < 0]
    if len(downside) == 0:
        return np.nan
    downside_vol = np.std(downside, ddof=1) * np.sqrt(TRADING_DAYS)
    return (ret_ann - rf_ann) / (downside_vol + 1e-10)


def compute_beta(port_daily_returns, bench_ret_series, port_dates):
    port_s  = pd.Series(port_daily_returns, index=port_dates)
    aligned = pd.concat([port_s, bench_ret_series], axis=1).dropna()
    if len(aligned) < 10:
        return np.nan
    cov_m = np.cov(aligned.values.T)
    return float(cov_m[0, 1] / (cov_m[1, 1] + 1e-12))


# ── Chart helpers ──────────────────────────────────────────────────────────────

def weights_bar(weights_dict, tickers):
    fig = go.Figure()
    for key, (label, w) in zip(["meu", "mvp", "sharpe"], weights_dict.items()):
        fig.add_trace(go.Bar(name=label, x=tickers,
                             y=np.round(w * 100, 2),
                             marker_color=PAL[key]))
    fig.update_layout(barmode="group", yaxis_title="Weight (%)",
                      xaxis_title="Ticker", legend_title="Portfolio", height=350,
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════

if "valid_tickers" not in st.session_state:
    st.session_state.valid_tickers = []
if "rf_rate" not in st.session_state:
    with st.spinner("Fetching risk-free rate…"):
        st.session_state.rf_rate = fetch_risk_free_rate()

# ══════════════════════════════════════════════════════════════════════════════
#  TICKER INPUT + CURRENCY VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

st.subheader("Enter your tickers")

raw_input = st.text_input(
    "Type one or more tickers separated by commas or spaces (e.g. AAPL, MSFT, BTC-USD)",
    placeholder="AAPL, MSFT, GOOGL, BTC-USD",
)

if st.button("Done — validate tickers"):
    if raw_input.strip():
        candidates = [t.strip().upper()
                      for t in raw_input.replace(",", " ").split() if t.strip()]
        valid, invalid = [], []
        with st.spinner("Validating tickers via Yahoo Finance…"):
            for t in candidates:
                (valid if validate_ticker(t) else invalid).append(t)

        if valid:
            with st.spinner("Checking currencies…"):
                currencies = {t: get_ticker_currency(t) for t in valid}

            unique_curr = set(c for c in currencies.values() if c != "N/A")

            if len(unique_curr) > 1:
                detail = "  \n".join(f"- **{t}** → {c}"
                                     for t, c in currencies.items())
                st.error(
                    "**Currency mismatch detected.** All tickers must be quoted "
                    "in the same currency for the analysis to be meaningful.\n\n"
                    + detail
                    + "\n\nPlease remove the conflicting tickers and try again."
                )
                st.session_state.valid_tickers = []
            else:
                st.session_state.valid_tickers = valid
                curr_label = list(unique_curr)[0] if unique_curr else "N/A"
                st.success(f"Valid tickers: **{', '.join(valid)}** — "
                           f"all quoted in **{curr_label}**.")

        if invalid:
            st.error(f"Could not find data for: **{', '.join(invalid)}** — skipped.")
    else:
        st.warning("Please type at least one ticker before clicking Done.")

if st.session_state.valid_tickers:
    st.info(f"**Active tickers:** {', '.join(st.session_state.valid_tickers)}")
else:
    st.warning("No valid tickers loaded yet. Enter tickers above and click **Done**.")

# ── Sidebar navigation ─────────────────────────────────────────────────────────

option = st.sidebar.selectbox(
    "What do you want to do?",
    ["Stock Analysis", "Correlation Analysis", "Portfolio Analysis"],
)

if not st.session_state.valid_tickers:
    st.stop()

valid_tickers = st.session_state.valid_tickers


# ══════════════════════════════════════════════════════════════════════════════
#  STOCK ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

if option == "Stock Analysis":

    symbol     = st.sidebar.selectbox("Ticker", valid_tickers)
    first_date = get_first_available_date(symbol)

    default_start_stock = max(
        first_date,
        (date.today() - pd.DateOffset(years=10)).date()
    ) if first_date else (date.today() - pd.DateOffset(years=10)).date()

    start = st.sidebar.date_input("Start Date",
                                   value=default_start_stock,
                                   min_value=first_date,
                                   max_value=date.today())
    end   = st.sidebar.date_input("End Date",
                                   value=date.today(),
                                   min_value=start,
                                   max_value=date.today())

    ret_type  = st.sidebar.radio("Return type", ["Log Returns", "Simple Returns"])
    show_ma   = st.sidebar.checkbox("Show MA50 / MA200", value=True)
    bench_sym = st.sidebar.text_input(
        "Benchmark for Beta (optional)",
        value="^GSPC",
        help=(
            "Beta measures how much this asset moves relative to a market index. "
            "Common benchmark tickers:\n"
            "• ^GSPC — S&P 500 (USA, broad market)\n"
            "• ^STOXX50E — Euro Stoxx 50 (Eurozone, large caps)\n"
            "• ^FTSE — FTSE 100 (UK)\n"
            "• ^GDAXI — DAX 40 (Germany)\n"
            "• ^FCHI — CAC 40 (France)\n"
            "• FTSEMIB.MI — FTSE MIB (Italy)\n"
            "Leave blank to skip Beta calculation."
        )
    )

    stock = yf.download(symbol, start=start, end=end, progress=False)
    if stock.empty:
        st.error("No data returned for the selected ticker and date range.")
        st.stop()

    close_series = stock["Close"].squeeze()

    # ── KPI row ───────────────────────────────────────────────────────────────
    st.write("# Resume")
    n_trading_stock = int(stock["Close"].squeeze().dropna().shape[0])
    st.markdown(
        f"<small>Period: {stock.index[0].date()} → {stock.index[-1].date()} "
        f"&nbsp;|&nbsp; Trading days: {n_trading_stock} "
        f"&nbsp;|&nbsp; Ticker: {symbol}</small>",
        unsafe_allow_html=True,
    )
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Last Close",  f"{float(close_series.iloc[-1]):.2f}")
    k2.metric("Last Open",   f"{float(stock['Open'].squeeze().iloc[-1]):.2f}")
    k3.metric("Volume",      f"{float(stock['Volume'].squeeze().iloc[-1]):,.0f}")

    if bench_sym.strip():
        try:
            bench_raw = yf.download(bench_sym.strip(), start=start, end=end,
                                    progress=False)
            if not bench_raw.empty:
                b_ret = np.log(bench_raw["Close"].squeeze()).diff().dropna()
                s_ret = np.log(close_series).diff().dropna()
                aln   = pd.concat([s_ret, b_ret], axis=1).dropna()
                if len(aln) >= 10:
                    cm = np.cov(aln.values.T)
                    k4.metric(f"Beta vs {bench_sym.strip()}", f"{cm[0,1]/cm[1,1]:.3f}")
        except Exception:
            pass

    stock_swap = stock.swaplevel(axis=1)[symbol]

    # ── Historical data ───────────────────────────────────────────────────────
    st.write("# Historical Data")
    st.dataframe(stock_swap)
    st.download_button("Download CSV",
                       stock_swap.to_csv().encode("utf-8"),
                       file_name=f"{symbol}_{start:%Y%m%d}_{end:%Y%m%d}.csv",
                       mime="text/csv")

    # ── Price chart + MA ──────────────────────────────────────────────────────
    st.write("# Close Prices")
    st.caption(
        "Daily closing price of the selected asset over the chosen period. "
        "**MA50** (dashed) = 50-day simple moving average, a short-term trend indicator. "
        "**MA200** (dotted) = 200-day simple moving average, widely used to identify long-term trend direction. "
        "A price crossing above MA200 is often interpreted as a bullish signal, and vice versa."
    )
    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(
        x=stock_swap.index, y=stock_swap["Close"],
        mode="lines", name="Close", line=dict(color=PAL["meu"], width=1.8)
    ))
    if show_ma:
        full_hist  = get_full_history(symbol)
        close_full = full_hist.swaplevel(axis=1)[symbol]["Close"]
        for window, color, dash in [(50, PAL["sharpe"], "dash"),
                                     (200, PAL["mvp"],   "dot")]:
            ma_full     = close_full.rolling(window).mean()
            ma_filtered = ma_full[ma_full.index >= pd.Timestamp(start)]
            if ma_filtered.notna().sum() > 0:
                fig_price.add_trace(go.Scatter(
                    x=ma_filtered.index, y=ma_filtered, mode="lines",
                    name=f"MA{window}",
                    line=dict(color=color, width=1.5, dash=dash)
                ))
    fig_price.update_layout(xaxis_title="Date", yaxis_title="Price",
                             height=420, legend=dict(orientation="h"),
                             paper_bgcolor="rgba(0,0,0,0)",
                             plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_price, use_container_width=True)

    # ── Return distribution ───────────────────────────────────────────────────
    st.write(f"# {ret_type}")
    st.caption(
        "**Log Returns** = ln(Pₜ / Pₜ₋₁): continuously compounded daily returns, "
        "preferred for statistical modelling as they are time-additive. "
        "**Simple Returns** = (Pₜ – Pₜ₋₁) / Pₜ₋₁: intuitive percentage change, "
        "more suitable for portfolio arithmetic. "
        "The histogram shows the empirical distribution of daily returns. "
        "The solid line is the fitted Normal distribution with the same mean and standard deviation — "
        "deviations from it (asymmetry, fat tails) are visible at a glance."
    )
    logret = (np.log(stock_swap["Close"]).diff().dropna()
              if ret_type == "Log Returns"
              else stock_swap["Close"].pct_change().dropna())

    st.write("**Descriptive statistics**")
    st.dataframe(logret.describe().to_frame().T.style.format("{:.6f}"))

    x_r = np.linspace(float(logret.min()), float(logret.max()), 300)

    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=logret.values,
        name="Empirical",
        opacity=0.50,
        nbinsx=60,
        marker_color=PAL["meu"],
        histnorm="probability density",
        showlegend=True,
    ))
    fig_hist.add_trace(go.Scatter(
        x=x_r,
        y=stats.norm.pdf(x_r, logret.mean(), logret.std()),
        mode="lines",
        name="Normal fit",
        line=dict(color=NORM_PAL["meu"], width=2.5),
    ))
    fig_hist.update_layout(
        barmode="overlay",
        xaxis_title=ret_type,
        yaxis_title="Density",
        height=480,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_hist, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  CORRELATION ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

elif option == "Correlation Analysis":

    portfolio   = st.sidebar.multiselect("Tickers", valid_tickers, default=valid_tickers)

    if len(portfolio) >= 1:
        fd = [d for d in (get_first_available_date(t) for t in portfolio) if d]
        common_start = max(fd) if fd else date.today()
    else:
        common_start = date.today()

    ten_years_ago = (date.today() - pd.DateOffset(years=10)).date()
    default_start_corr = max(common_start, ten_years_ago)

    start = st.sidebar.date_input("Start Date", value=default_start_corr,
                                   min_value=common_start, max_value=date.today())
    end   = st.sidebar.date_input("End Date", value=date.today(),
                                   min_value=start, max_value=date.today())
    st.markdown(", ".join(portfolio) if portfolio else "*No tickers selected.*")

    if len(portfolio) < 2:
        st.warning("Select at least 2 tickers for correlation analysis.")
        st.stop()

    df = yf.download(portfolio, start=start, end=end, progress=False)["Close"]
    if df.empty:
        st.error("No data returned.")
        st.stop()

    r = np.log(df).diff().dropna()
    st.markdown(
        f"<small>Period: {r.index[0].date()} → {r.index[-1].date()} "
        f"&nbsp;|&nbsp; Trading days: {len(r)} "
        f"&nbsp;|&nbsp; Assets: {len(portfolio)} "
        f"&nbsp;|&nbsp; NaN rows dropped: {max(len(df) - len(r) - 1, 0)}</small>",
        unsafe_allow_html=True,
    )
    st.caption(
        "Measures the joint variability of each pair of assets. "
        "Diagonal values = variance of each asset. "
        "Off-diagonal values = covariance between pairs: "
        "positive → tend to move together; negative → tend to move in opposite directions. "
        "Values are in units of (daily log-return)²."
    )
    st.dataframe(r.cov().style.format("{:.8f}"))

    st.write("### Correlation Matrix")
    st.caption(
        "Standardised version of the covariance matrix: values always between –1 and +1. "
        "+1 = perfect positive co-movement · 0 = no linear relationship · –1 = perfect inverse movement. "
        "Easier to compare across different assets than raw covariance. "
        "Low or negative correlations between assets are the basis of diversification."
    )
    st.dataframe(r.corr().style.format("{:.4f}"))

    st.write("### Heatmap")
    st.caption(
        "Visual representation of the correlation matrix. "
        "Green = positive correlation (assets tend to move together); "
        "Red = negative correlation (assets tend to move in opposite directions). "
        "Numbers inside each cell show the exact correlation coefficient. "
        "A well-diversified portfolio benefits from low or negative off-diagonal values."
    )
    fig_heat, ax = plt.subplots(
        figsize=(max(6, len(portfolio)), max(4, len(portfolio) - 1))
    )
    fig_heat.patch.set_facecolor("none")
    ax.set_facecolor("#FDF6E3")
    sns.heatmap(r.corr(), annot=True, fmt=".3f", cmap="RdYlGn",
                center=0, linewidths=0.5, ax=ax, annot_kws={"size": 10})
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    st.pyplot(fig_heat)

    st.write("### Normalised Price Chart (base 100)")
    st.caption(
        "All asset prices are rescaled so that each starts at 100 on the first date of the period. "
        "This allows direct visual comparison of relative performance across assets "
        "regardless of their absolute price levels or currencies. "
        "A value of 150 means the asset has gained 50% since the start; 80 means a 20% loss."
    )
    norm = df / df.iloc[0] * 100
    colors_cycle = [PAL["meu"], PAL["mvp"], PAL["sharpe"],
                    PAL["bench"], PAL["gold"], "#555555"]
    fig_norm = go.Figure()
    for i, col in enumerate(norm.columns):
        fig_norm.add_trace(go.Scatter(
            x=norm.index, y=norm[col], mode="lines", name=col,
            line=dict(color=colors_cycle[i % len(colors_cycle)], width=1.8)
        ))
    fig_norm.update_layout(xaxis_title="Date", yaxis_title="Value (base 100)",
                            height=420,
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_norm, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PORTFOLIO ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

elif option == "Portfolio Analysis":

    st.write("# Portfolio Analysis")

    # ── Sidebar ───────────────────────────────────────────────────────────────

    portfolio = st.sidebar.multiselect("Tickers", valid_tickers, default=valid_tickers)

    if len(portfolio) >= 1:
        fd = [d for d in (get_first_available_date(t) for t in portfolio) if d]
        common_start = max(fd) if fd else date.today()
    else:
        common_start = date.today()

    ten_years_ago = (date.today() - pd.DateOffset(years=10)).date()
    default_start_port = max(common_start, ten_years_ago)

    start = st.sidebar.date_input("Start Date", value=default_start_port,
                                   min_value=common_start, max_value=date.today())
    end   = st.sidebar.date_input("End Date", value=date.today(),
                                   min_value=start, max_value=date.today())

    lam = st.sidebar.slider("Risk Aversion λ", 0.5, 50.0, 5.0, 0.5,
                             help="Higher λ = more risk-averse  (MEU: max μ – λσ²)")

    auto_rf_pct = round(st.session_state.rf_rate * 100, 3)
    st.sidebar.markdown(f"**Risk-Free Rate (auto):** {auto_rf_pct:.3f}% *(US 3M T-Bill)*")
    rf_pct = st.sidebar.number_input("Override Risk-Free Rate (% annual)",
                                      0.0, 20.0, auto_rf_pct, 0.1)
    rf = rf_pct / 100

    bench_sym = st.sidebar.text_input("Benchmark for Beta (optional)", value="^GSPC")

    if len(portfolio) < 2:
        st.warning("Select at least 2 tickers to run portfolio analysis.")
        st.stop()

    # ── Data download ─────────────────────────────────────────────────────────

    with st.spinner("Downloading price data…"):
        prices = yf.download(portfolio, start=start, end=end, progress=False)["Close"]

    if prices.empty:
        st.error("No data returned.")
        st.stop()

    prices_raw   = prices.copy()
    prices       = prices.dropna(axis=1, how="all")
    tickers_used = list(prices.columns)

    r   = np.log(prices).diff().dropna()
    mu  = r.mean().values
    cov = r.cov().values

    n_obs = len(r)   # number of trading-day observations — drives quality banner

    st.markdown(
        f"<small>Period: {r.index[0].date()} → {r.index[-1].date()} "
        f"&nbsp;|&nbsp; Trading days: {n_obs} "
        f"&nbsp;|&nbsp; Assets: {len(tickers_used)} "
        f"&nbsp;|&nbsp; NaN rows dropped: {max(len(prices_raw) - n_obs - 1, 0)}</small>",
        unsafe_allow_html=True,
    )

    # ── Benchmark ─────────────────────────────────────────────────────────────

    bench_ret_series = None
    if bench_sym.strip():
        try:
            b_raw = yf.download(bench_sym.strip(), start=start, end=end,
                                progress=False)
            if not b_raw.empty:
                bench_ret_series = np.log(b_raw["Close"].squeeze()).diff().dropna()
        except Exception:
            pass

    # ── Optimisations ─────────────────────────────────────────────────────────

    with st.spinner("Optimising portfolios…"):
        w_meu    = meu_portfolio(mu, cov, lam)
        w_mvp    = mvp_portfolio(cov)
        w_sharpe = max_sharpe_portfolio(mu, cov, rf_daily=rf / TRADING_DAYS)

    ret_meu,  vol_meu  = portfolio_stats(w_meu,    mu, cov)
    ret_mvp,  vol_mvp  = portfolio_stats(w_mvp,    mu, cov)
    ret_shar, vol_shar = portfolio_stats(w_sharpe, mu, cov)

    meu_label = f"MEU (λ={lam})"

    port_ret_meu  = r.values @ w_meu
    port_ret_mvp  = r.values @ w_mvp
    port_ret_shar = r.values @ w_sharpe

    port_series = [
        (meu_label,      port_ret_meu,  ret_meu,  vol_meu,
         PAL["meu"],    HIST_PAL["meu"],    NORM_PAL["meu"]),
        ("Min Variance", port_ret_mvp,  ret_mvp,  vol_mvp,
         PAL["mvp"],    HIST_PAL["mvp"],    NORM_PAL["mvp"]),
        ("Max Sharpe",   port_ret_shar, ret_shar, vol_shar,
         PAL["sharpe"], HIST_PAL["sharpe"], NORM_PAL["sharpe"]),
    ]

    # ── KPI metrics (Max Sharpe) ───────────────────────────────────────────────

    st.write("## Key Performance Indicators — Max Sharpe Portfolio")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Ann. Return",     f"{ret_shar * 100:.2f}%")
    k2.metric("Ann. Volatility", f"{vol_shar * 100:.2f}%")
    k3.metric("Sharpe Ratio",    f"{sharpe_ratio(ret_shar, vol_shar, rf):.3f}")
    k4.metric("Max Drawdown",    f"{max_drawdown(port_ret_shar) * 100:.2f}%")

    # ── Efficient frontier ────────────────────────────────────────────────────

    st.write("## Efficient Frontier")
    st.caption("Drawn from the Global Minimum Variance portfolio upward — "
               "the inefficient lower branch is excluded.")

    with st.spinner("Computing efficient frontier…"):
        ef_vols, ef_rets = efficient_frontier(mu, cov, n_points=80)

    fig_ef = go.Figure()
    fig_ef.add_trace(go.Scatter(
        x=ef_vols * 100, y=ef_rets * 100, mode="lines", name="Efficient Frontier",
        line=dict(color=PAL["frontier"], width=2)
    ))
    for vx, ry, name, color, sym in [
        (vol_meu  * 100, ret_meu  * 100, meu_label,   PAL["meu"],    "star"),
        (vol_mvp  * 100, ret_mvp  * 100, "GMV",        PAL["mvp"],    "diamond"),
        (vol_shar * 100, ret_shar * 100, "Max Sharpe", PAL["sharpe"], "circle"),
    ]:
        fig_ef.add_trace(go.Scatter(
            x=[vx], y=[ry], mode="markers+text", name=name,
            text=[name], textposition="top center",
            marker=dict(size=14, color=color, symbol=sym)
        ))
    fig_ef.update_layout(
        xaxis_title="Annualised Volatility (%)",
        yaxis_title="Annualised Log-Return (%)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=520,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_ef, use_container_width=True)

    # ── Weights ───────────────────────────────────────────────────────────────

    st.write("## Portfolio Weights")
    fig_w = weights_bar(
        {meu_label: w_meu, "Min Variance": w_mvp, "Max Sharpe": w_sharpe},
        tickers_used
    )
    st.plotly_chart(fig_w, use_container_width=True)

    weights_df = pd.DataFrame(
        {meu_label:      np.round(w_meu    * 100, 2),
         "Min Variance": np.round(w_mvp    * 100, 2),
         "Max Sharpe":   np.round(w_sharpe * 100, 2)},
        index=tickers_used
    )
    weights_df.index.name = "Ticker"
    st.dataframe(weights_df.style.format("{:.2f}%"))

    # ── Cumulative wealth (base 100) ──────────────────────────────────────────

    st.write("## Cumulative Wealth — Base 100")
    st.caption("Growth of 100 units invested at the start of the period "
               "in each optimised portfolio.")

    fig_cum = go.Figure()
    for label, pret, _, _, pal_col, _, _ in port_series:
        cum_vals = np.concatenate([[100], 100 * np.exp(np.cumsum(pret))])
        fig_cum.add_trace(go.Scatter(
            x=prices.index, y=np.round(cum_vals, 4),
            mode="lines", name=label,
            line=dict(color=pal_col, width=2)
        ))

    if bench_ret_series is not None:
        b_aln  = bench_ret_series.reindex(r.index).fillna(0)
        b_cum  = np.concatenate([[100], 100 * np.exp(np.cumsum(b_aln.values))])
        fig_cum.add_trace(go.Scatter(
            x=prices.index, y=np.round(b_cum, 4),
            mode="lines", name=f"Benchmark ({bench_sym.strip()})",
            line=dict(color=PAL["bench"], width=1.5, dash="dot")
        ))

    fig_cum.update_layout(
        xaxis_title="Date", yaxis_title="Portfolio Value (base 100)",
        height=450, legend=dict(orientation="h", yanchor="bottom", y=1.02),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_cum, use_container_width=True)

    # ── Performance summary ───────────────────────────────────────────────────

    st.write("## Performance Summary")

    perf_rows = []
    for label, pret, ret_a, vol_a, *_ in port_series:
        mdd      = max_drawdown(pret)
        skew     = float(stats.skew(pret))
        kurt     = float(stats.kurtosis(pret))
        sr       = sharpe_ratio(ret_a, vol_a, rf)
        beta_val = (compute_beta(pret, bench_ret_series, r.index)
                    if bench_ret_series is not None else np.nan)

        perf_rows.append({
            "Portfolio":           label,
            "Ann. Return (%)":     round(ret_a * 100, 2),
            "Ann. Volatility (%)": round(vol_a * 100, 2),
            "Sharpe Ratio":        round(sr, 3),
            "Beta":                round(beta_val, 3) if not np.isnan(beta_val) else "—",
            "Skewness":            round(skew, 4),
            "Excess Kurtosis":     round(kurt, 4),
            "Max Drawdown (%)":    round(mdd * 100, 2),
        })

    perf_df = pd.DataFrame(perf_rows).set_index("Portfolio")
    st.dataframe(perf_df)
    st.caption(
        "**Sharpe Ratio** = (annualised return – risk-free rate) / annualised volatility. "
        "Measures excess return per unit of total risk — higher is better. "
        "**Beta** = sensitivity of the portfolio relative to the chosen benchmark (default: S&P 500, ^GSPC). "
        "Beta = 1 → moves with the market; Beta > 1 → amplifies market moves; Beta < 1 → less volatile than the market. "
        "**Skewness** > 0 = right tail (extreme gains more likely); < 0 = left tail (extreme losses more likely). "
        "**Excess Kurtosis** > 0 = fat tails / leptokurtic (more extreme events than a Normal distribution). "
        "**Max Drawdown** = worst peak-to-trough loss over the entire period."
    )

    # ── VaR / ES — Bootstrap + Parametric ─────────────────────────────────────

    st.write("## Risk Measures — Bootstrap & Parametric")

    # ── Data quality banner ────────────────────────────────────────────────────
    dq_method, dq_msg = data_quality_banner(n_obs)
    getattr(st, dq_method)(dq_msg)

    st.caption(
        "**VaR (Value at Risk)** = maximum loss not exceeded at a given confidence level "
        "(e.g. VaR 95% → the loss will be smaller than this on 95% of trading days). "
        "**ES (Expected Shortfall / CVaR)** = average loss on the days that breach VaR: "
        "'how bad is it when things go wrong?'. "
        "**Bootstrap method** = for each horizon *T*, 1 000 paths of *T* daily returns are drawn "
        "with replacement from the empirical distribution and summed. "
        "VaR and ES are read off the resulting path-return distribution: "
        "no normality assumption, fat tails are preserved automatically. "
        "**Parametric method** = assumes Gaussian returns (μ, σ) scaled by √T. "
        "Compare the two to see how much the real tails deviate from the Normal. "
        "Horizons: 1 M ≈ 21 trading days · 3 M ≈ 63 · 6 M ≈ 126 · 1 Y ≈ 252."
    )

    CONFIDENCES = [0.90, 0.95, 0.99]
    HORIZONS    = {"Daily": 1, "1 Month": 21, "3 Months": 63,
                   "6 Months": 126, "Annual": TRADING_DAYS}

    # Pre-compute parametric daily VaR/ES (scaled below per horizon)
    cache_param = {}
    for label, pret, *_ in port_series:
        for conf in CONFIDENCES:
            cache_param[(label, conf)] = var_es_parametric(pret, conf)

    risk_rows = []
    for label, pret, *_ in port_series:          # ← pret properly unpacked here
        for h_label, h_days in HORIZONS.items():
            scale = np.sqrt(h_days)
            row   = {"Portfolio": label, "Horizon": h_label}
            for conf in CONFIDENCES:
                cs = f"{int(conf * 100)}%"

                # Bootstrap: simulate h_days-step paths, no scaling needed
                vh, eh = var_es_bootstrap(pret, horizon_days=h_days,
                                          confidence=conf, n_sims=1000)

                # Parametric: Gaussian daily VaR scaled by √T
                vp, ep = cache_param[(label, conf)]

                row[f"VaR Boot {cs} (%)"]  = round(vh * 100, 4)
                row[f"ES Boot  {cs} (%)"]  = round(eh * 100, 4)
                row[f"VaR Param {cs} (%)"] = round(vp * scale * 100, 4)
                row[f"ES Param  {cs} (%)"] = round(ep * scale * 100, 4)

            risk_rows.append(row)

    ordered_cols = []
    for conf in CONFIDENCES:
        cs = f"{int(conf * 100)}%"
        ordered_cols += [f"VaR Boot {cs} (%)", f"ES Boot  {cs} (%)",
                         f"VaR Param {cs} (%)", f"ES Param  {cs} (%)"]

    risk_df = (pd.DataFrame(risk_rows)
               .set_index(["Portfolio", "Horizon"])[ordered_cols])
    st.dataframe(risk_df)

    # ── Return distributions ───────────────────────────────────────────────────

    st.write("## Daily Log-Return Distributions")
    st.caption(
        "Muted bars = empirical histogram · Solid vivid lines = fitted Normal. "
        "Use the selector to show/hide individual series."
    )

    all_opts = []
    for label, *_ in port_series:
        all_opts += [f"{label} — Histogram", f"{label} — Normal fit"]

    selected = st.multiselect("Select series to display",
                               options=all_opts, default=all_opts)

    fig_dist = go.Figure()
    x_min = min(p.min() for _, p, *_ in port_series) * 100
    x_max = max(p.max() for _, p, *_ in port_series) * 100
    x_rng = np.linspace(x_min, x_max, 400)

    for label, pret, _, _, _, h_col, n_col in port_series:
        pct = pret * 100
        if f"{label} — Histogram" in selected:
            fig_dist.add_trace(go.Histogram(
                x=pct, name=f"{label} — Histogram",
                opacity=0.50, nbinsx=60, marker_color=h_col,
                histnorm="probability density", showlegend=True,
            ))
        if f"{label} — Normal fit" in selected:
            fig_dist.add_trace(go.Scatter(
                x=x_rng, y=stats.norm.pdf(x_rng, pct.mean(), pct.std()),
                mode="lines", name=f"{label} — Normal fit",
                line=dict(color=n_col, width=2.5),
            ))

    if not selected:
        st.info("Select at least one series from the menu above.")
    else:
        fig_dist.update_layout(
            barmode="overlay",
            xaxis_title="Daily Log-Return (%)", yaxis_title="Density",
            height=480,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_dist, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  FOOTER
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="upo-footer">
  <strong>UPO Analytics — Portfolio &amp; Market Intelligence Platform</strong><br>
  Developed for academic purposes at
  <em>Università del Piemonte Orientale (UPO)</em>
  — Department of Economics and Business.<br><br>

  <strong>Authors &amp; Student IDs</strong><br>
  Contini Giacomo &nbsp;·&nbsp; 20043816 &nbsp;&nbsp;|&nbsp;&nbsp;
  Paone Beatrice &nbsp;·&nbsp; 20038686 &nbsp;&nbsp;|&nbsp;&nbsp;
  Stojanova Sandra &nbsp;·&nbsp; 20044497 &nbsp;&nbsp;|&nbsp;&nbsp;
  Tierno Francesco &nbsp;·&nbsp; 20039651<br><br>

  &copy; 2026 UPO Analytics. All rights reserved. This application is intended
  solely for educational and research purposes. Data is sourced from publicly
  available financial databases (Yahoo Finance) and does not constitute financial
  advice, investment recommendations, or an offer to buy or sell any security.
  Past performance is not indicative of future results. The authors accept no
  liability for any decisions made on the basis of the information provided
  herein. All financial data is provided "as is" without warranty of any kind.
  Market data may be delayed. Users should consult a qualified financial adviser
  before making investment decisions.<br><br>

  <em>Data: Yahoo Finance &nbsp;·&nbsp;
  Computation: Python — NumPy, SciPy, yfinance &nbsp;·&nbsp;
  Visualisation: Plotly, Seaborn</em>
</div>
""", unsafe_allow_html=True)