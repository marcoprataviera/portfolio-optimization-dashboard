import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.dates as mdates
from contextlib import nullcontext

DEFAULT_STOCKS = ["AAPL", "MSFT", "NVDA", "JPM", "XOM"]
TICKER_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA",
    "JPM", "V", "MA", "UNH", "JNJ", "PG", "KO", "COST",
    "XOM", "CVX", "CAT", "NEE", "SPY",
]
MIN_SELECTED_TICKERS = 2
MAX_SELECTED_TICKERS = 20
RANDOM_SELECTION_COUNT = 5

TRADING_DAYS = 252
ROLLING_WINDOW = 63
MIN_WEIGHT_PCT = 0.0
MAX_WEIGHT_PCT = 40.0
DEFAULT_SINGLE_NAME_CAP_PCT = 40.0
DEFAULT_BENCHMARK = "SPY"
DEFAULT_REBALANCE = "Monthly"
VAR_CONFIDENCE = 0.95
RISK_FREE_RATE = 0.02

COLORS = {
    "portfolio": "#1f4e79",
    "benchmark": "#6aa84f",
    "drawdown": "#c44e52",
    "accent": "#c81d4f",
    "warning": "#f39c12",
    "success": "#2e7d32",
    "text": "#0f172a",
    "muted": "#64748b",
    "grid": "#dbe4ee",
    "border": "#e2e8f0",
    "panel": "#f8fafc",
    "panel_alt": "#ffffff",
}

CUSTOM_CSS = """
<style>
html, body, [class*="css"] {
    font-family: "Inter", "Segoe UI", "Helvetica Neue", sans-serif;
}

.block-container {
    max-width: 1320px;
    padding-top: 2.85rem;
    padding-bottom: 2.4rem;
    padding-left: 1.25rem;
    padding-right: 1.25rem;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #f8fafc 0%, #eef4fb 100%);
    border-right: 1px solid #e2e8f0;
}

[data-testid="stSidebar"] .stExpander {
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    background: rgba(255, 255, 255, 0.84);
    overflow: hidden;
}

[data-testid="stSidebar"] .stButton > button {
    width: 100%;
    border-radius: 12px;
    border: 1px solid #cbd5e1;
    font-weight: 700;
}

.hero-shell {
    padding: 0.25rem 0 0.8rem 0;
}

.hero-title {
    font-size: 2.22rem;
    font-weight: 850;
    line-height: 1.06;
    letter-spacing: -0.03em;
    color: #0f172a;
    margin: 0 0 0.35rem 0;
}

.hero-subtitle {
    font-size: 0.98rem;
    color: #475569;
    max-width: 920px;
    line-height: 1.55;
    margin-bottom: 0.9rem;
}

.section-shell {
    margin-top: 1.35rem;
    margin-bottom: 0.55rem;
    border-bottom: 1px solid #e2e8f0;
    padding-bottom: 0.62rem;
}

.section-shell h2 {
    margin: 0;
    font-size: 1.32rem;
    line-height: 1.15;
    font-weight: 800;
    color: #0f172a;
    letter-spacing: -0.02em;
}

.section-shell p {
    margin: 0.4rem 0 0 0;
    color: #64748b;
    font-size: 0.92rem;
    line-height: 1.5;
}

.summary-band {
    background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 0.9rem 1rem;
    color: #334155;
    margin: 0.35rem 0 0.95rem 0;
    box-shadow: 0 6px 20px rgba(15, 23, 42, 0.04);
    font-size: 0.94rem;
    line-height: 1.55;
}

.summary-band strong {
    color: #0f172a;
}

div[data-testid="stMetric"] {
    background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 0.82rem 0.95rem;
    box-shadow: 0 8px 24px rgba(15, 23, 42, 0.05);
}

div[data-testid="stMetricLabel"] {
    color: #64748b;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-size: 0.71rem;
}

div[data-testid="stMetricValue"] {
    color: #0f172a;
    font-weight: 850;
}

div[data-testid="stMetricDelta"] {
    font-size: 0.8rem;
}

div[data-testid="stDataFrame"] {
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    overflow: hidden;
    background: #ffffff;
    box-shadow: 0 6px 20px rgba(15, 23, 42, 0.04);
}

.table-caption {
    font-size: 0.88rem;
    color: #64748b;
    margin: 0.1rem 0 0.75rem 0;
    line-height: 1.5;
}

.small-note {
    font-size: 0.81rem;
    color: #64748b;
    margin-top: 0.45rem;
    line-height: 1.48;
}

.disclaimer-caption {
    margin: 0.12rem 0 0.45rem 0;
    padding: 0.68rem 0.82rem;
    background: #fff7ed;
    border: 1px solid #fed7aa;
    border-radius: 12px;
    color: #9a3412;
    font-size: 0.81rem;
    line-height: 1.42;
}

.source-caption {
    margin: 0.12rem 0 0.85rem 0;
    padding: 0.68rem 0.82rem;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    color: #475569;
    font-size: 0.81rem;
    line-height: 1.42;
}

.sidebar-header {
    font-size: 0.85rem;
    font-weight: 800;
    color: #0f172a;
    letter-spacing: 0.04em;
    margin-bottom: 0.45rem;
    text-transform: uppercase;
}

.sidebar-note {
    font-size: 0.81rem;
    color: #475569;
    line-height: 1.48;
}

.chart-spacer {
    margin-top: 0.15rem;
    margin-bottom: 0.3rem;
}
</style>
"""


def apply_custom_css():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def weights(stocks):
    equal_weight = 100 / len(stocks)
    return {ticker: equal_weight for ticker in stocks}


def _strip_tz(series):
    series = pd.to_datetime(series)
    return series.dt.tz_localize(None) if getattr(series.dt, "tz", None) is not None else series


def _safe_divide(numerator, denominator):
    if pd.isna(denominator) or np.isclose(denominator, 0):
        return np.nan
    return numerator / denominator


def _fmt_pct(value):
    return "N/A" if pd.isna(value) else f"{value:.2%}"


def _fmt_ratio(value):
    return "N/A" if pd.isna(value) else f"{value:.3f}"


def _fmt_currency(value):
    return "N/A" if pd.isna(value) else f"${value:,.2f}"


def _annualized_sharpe(return_series, rf_daily):
    vol = return_series.std()
    if pd.isna(vol) or np.isclose(vol, 0):
        return np.nan
    return ((return_series.mean() - rf_daily) / vol) * np.sqrt(TRADING_DAYS)


def _annualized_sortino(return_series, rf_daily):
    downside = return_series[return_series < rf_daily]
    downside_vol = downside.std()
    if len(downside) < 2 or pd.isna(downside_vol) or np.isclose(downside_vol, 0):
        return np.nan
    return ((return_series.mean() - rf_daily) / downside_vol) * np.sqrt(TRADING_DAYS)


def _drawdown_stats(series, dates):
    previous_peaks = series.cummax()
    drawdowns = (series - previous_peaks) / previous_peaks
    idx = drawdowns.idxmin()
    return drawdowns, drawdowns.min(), dates.loc[idx]


def _effective_max_weight_pct(single_stock_cap_pct=None):
    if single_stock_cap_pct is None:
        return MAX_WEIGHT_PCT
    return min(MAX_WEIGHT_PCT, float(single_stock_cap_pct))


def _validate_constraint_bounds(num_assets, min_weight, max_weight):
    if min_weight < 0 or max_weight > 1 or min_weight > max_weight:
        raise ValueError("Invalid portfolio constraints.")
    if min_weight * num_assets > 1 + 1e-12:
        raise ValueError("Minimum weight constraint is infeasible for the number of assets.")
    if max_weight * num_assets < 1 - 1e-12:
        raise ValueError("Maximum weight constraint is infeasible for the number of assets.")


def _generate_feasible_weight_vector(num_assets, min_weight, max_weight, rng):
    constrained_weights = np.full(num_assets, min_weight, dtype=float)
    remaining = 1.0 - constrained_weights.sum()
    if np.isclose(remaining, 0):
        return constrained_weights

    headroom = np.full(num_assets, max_weight - min_weight, dtype=float)
    draw_order = rng.permutation(num_assets)

    for position, idx in enumerate(draw_order[:-1]):
        future = draw_order[position + 1:]
        max_future_capacity = headroom[future].sum()
        lower = max(0.0, remaining - max_future_capacity)
        upper = min(headroom[idx], remaining)
        extra = rng.uniform(lower, upper) if upper > lower else lower
        constrained_weights[idx] += extra
        remaining -= extra

    constrained_weights[draw_order[-1]] += remaining
    return constrained_weights


def generate_constrained_weights(num_assets, min_weight=MIN_WEIGHT_PCT / 100, max_weight=MAX_WEIGHT_PCT / 100, rng=None):
    _validate_constraint_bounds(num_assets, min_weight, max_weight)
    rng = np.random.default_rng() if rng is None else rng

    if np.isclose(min_weight, 0):
        min_active_assets = int(np.ceil(1.0 / max_weight - 1e-12))
        min_active_assets = max(1, min_active_assets)
        active_assets = int(rng.integers(min_active_assets, num_assets + 1))
        active_idx = rng.choice(num_assets, size=active_assets, replace=False)
        active_weights = _generate_feasible_weight_vector(active_assets, 0.0, max_weight, rng)
        constrained_weights = np.zeros(num_assets, dtype=float)
        constrained_weights[active_idx] = active_weights
    else:
        constrained_weights = _generate_feasible_weight_vector(num_assets, min_weight, max_weight, rng)

    if (
        np.any(constrained_weights < min_weight - 1e-10)
        or np.any(constrained_weights > max_weight + 1e-10)
        or not np.isclose(constrained_weights.sum(), 1.0)
    ):
        raise ValueError("Unable to generate a feasible constrained portfolio.")

    return constrained_weights


def validate_weighting(weighting, min_weight_pct=MIN_WEIGHT_PCT, single_stock_cap_pct=DEFAULT_SINGLE_NAME_CAP_PCT):
    effective_max_pct = _effective_max_weight_pct(single_stock_cap_pct)
    weight_values = np.array(list(weighting.values()), dtype=float)

    if not np.isclose(weight_values.sum(), 100, atol=0.1):
        return False, f"Weights must sum to 100. Current sum: {weight_values.sum():.1f}%."

    below_min = [
        f"{ticker} {weight:.1f}%"
        for ticker, weight in weighting.items()
        if weight < min_weight_pct - 1e-9
    ]
    above_max = [
        f"{ticker} {weight:.1f}%"
        for ticker, weight in weighting.items()
        if weight > effective_max_pct + 1e-9
    ]

    violations = []
    if below_min:
        violations.append(f"Below minimum {min_weight_pct:.0f}%: " + ", ".join(below_min))
    if above_max:
        violations.append(f"Above maximum {effective_max_pct:.0f}%: " + ", ".join(above_max))

    return len(violations) == 0, " | ".join(violations)


def sanitize_tickers(ticker_values, min_count=MIN_SELECTED_TICKERS, max_count=MAX_SELECTED_TICKERS):
    cleaned = [ticker.strip().upper().replace(" ", "") for ticker in ticker_values]

    if any(not ticker for ticker in cleaned):
        raise ValueError("All selected ticker slots need a symbol.")

    duplicates = sorted({ticker for ticker in cleaned if cleaned.count(ticker) > 1})
    if duplicates:
        raise ValueError("Tickers must be unique. Duplicate(s): " + ", ".join(duplicates))

    if len(cleaned) < min_count:
        raise ValueError(f"Select at least {min_count} tickers.")
    if len(cleaned) > max_count:
        raise ValueError(f"Select no more than {max_count} tickers.")

    return cleaned


def normalize_weighting_dict(weighting):
    total = sum(float(value) for value in weighting.values())
    if np.isclose(total, 0):
        raise ValueError("Weights must sum to a positive value.")
    return {ticker: float(value) / total * 100 for ticker, value in weighting.items()}


@st.cache_data(show_spinner=False)
def load_price_history(stocks, period_len="max", price_col="Close"):
    stocks = list(stocks)
    frames = []

    for ticker in stocks:
        hist = yf.Ticker(ticker).history(period=period_len)
        if hist.empty:
            raise ValueError(f"No data returned for {ticker}.")

        df = hist[[price_col]].reset_index().rename(columns={price_col: ticker})
        df["Date"] = _strip_tz(df["Date"])
        frames.append(df[["Date", ticker]])

    if not frames:
        raise ValueError("No price history could be loaded for the selected tickers.")

    price_data = frames[0]
    for frame in frames[1:]:
        price_data = price_data.merge(frame, on="Date", how="inner")

    price_data = price_data.sort_values("Date").reset_index(drop=True)

    if price_data.empty:
        raise ValueError("The selected tickers do not share overlapping price history.")

    return price_data


def get_price_history_cached(stocks, period_len="max", price_col="Close"):
    return load_price_history(tuple(stocks), period_len=period_len, price_col=price_col).copy()


@st.cache_data(show_spinner=False)
def fetch_benchmark_history(benchmark_col, start_date, end_date):
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    benchmark_df = (
        yf.Ticker(benchmark_col)
        .history(start=start_date, end=end_date + pd.Timedelta(days=1))[["Close"]]
        .reset_index()
        .rename(columns={"Close": "Benchmark Price"})
    )

    if benchmark_df.empty:
        raise ValueError(f"No benchmark data returned for {benchmark_col}.")

    benchmark_df["Date"] = _strip_tz(benchmark_df["Date"])
    benchmark_df["Benchmark Value"] = benchmark_df["Benchmark Price"] / benchmark_df["Benchmark Price"].iloc[0]

    return benchmark_df[["Date", "Benchmark Value"]].copy()


def get_benchmark_history_cached(benchmark_col, start_date, end_date):
    return fetch_benchmark_history(
        benchmark_col,
        str(pd.to_datetime(start_date).date()),
        str(pd.to_datetime(end_date).date()),
    ).copy()


def build_rebalance_mask(dates, frequency):
    dates = pd.Series(pd.to_datetime(dates)).reset_index(drop=True)

    if frequency == "Daily":
        return pd.Series(True, index=dates.index)

    if frequency == "Buy & Hold":
        mask = pd.Series(False, index=dates.index)
        mask.iloc[0] = True
        return mask

    if frequency == "Monthly":
        periods = dates.dt.to_period("M")
    elif frequency == "Quarterly":
        periods = dates.dt.to_period("Q")
    elif frequency == "Annual":
        periods = dates.dt.to_period("Y")
    else:
        raise ValueError(f"Unsupported rebalance frequency: {frequency}")

    mask = periods.ne(periods.shift(-1))
    mask.iloc[0] = True
    return mask


def simulate_rebalanced_portfolio(price_df, weighting, rebalance_frequency=DEFAULT_REBALANCE, stocks=None):
    if stocks is None:
        stocks = list(weighting.keys())

    df = price_df.copy().sort_values("Date").reset_index(drop=True)
    df["Date"] = _strip_tz(df["Date"])

    if df.empty:
        raise ValueError("Price history is empty.")

    weight_vector = np.array([weighting[ticker] / 100 for ticker in stocks], dtype=float)
    if not np.isclose(weight_vector.sum(), 1.0, atol=1e-6):
        raise ValueError("Weights must sum to 100.")

    prices = df[stocks].astype(float)
    if (prices <= 0).any().any():
        raise ValueError("All prices must be positive for rebalancing simulation.")

    rebalance_mask = build_rebalance_mask(df["Date"], rebalance_frequency)

    portfolio_values = np.zeros(len(df), dtype=float)
    first_prices = prices.iloc[0].values
    current_value = 1.0
    shares = current_value * weight_vector / first_prices
    portfolio_values[0] = current_value

    for i in range(1, len(df)):
        current_prices = prices.iloc[i].values
        holding_values = shares * current_prices
        current_value = holding_values.sum()
        portfolio_values[i] = current_value

        if rebalance_mask.iloc[i]:
            shares = current_value * weight_vector / current_prices

    simulated = df.copy()
    simulated["Portfolio Returns"] = portfolio_values
    simulated["Portfolio Daily Return"] = simulated["Portfolio Returns"].pct_change()
    return simulated


def historical_var_cvar(return_series, confidence=VAR_CONFIDENCE):
    clean_returns = pd.Series(return_series).dropna()
    if clean_returns.empty:
        return np.nan, np.nan

    cutoff = np.quantile(clean_returns, 1 - confidence)
    tail = clean_returns[clean_returns <= cutoff]

    var_value = max(0.0, -cutoff)
    cvar_value = max(0.0, -tail.mean()) if not tail.empty else np.nan
    return var_value, cvar_value


def prepare_analysis_data(
    portfolio_df,
    start_date,
    end_date,
    stocks,
    portfolio_col="Portfolio Returns",
    benchmark_col=DEFAULT_BENCHMARK,
    benchmark_history=None,
):
    df = portfolio_df.copy()
    df["Date"] = _strip_tz(df["Date"])

    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    filtered = df[
        (df["Date"] >= start_date) &
        (df["Date"] <= end_date)
    ].sort_values("Date").copy()

    if filtered.empty:
        raise ValueError("No portfolio data is available for the selected date range.")

    filtered["Portfolio Value"] = filtered[portfolio_col] / filtered[portfolio_col].iloc[0]

    if benchmark_history is None:
        benchmark_history = fetch_benchmark_history(benchmark_col, start_date, end_date)

    analysis_df = (
        filtered[["Date", "Portfolio Value"] + stocks]
        .merge(benchmark_history, on="Date", how="inner")
        .sort_values("Date")
        .reset_index(drop=True)
    )

    if analysis_df.empty:
        raise ValueError("No overlapping dates found between the portfolio and benchmark.")

    analysis_df["Portfolio Value"] = analysis_df["Portfolio Value"] / analysis_df["Portfolio Value"].iloc[0]
    analysis_df["Benchmark Value"] = analysis_df["Benchmark Value"] / analysis_df["Benchmark Value"].iloc[0]

    return_frame = analysis_df.copy()
    return_columns = ["Portfolio Value", "Benchmark Value"] + stocks
    return_frame[return_columns] = return_frame[return_columns].pct_change()
    return_frame = return_frame.dropna().reset_index(drop=True).rename(
        columns={
            "Portfolio Value": "Portfolio Daily Return",
            "Benchmark Value": "Benchmark Daily Return",
        }
    )

    if return_frame.empty:
        raise ValueError("Not enough data to compute return statistics.")

    return analysis_df, return_frame


def summarize_series(
    analysis_df,
    return_frame,
    value_col,
    return_col,
    benchmark_return_col="Benchmark Daily Return",
    risk_free_rate=RISK_FREE_RATE,
    confidence=VAR_CONFIDENCE,
):
    start_date = pd.to_datetime(analysis_df["Date"].iloc[0])
    end_date = pd.to_datetime(analysis_df["Date"].iloc[-1])
    n_years = max((end_date - start_date).days / 365.25, 1 / TRADING_DAYS)

    total_return = analysis_df[value_col].iloc[-1] - 1
    cagr = analysis_df[value_col].iloc[-1] ** (1 / n_years) - 1

    _, max_drawdown, max_drawdown_date = _drawdown_stats(
        analysis_df[value_col],
        analysis_df["Date"],
    )

    rf_daily = risk_free_rate / TRADING_DAYS
    series_returns = return_frame[return_col]
    volatility = series_returns.std() * np.sqrt(TRADING_DAYS)
    sharpe = _annualized_sharpe(series_returns, rf_daily)
    sortino = _annualized_sortino(series_returns, rf_daily)
    var_95, cvar_95 = historical_var_cvar(series_returns, confidence=confidence)

    beta = np.nan
    alpha = np.nan

    if benchmark_return_col is not None and return_col != benchmark_return_col:
        benchmark_var = return_frame[benchmark_return_col].var()
        beta = _safe_divide(
            return_frame[return_col].cov(return_frame[benchmark_return_col]),
            benchmark_var,
        )

        if not pd.isna(beta):
            alpha = (
                (series_returns.mean() - rf_daily)
                - beta * (return_frame[benchmark_return_col].mean() - rf_daily)
            ) * TRADING_DAYS

    return {
        "total_return": total_return,
        "cagr": cagr,
        "max_drawdown": max_drawdown,
        "max_drawdown_date": max_drawdown_date,
        "volatility": volatility,
        "sharpe": sharpe,
        "sortino": sortino,
        "beta": beta,
        "alpha": alpha,
        "var_95": var_95,
        "cvar_95": cvar_95,
    }


def portfolio_performance_table(analysis_df, return_frame, stocks, weighting, benchmark_col=DEFAULT_BENCHMARK, risk_free_rate=RISK_FREE_RATE):
    portfolio_weights = np.array([weighting[ticker] / 100 for ticker in stocks], dtype=float)
    cov_matrix = return_frame[stocks].cov().values
    portfolio_var = float(portfolio_weights @ cov_matrix @ portfolio_weights)

    benchmark_var = return_frame["Benchmark Daily Return"].var()
    stock_betas = {
        ticker: _safe_divide(return_frame[ticker].cov(return_frame["Benchmark Daily Return"]), benchmark_var)
        for ticker in stocks
    }

    portfolio_summary = summarize_series(
        analysis_df,
        return_frame,
        value_col="Portfolio Value",
        return_col="Portfolio Daily Return",
        risk_free_rate=risk_free_rate,
    )

    benchmark_summary = summarize_series(
        analysis_df,
        return_frame,
        value_col="Benchmark Value",
        return_col="Benchmark Daily Return",
        benchmark_return_col=None,
        risk_free_rate=risk_free_rate,
    )

    metrics = [
        ("Portfolio CAGR", portfolio_summary["cagr"]),
        (f"{benchmark_col} CAGR", benchmark_summary["cagr"]),
        ("Active CAGR", portfolio_summary["cagr"] - benchmark_summary["cagr"]),
        ("Portfolio Total Return", portfolio_summary["total_return"]),
        (f"{benchmark_col} Total Return", benchmark_summary["total_return"]),
        ("Active Total Return", portfolio_summary["total_return"] - benchmark_summary["total_return"]),
        ("Portfolio Max Drawdown", portfolio_summary["max_drawdown"]),
        (f"{benchmark_col} Max Drawdown", benchmark_summary["max_drawdown"]),
        ("Portfolio Max Drawdown Date", portfolio_summary["max_drawdown_date"]),
        (f"{benchmark_col} Max Drawdown Date", benchmark_summary["max_drawdown_date"]),
        ("Portfolio Variance", portfolio_var),
    ]

    for ticker in stocks:
        metrics.append((f"Beta {ticker}", stock_betas[ticker]))

    metrics.extend([
        ("Portfolio Beta", portfolio_summary["beta"]),
        ("Portfolio Alpha", portfolio_summary["alpha"]),
        ("Portfolio Sharpe Ratio", portfolio_summary["sharpe"]),
        (f"{benchmark_col} Sharpe Ratio", benchmark_summary["sharpe"]),
        ("Portfolio Sortino Ratio", portfolio_summary["sortino"]),
        ("Portfolio Volatility", portfolio_summary["volatility"]),
        (f"{benchmark_col} Volatility", benchmark_summary["volatility"]),
        ("Portfolio 1-Day 95% VaR", portfolio_summary["var_95"]),
        ("Portfolio 1-Day 95% CVaR", portfolio_summary["cvar_95"]),
        (f"{benchmark_col} 1-Day 95% VaR", benchmark_summary["var_95"]),
        (f"{benchmark_col} 1-Day 95% CVaR", benchmark_summary["cvar_95"]),
    ])

    return pd.DataFrame(metrics, columns=["Metric", "Value"])


def format_perf_table(perf_table):
    perf_table = perf_table.copy()

    def _fmt(row):
        metric = row["Metric"]
        value = row["Value"]

        if pd.isna(value):
            return "N/A"
        if "Date" in metric:
            return pd.to_datetime(value).strftime("%Y-%m-%d")
        if metric == "Portfolio Variance":
            return f"{value:.6f}"
        if any(keyword in metric for keyword in ["Beta", "Sharpe", "Sortino"]):
            return f"{value:.3f}"
        if any(keyword in metric for keyword in ["CAGR", "Return", "Drawdown", "Alpha", "Volatility", "VaR", "CVaR"]):
            return f"{value:.2%}"
        return f"{value:.4f}"

    perf_table["Value"] = perf_table.apply(_fmt, axis=1)
    return perf_table


def rolling_metrics_table(return_frame, risk_free_rate=RISK_FREE_RATE, window=ROLLING_WINDOW):
    rf_daily = risk_free_rate / TRADING_DAYS
    rolling_portfolio_std = return_frame["Portfolio Daily Return"].rolling(window).std()
    rolling_benchmark_std = return_frame["Benchmark Daily Return"].rolling(window).std()
    rolling_covariance = return_frame["Portfolio Daily Return"].rolling(window).cov(return_frame["Benchmark Daily Return"])
    rolling_benchmark_var = return_frame["Benchmark Daily Return"].rolling(window).var()

    rolling_df = pd.DataFrame({
        "Date": return_frame["Date"],
        "Rolling Volatility": rolling_portfolio_std * np.sqrt(TRADING_DAYS),
        "Benchmark Rolling Volatility": rolling_benchmark_std * np.sqrt(TRADING_DAYS),
        "Rolling Sharpe": (
            (return_frame["Portfolio Daily Return"].rolling(window).mean() - rf_daily) / rolling_portfolio_std
        ) * np.sqrt(TRADING_DAYS),
        "Rolling Beta": rolling_covariance / rolling_benchmark_var,
    })

    return rolling_df.dropna().reset_index(drop=True)


def build_allocation_table(
    price_df,
    stocks,
    weighting,
    investment_amount,
    whole_shares_only=False,
    transaction_cost_bps=0.0,
):
    latest_row = price_df.sort_values("Date").iloc[-1]
    cost_rate = transaction_cost_bps / 10000

    rows = []
    for ticker in stocks:
        latest_price = float(latest_row[ticker])
        target_weight = float(weighting[ticker])
        target_budget = investment_amount * (target_weight / 100)

        if latest_price <= 0:
            raise ValueError(f"Latest price for {ticker} must be positive.")

        all_in_price = latest_price * (1 + cost_rate)

        if whole_shares_only:
            estimated_shares = np.floor(target_budget / all_in_price)
        else:
            estimated_shares = target_budget / all_in_price

        gross_invested = estimated_shares * latest_price
        transaction_cost = gross_invested * cost_rate
        total_cash_used = gross_invested + transaction_cost
        leftover_cash = max(target_budget - total_cash_used, 0.0)

        rows.append({
            "Ticker": ticker,
            "Latest Price": latest_price,
            "Target Weight": target_weight,
            "Realized Weight": np.nan,
            "Target Budget": target_budget,
            "Estimated Shares": estimated_shares,
            "Gross Invested": gross_invested,
            "Transaction Cost": transaction_cost,
            "Total Cash Used": total_cash_used,
            "Leftover Cash": leftover_cash,
        })

    allocation_df = pd.DataFrame(rows)
    gross_invested_total = allocation_df["Gross Invested"].sum()

    allocation_df["Realized Weight"] = np.where(
        gross_invested_total > 0,
        allocation_df["Gross Invested"] / gross_invested_total * 100,
        np.nan,
    )

    summary = {
        "target_budget": allocation_df["Target Budget"].sum(),
        "gross_invested": allocation_df["Gross Invested"].sum(),
        "transaction_cost": allocation_df["Transaction Cost"].sum(),
        "total_cash_used": allocation_df["Total Cash Used"].sum(),
        "leftover_cash": allocation_df["Leftover Cash"].sum(),
    }

    total_row = pd.DataFrame([{
        "Ticker": "Total",
        "Latest Price": np.nan,
        "Target Weight": allocation_df["Target Weight"].sum(),
        "Realized Weight": allocation_df["Realized Weight"].sum() if gross_invested_total > 0 else np.nan,
        "Target Budget": summary["target_budget"],
        "Estimated Shares": np.nan,
        "Gross Invested": summary["gross_invested"],
        "Transaction Cost": summary["transaction_cost"],
        "Total Cash Used": summary["total_cash_used"],
        "Leftover Cash": summary["leftover_cash"],
    }])

    allocation_df = pd.concat([allocation_df, total_row], ignore_index=True)
    return allocation_df, summary


def format_allocation_table(allocation_df):
    formatted = allocation_df.copy()

    currency_cols = [
        "Latest Price",
        "Target Budget",
        "Gross Invested",
        "Transaction Cost",
        "Total Cash Used",
        "Leftover Cash",
    ]
    for col in currency_cols:
        formatted[col] = formatted[col].apply(lambda x: "" if pd.isna(x) else f"${x:,.2f}")

    pct_cols = ["Target Weight", "Realized Weight"]
    for col in pct_cols:
        formatted[col] = formatted[col].apply(lambda x: "" if pd.isna(x) else f"{x:.2f}%")

    formatted["Estimated Shares"] = formatted["Estimated Shares"].apply(
        lambda x: "" if pd.isna(x) else f"{x:,.4f}"
    )

    return formatted


def portfolio_stats(weights_vec, mean_returns, cov_matrix, rf=RISK_FREE_RATE):
    annual_return = float(np.dot(weights_vec, mean_returns) * TRADING_DAYS)
    annual_vol = float(np.sqrt(weights_vec @ cov_matrix @ weights_vec) * np.sqrt(TRADING_DAYS))
    sharpe = np.nan if np.isclose(annual_vol, 0) else (annual_return - rf) / annual_vol
    return {
        "return": annual_return,
        "volatility": annual_vol,
        "sharpe": sharpe,
    }


def eff_frontier(
    stocks,
    returns,
    cov_matrix,
    num_ports=4000,
    rf=RISK_FREE_RATE,
    min_weight_pct=MIN_WEIGHT_PCT,
    single_stock_cap_pct=DEFAULT_SINGLE_NAME_CAP_PCT,
    random_seed=42,
):
    weights_list = []
    returns_list = []
    vol_list = []
    sharpe_list = []

    effective_max_pct = _effective_max_weight_pct(single_stock_cap_pct)
    min_weight = min_weight_pct / 100
    max_weight = effective_max_pct / 100
    _validate_constraint_bounds(len(stocks), min_weight, max_weight)

    rng = np.random.default_rng(random_seed)
    mean_returns = returns[stocks].mean().values

    for _ in range(num_ports):
        constrained_weights = generate_constrained_weights(
            num_assets=len(stocks),
            min_weight=min_weight,
            max_weight=max_weight,
            rng=rng,
        )

        stats = portfolio_stats(constrained_weights, mean_returns, cov_matrix, rf=rf)

        weights_list.append(constrained_weights)
        returns_list.append(stats["return"])
        vol_list.append(stats["volatility"])
        sharpe_list.append(stats["sharpe"])

    return weights_list, returns_list, vol_list, sharpe_list


def build_strategy_comparison_table(
    price_df,
    stocks,
    current_weighting,
    max_sharpe_weighting,
    min_vol_weighting,
    benchmark_col,
    rebalance_frequency,
    risk_free_rate=RISK_FREE_RATE,
    benchmark_history=None,
):
    strategies = [
        ("Current", current_weighting),
        ("Max Sharpe", max_sharpe_weighting),
        ("Min Vol", min_vol_weighting),
    ]

    rows = []
    start_date = pd.to_datetime(price_df["Date"].min())
    end_date = pd.to_datetime(price_df["Date"].max())

    for label, weighting in strategies:
        normalized_weighting = normalize_weighting_dict(weighting)

        simulated = simulate_rebalanced_portfolio(
            price_df=price_df,
            weighting=normalized_weighting,
            rebalance_frequency=rebalance_frequency,
            stocks=stocks,
        )

        analysis_df, return_frame = prepare_analysis_data(
            portfolio_df=simulated,
            start_date=start_date,
            end_date=end_date,
            stocks=stocks,
            benchmark_col=benchmark_col,
            benchmark_history=benchmark_history,
        )

        summary = summarize_series(
            analysis_df=analysis_df,
            return_frame=return_frame,
            value_col="Portfolio Value",
            return_col="Portfolio Daily Return",
            risk_free_rate=risk_free_rate,
        )

        row = {
            "Portfolio": label,
            "Rebalance": rebalance_frequency,
        }

        for ticker in stocks:
            row[ticker] = f"{normalized_weighting[ticker]:.1f}%"

        row.update({
            "Total Return": _fmt_pct(summary["total_return"]),
            "CAGR": _fmt_pct(summary["cagr"]),
            "Volatility": _fmt_pct(summary["volatility"]),
            "Sharpe": _fmt_ratio(summary["sharpe"]),
            "Sortino": _fmt_ratio(summary["sortino"]),
            "Beta": _fmt_ratio(summary["beta"]),
            "Alpha": _fmt_pct(summary["alpha"]),
            "Max Drawdown": _fmt_pct(summary["max_drawdown"]),
            "VaR 95%": _fmt_pct(summary["var_95"]),
            "CVaR 95%": _fmt_pct(summary["cvar_95"]),
        })

        rows.append(row)

    return pd.DataFrame(rows)


class DashboardHaltError(RuntimeError):
    """Raised when the dashboard should stop cleanly after showing an error."""


def _has_streamlit_context():
    try:
        from streamlit.runtime.scriptrunner_utils.script_run_context import get_script_run_ctx
    except Exception:
        try:
            from streamlit.runtime.scriptrunner.script_run_context import get_script_run_ctx
        except Exception:
            return False

    return get_script_run_ctx() is not None


def spinner_context(text):
    return st.spinner(text) if _has_streamlit_context() else nullcontext()


def stop_with_error(message):
    if _has_streamlit_context():
        st.error(message)
        st.stop()
    raise DashboardHaltError(message)


def initialize_session_state():
    for ticker in TICKER_UNIVERSE:
        default_weight = 100.0 / len(DEFAULT_STOCKS) if ticker in DEFAULT_STOCKS else 0.0
        st.session_state.setdefault(f"weight_{ticker}", default_weight)

    st.session_state.setdefault("selected_tickers", DEFAULT_STOCKS.copy())
    st.session_state.setdefault("benchmark_col", DEFAULT_BENCHMARK)
    st.session_state.setdefault("rebalance_frequency", DEFAULT_REBALANCE)
    st.session_state.setdefault("investment_amount", 100000.0)
    st.session_state.setdefault("whole_shares_only", True)
    st.session_state.setdefault("transaction_cost_bps", 10.0)
    st.session_state.setdefault("normalize_weights", True)
    st.session_state.setdefault("fill_drawdown", True)
    st.session_state.setdefault("single_stock_cap_pct", DEFAULT_SINGLE_NAME_CAP_PCT)
    st.session_state.setdefault("year_range", None)


def set_equal_weight_selection(selected_tickers):
    if not selected_tickers:
        return
    equal_weight = 100.0 / len(selected_tickers)
    for ticker in TICKER_UNIVERSE:
        st.session_state[f"weight_{ticker}"] = equal_weight if ticker in selected_tickers else 0.0


def add_ticker_to_selection():
    current = list(st.session_state.get("selected_tickers", DEFAULT_STOCKS.copy()))
    if len(current) >= MAX_SELECTED_TICKERS:
        return

    available = [ticker for ticker in TICKER_UNIVERSE if ticker not in current]
    if available:
        current.append(available[0])
        st.session_state["selected_tickers"] = current
        st.session_state.setdefault(f"weight_{available[0]}", 0.0)


def remove_ticker_from_selection():
    current = list(st.session_state.get("selected_tickers", DEFAULT_STOCKS.copy()))
    if len(current) > MIN_SELECTED_TICKERS:
        current.pop()
        st.session_state["selected_tickers"] = current


def randomize_five_tickers():
    rng = np.random.default_rng()
    selection = list(rng.choice(TICKER_UNIVERSE, size=RANDOM_SELECTION_COUNT, replace=False))
    st.session_state["selected_tickers"] = selection
    set_equal_weight_selection(selection)


def section_header(title, subtitle):
    st.markdown(
        f"""
        <div class="section-shell">
            <h2>{title}</h2>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def summary_band(text):
    st.markdown(f"<div class='summary-band'>{text}</div>", unsafe_allow_html=True)


def table_caption(text):
    st.markdown(f"<div class='table-caption'>{text}</div>", unsafe_allow_html=True)


def small_note(text):
    st.markdown(f"<div class='small-note'>{text}</div>", unsafe_allow_html=True)


def disclaimer_caption(text):
    st.markdown(
        f"<div class='disclaimer-caption'>{text}</div>",
        unsafe_allow_html=True,
    )


def source_caption(text):
    st.markdown(
        f"<div class='source-caption'>{text}</div>",
        unsafe_allow_html=True,
    )


def render_dataframe(df, height=None, max_rows_visible=None):
    if height is None:
        visible_rows = len(df) if max_rows_visible is None else min(len(df), max_rows_visible)
        header_px = 38
        row_px = 35
        padding_px = 6
        height = header_px + (visible_rows * row_px) + padding_px
        if len(df) <= 3:
            height = max(height, 112)
        else:
            height = max(height, 170)
    st.dataframe(df, use_container_width=True, hide_index=True, height=height)


def render_chart(fig, width_mode="standard"):
    st.markdown("<div class='chart-spacer'></div>", unsafe_allow_html=True)
    if width_mode == "full":
        _, center, _ = st.columns([0.02, 0.96, 0.02])
    elif width_mode == "narrow":
        _, center, _ = st.columns([0.17, 0.66, 0.17])
    else:
        _, center, _ = st.columns([0.06, 0.88, 0.06])

    with center:
        st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def render_kpi_cards(portfolio_summary, benchmark_summary, benchmark_col):
    cols = st.columns(5, gap="medium")
    cols[0].metric(
        "CAGR",
        _fmt_pct(portfolio_summary["cagr"]),
        delta=f"vs {benchmark_col}: {_fmt_pct(portfolio_summary['cagr'] - benchmark_summary['cagr'])}",
        delta_color="normal",
    )
    cols[1].metric(
        "Total Return",
        _fmt_pct(portfolio_summary["total_return"]),
        delta=f"Active: {_fmt_pct(portfolio_summary['total_return'] - benchmark_summary['total_return'])}",
        delta_color="normal",
    )
    cols[2].metric(
        "Sharpe Ratio",
        _fmt_ratio(portfolio_summary["sharpe"]),
        delta=f"{benchmark_col}: {_fmt_ratio(benchmark_summary['sharpe'])}",
        delta_color="normal",
    )
    cols[3].metric(
        "Max Drawdown",
        _fmt_pct(portfolio_summary["max_drawdown"]),
        delta=f"{benchmark_col}: {_fmt_pct(benchmark_summary['max_drawdown'])}",
        delta_color="normal",
    )
    cols[4].metric(
        "Portfolio Beta",
        _fmt_ratio(portfolio_summary["beta"]),
        delta=f"Alpha: {_fmt_pct(portfolio_summary['alpha'])}",
        delta_color="normal",
    )


def render_risk_cards(portfolio_summary, benchmark_summary, benchmark_col):
    cols = st.columns(4, gap="medium")
    cols[0].metric(
        "Volatility",
        _fmt_pct(portfolio_summary["volatility"]),
        delta=f"{benchmark_col}: {_fmt_pct(benchmark_summary['volatility'])}",
        delta_color="normal",
    )
    cols[1].metric(
        "Sortino Ratio",
        _fmt_ratio(portfolio_summary["sortino"]),
        delta=f"Sharpe: {_fmt_ratio(portfolio_summary['sharpe'])}",
        delta_color="normal",
    )
    cols[2].metric(
        "VaR 95%",
        _fmt_pct(portfolio_summary["var_95"]),
        delta="1-Day Historical",
        delta_color="normal",
    )
    cols[3].metric(
        "CVaR 95%",
        _fmt_pct(portfolio_summary["cvar_95"]),
        delta="Expected Tail Loss",
        delta_color="normal",
    )


def format_date_axis(ax):
    locator = mdates.AutoDateLocator(minticks=5, maxticks=8)
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)


def style_axis(ax, percent_y=False, percent_x=False):
    ax.grid(True, alpha=0.24, color=COLORS["grid"], linewidth=0.9)
    ax.tick_params(colors=COLORS["muted"], labelsize=9.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLORS["border"])
    ax.spines["bottom"].set_color(COLORS["border"])
    ax.margins(x=0.02)

    if percent_y:
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    if percent_x:
        ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))


def build_performance_figure(
    analysis_df,
    portfolio_drawdowns,
    benchmark_drawdowns,
    benchmark_col,
    rebalance_frequency,
    weighting,
    stocks,
    fill_drawdown,
    portfolio_summary,
    benchmark_summary,
):
    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(10.5, 5.85),
        sharex=True,
        gridspec_kw={"height_ratios": [3.0, 1.05], "hspace": 0.08},
    )

    ax1.plot(
        analysis_df["Date"],
        analysis_df["Portfolio Value"],
        label="Portfolio",
        linewidth=2.35,
        color=COLORS["portfolio"],
    )
    ax1.plot(
        analysis_df["Date"],
        analysis_df["Benchmark Value"],
        label=f"Benchmark ({benchmark_col})",
        linewidth=1.95,
        linestyle="--",
        color=COLORS["benchmark"],
    )

    last_date = analysis_df["Date"].iloc[-1]
    last_port = analysis_df["Portfolio Value"].iloc[-1]
    last_bench = analysis_df["Benchmark Value"].iloc[-1]

    ax1.scatter(last_date, last_port, color=COLORS["portfolio"], s=28, zorder=4)
    ax1.scatter(last_date, last_bench, color=COLORS["benchmark"], s=28, zorder=4)

    ax1.annotate(
        f"Portfolio {last_port:.2f}x",
        xy=(last_date, last_port),
        xytext=(-46, 12),
        textcoords="offset points",
        ha="right",
        fontsize=8.7,
        color=COLORS["portfolio"],
        bbox=dict(boxstyle="round,pad=0.22", fc="white", ec=COLORS["border"], alpha=0.96),
    )
    ax1.annotate(
        f"{benchmark_col} {last_bench:.2f}x",
        xy=(last_date, last_bench),
        xytext=(-46, -18),
        textcoords="offset points",
        ha="right",
        fontsize=8.7,
        color=COLORS["benchmark"],
        bbox=dict(boxstyle="round,pad=0.22", fc="white", ec=COLORS["border"], alpha=0.96),
    )

    ax1.set_title(
        "Growth of $1 and Benchmark Comparison",
        loc="left",
        pad=10,
        fontsize=13.2,
        color=COLORS["text"],
    )
    ax1.set_ylabel("Cumulative Value")
    ax1.legend(loc="upper left", fontsize=9.2, ncol=2)

    info_text = (
        f"CAGR {_fmt_pct(portfolio_summary['cagr'])} | Sharpe {_fmt_ratio(portfolio_summary['sharpe'])} | "
        f"Beta {_fmt_ratio(portfolio_summary['beta'])}\n"
        f"Benchmark {benchmark_col} | Rebalance {rebalance_frequency}"
    )
    ax1.text(
        0.985,
        0.97,
        info_text,
        transform=ax1.transAxes,
        va="top",
        ha="right",
        fontsize=8.8,
        color=COLORS["muted"],
        bbox=dict(boxstyle="round,pad=0.34", fc="white", ec=COLORS["border"], alpha=0.95),
    )

    ax2.plot(
        analysis_df["Date"],
        portfolio_drawdowns,
        color=COLORS["drawdown"],
        linewidth=1.75,
        label="Portfolio Drawdown",
    )
    ax2.plot(
        analysis_df["Date"],
        benchmark_drawdowns,
        color=COLORS["warning"],
        linewidth=1.45,
        linestyle="--",
        label=f"{benchmark_col} Drawdown",
    )
    ax2.axhline(0, color=COLORS["muted"], linewidth=0.9, alpha=0.6)

    if fill_drawdown:
        ax2.fill_between(
            analysis_df["Date"],
            portfolio_drawdowns,
            0,
            color=COLORS["drawdown"],
            alpha=0.11,
        )

    ax2.scatter(
        portfolio_summary["max_drawdown_date"],
        portfolio_summary["max_drawdown"],
        color=COLORS["drawdown"],
        s=26,
        zorder=4,
    )
    ax2.scatter(
        benchmark_summary["max_drawdown_date"],
        benchmark_summary["max_drawdown"],
        color=COLORS["warning"],
        s=26,
        zorder=4,
    )

    ax2.annotate(
        f"Portfolio MDD: {_fmt_pct(portfolio_summary['max_drawdown'])}",
        xy=(portfolio_summary["max_drawdown_date"], portfolio_summary["max_drawdown"]),
        xytext=(8, 8),
        textcoords="offset points",
        fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.22", fc="white", ec=COLORS["border"], alpha=0.95),
    )
    ax2.annotate(
        f"{benchmark_col} MDD: {_fmt_pct(benchmark_summary['max_drawdown'])}",
        xy=(benchmark_summary["max_drawdown_date"], benchmark_summary["max_drawdown"]),
        xytext=(8, -18),
        textcoords="offset points",
        fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.22", fc="white", ec=COLORS["border"], alpha=0.95),
    )

    ax2.set_ylabel("Drawdown")
    ax2.set_xlabel("Date")
    ax2.legend(loc="lower left", ncol=2, fontsize=8.8)

    format_date_axis(ax2)
    style_axis(ax1)
    style_axis(ax2, percent_y=True)
    fig.tight_layout(pad=1.0)

    return fig


def build_correlation_heatmap(corr, stocks):
    fig, ax = plt.subplots(figsize=(6.35, 4.55))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1.0, vmax=1.0, aspect="auto")

    ax.set_xticks(np.arange(len(stocks)))
    ax.set_yticks(np.arange(len(stocks)))
    ax.set_xticklabels(stocks, rotation=28, ha="right", fontsize=9.5)
    ax.set_yticklabels(stocks, fontsize=9.5)
    ax.set_title("Return Correlation Heatmap", loc="left", pad=10, fontsize=13.0, color=COLORS["text"])

    for i in range(len(stocks)):
        for j in range(len(stocks)):
            value = corr.iloc[i, j]
            text_color = "white" if abs(value) >= 0.55 else COLORS["text"]
            ax.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=8.9,
                fontweight="bold",
            )

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_xticks(np.arange(-0.5, len(stocks), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(stocks), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.15)
    ax.tick_params(which="minor", bottom=False, left=False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.ax.set_ylabel("Correlation", rotation=270, labelpad=15)
    cbar.ax.tick_params(labelsize=8.5)

    fig.tight_layout(pad=0.9)
    return fig


def build_frontier_figure(
    frontier_vols,
    frontier_returns,
    sharpe_list,
    max_sharpe_idx,
    min_vol_idx,
    current_vol,
    current_return,
    effective_max_pct,
):
    fig, ax = plt.subplots(figsize=(8.95, 5.4))

    scatter = ax.scatter(
        frontier_vols,
        frontier_returns,
        c=sharpe_list,
        cmap="Blues",
        s=16,
        alpha=0.4,
        edgecolors="none",
        label="Constrained Portfolios",
    )

    ax.scatter(
        current_vol,
        current_return,
        color=COLORS["accent"],
        s=90,
        edgecolors="white",
        linewidths=1.1,
        label="Current Portfolio",
        zorder=5,
    )
    ax.scatter(
        frontier_vols[max_sharpe_idx],
        frontier_returns[max_sharpe_idx],
        color=COLORS["success"],
        marker="*",
        s=210,
        label="Max Sharpe",
        zorder=6,
    )
    ax.scatter(
        frontier_vols[min_vol_idx],
        frontier_returns[min_vol_idx],
        color=COLORS["warning"],
        marker="D",
        s=92,
        label="Min Vol",
        zorder=6,
    )

    ax.annotate(
        "Current",
        xy=(current_vol, current_return),
        xytext=(10, -10),
        textcoords="offset points",
        fontsize=8.8,
        color=COLORS["accent"],
        fontweight="bold",
    )
    ax.annotate(
        "Max Sharpe",
        xy=(frontier_vols[max_sharpe_idx], frontier_returns[max_sharpe_idx]),
        xytext=(10, 8),
        textcoords="offset points",
        fontsize=8.8,
        color=COLORS["success"],
        fontweight="bold",
    )
    ax.annotate(
        "Min Vol",
        xy=(frontier_vols[min_vol_idx], frontier_returns[min_vol_idx]),
        xytext=(10, -15),
        textcoords="offset points",
        fontsize=8.8,
        color=COLORS["warning"],
        fontweight="bold",
    )

    ax.set_title(
        f"Efficient Frontier ({MIN_WEIGHT_PCT:.0f}% to {effective_max_pct:.0f}% per asset)",
        loc="left",
        pad=10,
        fontsize=13.0,
        color=COLORS["text"],
    )
    ax.set_xlabel("Annualized Volatility")
    ax.set_ylabel("Annualized Return")
    ax.legend(loc="upper left", ncol=2, fontsize=8.8)

    style_axis(ax, percent_y=True, percent_x=True)

    cbar = fig.colorbar(scatter, ax=ax, fraction=0.047, pad=0.03)
    cbar.ax.set_ylabel("Sharpe Ratio", rotation=270, labelpad=15)
    cbar.ax.tick_params(labelsize=8.5)

    fig.tight_layout(pad=0.95)
    return fig


def build_rolling_metrics_figure(rolling_df, benchmark_col):
    fig, axes = plt.subplots(3, 1, figsize=(10.4, 5.95), sharex=True)
    fig.subplots_adjust(hspace=0.16)

    axes[0].plot(
        rolling_df["Date"],
        rolling_df["Rolling Volatility"],
        color=COLORS["portfolio"],
        linewidth=1.9,
        label="Portfolio",
    )
    axes[0].plot(
        rolling_df["Date"],
        rolling_df["Benchmark Rolling Volatility"],
        color=COLORS["benchmark"],
        linewidth=1.6,
        linestyle="--",
        label=benchmark_col,
    )
    axes[0].set_title(
        f"Rolling {ROLLING_WINDOW}-Day Risk Dashboard",
        loc="left",
        pad=9,
        fontsize=13.0,
        color=COLORS["text"],
    )
    axes[0].set_ylabel("Volatility")
    axes[0].legend(loc="upper left", ncol=2, fontsize=8.7)

    axes[1].plot(
        rolling_df["Date"],
        rolling_df["Rolling Sharpe"],
        color=COLORS["accent"],
        linewidth=1.85,
    )
    axes[1].axhline(0, color=COLORS["muted"], linestyle="--", linewidth=0.9)
    axes[1].set_ylabel("Sharpe")

    axes[2].plot(
        rolling_df["Date"],
        rolling_df["Rolling Beta"],
        color=COLORS["warning"],
        linewidth=1.85,
    )
    axes[2].axhline(1.0, color=COLORS["muted"], linestyle="--", linewidth=0.9)
    axes[2].set_ylabel("Beta")
    axes[2].set_xlabel("Date")

    style_axis(axes[0], percent_y=True)
    style_axis(axes[1])
    style_axis(axes[2])
    format_date_axis(axes[2])

    fig.tight_layout(pad=0.95)
    return fig


def main():
    st.set_page_config(
        page_title="Portfolio Optimization Dashboard",
        page_icon="📈",
        layout="wide",
    )
    apply_custom_css()
    initialize_session_state()

    st.markdown(
        """
        <div class="hero-shell">
            <div class="hero-title">Portfolio Optimization Dashboard</div>
            <div class="hero-subtitle">
                Institutional-grade portfolio analytics for performance, risk, optimization, and execution planning.
                Compare your portfolio against a benchmark, inspect drawdowns and tail risk, and evaluate constrained efficient-frontier outcomes.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown("<div class='sidebar-header'>Portfolio Controls</div>", unsafe_allow_html=True)

    with st.sidebar.expander("Universe & Weights", expanded=True):
        st.markdown(
            "<div class='sidebar-note'>Select between 2 and 20 tickers, add or remove names dynamically, or randomize a five-stock starting basket from a diversified universe.</div>",
            unsafe_allow_html=True,
        )

        add_col, remove_col, random_col, equal_col = st.columns(4, gap="small")
        with add_col:
            if st.button("+ Add", key="add_ticker_btn", use_container_width=True):
                add_ticker_to_selection()
                st.rerun()
        with remove_col:
            if st.button("- Remove", key="remove_ticker_btn", use_container_width=True):
                remove_ticker_from_selection()
                st.rerun()
        with random_col:
            if st.button("Randomize 5 Tickers", key="randomize_btn", use_container_width=True):
                randomize_five_tickers()
                st.rerun()
        with equal_col:
            if st.button("Equal Weight", key="equal_weight_btn", use_container_width=True):
                set_equal_weight_selection(st.session_state["selected_tickers"])
                st.rerun()

        st.multiselect(
            "Selected Tickers",
            options=TICKER_UNIVERSE,
            key="selected_tickers",
            max_selections=MAX_SELECTED_TICKERS,
        )

        st.markdown(
            f"<div class='sidebar-note'>Selected tickers: {len(st.session_state['selected_tickers'])} / {MAX_SELECTED_TICKERS}</div>",
            unsafe_allow_html=True,
        )

        try:
            stocks = sanitize_tickers(st.session_state["selected_tickers"])
        except ValueError as exc:
            stop_with_error(str(exc))

        raw_weighting = {}
        for ticker in stocks:
            raw_weighting[ticker] = st.slider(
                f"{ticker} Weight (%)",
                min_value=0.0,
                max_value=100.0,
                step=1.0,
                key=f"weight_{ticker}",
            )

    with spinner_context("Loading price history for the selected universe..."):
        try:
            price_history_df = get_price_history_cached(stocks, price_col="Close")
        except ValueError as exc:
            stop_with_error(str(exc))

    min_year = int(price_history_df["Date"].dt.year.min())
    max_year = int(price_history_df["Date"].dt.year.max())
    default_year_range = (max(min_year, 2020), max_year)

    if st.session_state["year_range"] is None:
        st.session_state["year_range"] = default_year_range
    else:
        start_year, end_year = st.session_state["year_range"]
        start_year = min(max(start_year, min_year), max_year)
        end_year = min(max(end_year, min_year), max_year)
        if start_year > end_year:
            start_year, end_year = default_year_range
        st.session_state["year_range"] = (start_year, end_year)

    with st.sidebar.expander("Backtest Settings", expanded=True):
        st.markdown(
            "<div class='sidebar-note'>Control the evaluation window, benchmark, rebalance cadence, and display toggles used throughout the dashboard.</div>",
            unsafe_allow_html=True,
        )

        year_range = st.slider(
            "Years",
            min_value=min_year,
            max_value=max_year,
            value=st.session_state["year_range"],
            key="year_range",
        )

        benchmark_col = st.selectbox(
            "Benchmark",
            options=["SPY", "VFV.TO", "XEQT.TO"],
            key="benchmark_col",
        )

        rebalance_frequency = st.selectbox(
            "Rebalance",
            options=["Daily", "Monthly", "Quarterly", "Annual", "Buy & Hold"],
            key="rebalance_frequency",
        )

        normalize_weights = st.checkbox(
            "Normalize weights to 100%",
            key="normalize_weights",
        )

        fill_drawdown = st.checkbox(
            "Fill drawdown area",
            key="fill_drawdown",
        )

    cap_floor_pct = float(max(100 / len(stocks), MIN_WEIGHT_PCT))
    with st.sidebar.expander("Execution & Constraints", expanded=True):
        st.markdown(
            "<div class='sidebar-note'>Set the capital base, share rounding, trading-cost assumptions, and institutional concentration limits.</div>",
            unsafe_allow_html=True,
        )

        investment_amount = st.number_input(
            "Invest $",
            min_value=0.0,
            step=1000.0,
            key="investment_amount",
        )

        whole_shares_only = st.checkbox(
            "Whole shares only",
            key="whole_shares_only",
        )

        transaction_cost_bps = st.slider(
            "Transaction Cost (bps)",
            min_value=0.0,
            max_value=100.0,
            step=1.0,
            key="transaction_cost_bps",
        )

        if cap_floor_pct > MAX_WEIGHT_PCT:
            st.markdown(
                "<div class='sidebar-note'>The current ticker count makes the single-stock cap infeasible. Add more tickers or relax the cap assumptions.</div>",
                unsafe_allow_html=True,
            )
            single_stock_cap_pct = cap_floor_pct
        else:
            st.session_state["single_stock_cap_pct"] = min(
                max(st.session_state["single_stock_cap_pct"], cap_floor_pct),
                MAX_WEIGHT_PCT,
            )
            single_stock_cap_pct = st.slider(
                "Single-Stock Cap (%)",
                min_value=cap_floor_pct,
                max_value=float(MAX_WEIGHT_PCT),
                step=1.0,
                key="single_stock_cap_pct",
            )

    if cap_floor_pct > MAX_WEIGHT_PCT:
        minimum_feasible_tickers = int(np.ceil(100 / MAX_WEIGHT_PCT))
        stop_with_error(
            f"Infeasible constraints: with {len(stocks)} tickers and a {MAX_WEIGHT_PCT:.0f}% maximum single-stock cap, the portfolio cannot sum to 100%. Add at least {minimum_feasible_tickers} tickers."
        )

    if investment_amount <= 0:
        stop_with_error("Investment amount must be greater than 0.")

    if transaction_cost_bps < 0:
        stop_with_error("Transaction cost bps must be 0 or greater.")

    total_weight = sum(raw_weighting.values())
    if np.isclose(total_weight, 0):
        stop_with_error("All weights are 0. Assign at least one ticker a positive weight.")

    if normalize_weights:
        try:
            weighting = normalize_weighting_dict(raw_weighting)
        except ValueError as exc:
            stop_with_error(str(exc))
    else:
        if not np.isclose(total_weight, 100):
            stop_with_error(f"Weights must sum to 100 when normalization is off. Current sum: {total_weight:.1f}")
        weighting = raw_weighting.copy()

    constraints_ok, constraint_message = validate_weighting(
        weighting,
        min_weight_pct=MIN_WEIGHT_PCT,
        single_stock_cap_pct=single_stock_cap_pct,
    )
    if not constraints_ok:
        stop_with_error("Portfolio constraints violated. " + constraint_message)

    start_date = pd.Timestamp(f"{year_range[0]}-01-01")
    end_date = pd.Timestamp(f"{year_range[1]}-12-31")

    with spinner_context("Running portfolio analytics..."):
        selected_price_df = price_history_df[
            (price_history_df["Date"] >= start_date) &
            (price_history_df["Date"] <= end_date)
        ].copy()

        if selected_price_df.empty:
            stop_with_error("No price data available for the selected year range.")

        try:
            benchmark_history = get_benchmark_history_cached(benchmark_col, start_date, end_date)
        except ValueError as exc:
            stop_with_error(str(exc))

        try:
            portfolio_df = simulate_rebalanced_portfolio(
                price_df=selected_price_df,
                weighting=weighting,
                rebalance_frequency=rebalance_frequency,
                stocks=stocks,
            )

            analysis_df, return_frame = prepare_analysis_data(
                portfolio_df=portfolio_df,
                start_date=start_date,
                end_date=end_date,
                stocks=stocks,
                benchmark_col=benchmark_col,
                benchmark_history=benchmark_history,
            )
        except ValueError as exc:
            stop_with_error(str(exc))

        try:
            allocation_df, allocation_summary = build_allocation_table(
                price_df=price_history_df,
                stocks=stocks,
                weighting=weighting,
                investment_amount=investment_amount,
                whole_shares_only=whole_shares_only,
                transaction_cost_bps=transaction_cost_bps,
            )
        except ValueError as exc:
            stop_with_error(str(exc))

        portfolio_summary = summarize_series(
            analysis_df,
            return_frame,
            value_col="Portfolio Value",
            return_col="Portfolio Daily Return",
            risk_free_rate=RISK_FREE_RATE,
        )

        benchmark_summary = summarize_series(
            analysis_df,
            return_frame,
            value_col="Benchmark Value",
            return_col="Benchmark Daily Return",
            benchmark_return_col=None,
            risk_free_rate=RISK_FREE_RATE,
        )

        perf_table = portfolio_performance_table(
            analysis_df=analysis_df,
            return_frame=return_frame,
            stocks=stocks,
            weighting=weighting,
            benchmark_col=benchmark_col,
        )

        asset_returns = selected_price_df[stocks].pct_change().dropna()
        if asset_returns.empty:
            stop_with_error("Not enough return history to build optimization results for the selected year range.")

        cov_matrix = asset_returns[stocks].cov().values

        try:
            weights_list, frontier_returns, frontier_vols, sharpe_list = eff_frontier(
                stocks=stocks,
                returns=asset_returns,
                cov_matrix=cov_matrix,
                num_ports=4000,
                single_stock_cap_pct=single_stock_cap_pct,
            )
        except ValueError as exc:
            stop_with_error(str(exc))

        max_sharpe_idx = int(np.nanargmax(sharpe_list))
        min_vol_idx = int(np.nanargmin(frontier_vols))

        max_sharpe_weights = weights_list[max_sharpe_idx]
        min_vol_weights = weights_list[min_vol_idx]

        max_sharpe_weighting = normalize_weighting_dict({
            stock: weight * 100 for stock, weight in zip(stocks, max_sharpe_weights)
        })
        min_vol_weighting = normalize_weighting_dict({
            stock: weight * 100 for stock, weight in zip(stocks, min_vol_weights)
        })

        mean_returns = asset_returns[stocks].mean().values
        current_weights_vec = np.array([weighting[ticker] / 100 for ticker in stocks], dtype=float)

        current_stats = portfolio_stats(current_weights_vec, mean_returns, cov_matrix, rf=RISK_FREE_RATE)
        max_sharpe_stats = portfolio_stats(max_sharpe_weights, mean_returns, cov_matrix, rf=RISK_FREE_RATE)
        min_vol_stats = portfolio_stats(min_vol_weights, mean_returns, cov_matrix, rf=RISK_FREE_RATE)

        opt_df = pd.DataFrame([
            {
                "Portfolio": "Max Sharpe",
                **{ticker: f"{max_sharpe_weighting[ticker]:.2f}%" for ticker in stocks},
                "Annual Return": _fmt_pct(max_sharpe_stats["return"]),
                "Volatility": _fmt_pct(max_sharpe_stats["volatility"]),
                "Sharpe": _fmt_ratio(max_sharpe_stats["sharpe"]),
            },
            {
                "Portfolio": "Min Vol",
                **{ticker: f"{min_vol_weighting[ticker]:.2f}%" for ticker in stocks},
                "Annual Return": _fmt_pct(min_vol_stats["return"]),
                "Volatility": _fmt_pct(min_vol_stats["volatility"]),
                "Sharpe": _fmt_ratio(min_vol_stats["sharpe"]),
            },
        ])

        comparison_df = build_strategy_comparison_table(
            price_df=selected_price_df,
            stocks=stocks,
            current_weighting=weighting,
            max_sharpe_weighting=max_sharpe_weighting,
            min_vol_weighting=min_vol_weighting,
            benchmark_col=benchmark_col,
            rebalance_frequency=rebalance_frequency,
            benchmark_history=benchmark_history,
        )

        rolling_df = rolling_metrics_table(return_frame, window=ROLLING_WINDOW)

    allocation_caption = " | ".join([f"{ticker} {weighting[ticker]:.1f}%" for ticker in stocks])
    effective_max_pct = _effective_max_weight_pct(single_stock_cap_pct)

    summary_band(
        f"<strong>Current Allocation:</strong> {allocation_caption} &nbsp;&nbsp;&bull;&nbsp;&nbsp; "
        f"<strong>Benchmark:</strong> {benchmark_col} &nbsp;&nbsp;&bull;&nbsp;&nbsp; "
        f"<strong>Rebalance:</strong> {rebalance_frequency} &nbsp;&nbsp;&bull;&nbsp;&nbsp; "
        f"<strong>Capital:</strong> {_fmt_currency(investment_amount)} &nbsp;&nbsp;&bull;&nbsp;&nbsp; "
        f"<strong>Single-Name Cap:</strong> {effective_max_pct:.0f}%"
    )

    render_kpi_cards(portfolio_summary, benchmark_summary, benchmark_col)

    section_header(
        "Portfolio Performance",
        "Portfolio-versus-benchmark growth and drawdown behavior, with cleaner benchmark labeling and a tighter presentation of performance context.",
    )

    disclaimer_caption("For educational purposes only. This dashboard is not financial advice.")
    source_caption("Source: market price history is retrieved via yfinance and may be delayed, revised, incomplete, or subject to third-party data limitations.")

    perf_fig = build_performance_figure(
        analysis_df=analysis_df,
        portfolio_drawdowns=(analysis_df["Portfolio Value"] - analysis_df["Portfolio Value"].cummax()) / analysis_df["Portfolio Value"].cummax(),
        benchmark_drawdowns=(analysis_df["Benchmark Value"] - analysis_df["Benchmark Value"].cummax()) / analysis_df["Benchmark Value"].cummax(),
        benchmark_col=benchmark_col,
        rebalance_frequency=rebalance_frequency,
        weighting=weighting,
        stocks=stocks,
        fill_drawdown=fill_drawdown,
        portfolio_summary=portfolio_summary,
        benchmark_summary=benchmark_summary,
    )
    render_chart(perf_fig, width_mode="standard")
    table_caption("The top panel shows normalized growth of $1 alongside the benchmark. The lower panel isolates drawdowns and highlights each series' worst peak-to-trough loss.")

    section_header(
        "Allocation Analysis",
        "Execution-aware portfolio sizing and institutional summary metrics, organized for faster review and cleaner decision-making.",
    )

    st.markdown("#### Execution-Aware Allocation")
    table_caption(
        f"Investment amount {_fmt_currency(investment_amount)} | Gross invested {_fmt_currency(allocation_summary['gross_invested'])} | "
        f"Transaction costs {_fmt_currency(allocation_summary['transaction_cost'])} | Leftover cash {_fmt_currency(allocation_summary['leftover_cash'])}"
    )
    render_dataframe(format_allocation_table(allocation_df), max_rows_visible=min(len(allocation_df), 12))
    small_note("This table includes share rounding and entry costs. Backtests and optimizer outputs below still use frictionless target weights.")

    st.markdown("#### Performance Metrics")
    table_caption(
        f"Constraint set: min {MIN_WEIGHT_PCT:.0f}% | max {MAX_WEIGHT_PCT:.0f}% | active stock cap {effective_max_pct:.0f}%"
    )
    render_dataframe(format_perf_table(perf_table), max_rows_visible=11)

    section_header(
        "Optimization Results",
        "Constrained frontier visualization, optimizer recommendations, and a direct comparison between the current portfolio and optimized alternatives.",
    )

    frontier_fig = build_frontier_figure(
        frontier_vols=frontier_vols,
        frontier_returns=frontier_returns,
        sharpe_list=sharpe_list,
        max_sharpe_idx=max_sharpe_idx,
        min_vol_idx=min_vol_idx,
        current_vol=current_stats["volatility"],
        current_return=current_stats["return"],
        effective_max_pct=effective_max_pct,
    )
    render_chart(frontier_fig, width_mode="standard")
    table_caption("The efficient frontier respects portfolio concentration rules and highlights the constrained maximum-Sharpe and minimum-volatility solutions.")

    opt_left, opt_right = st.columns([1.0, 1.0], gap="large")

    with opt_left:
        st.markdown("#### Optimized Portfolio Weights")
        render_dataframe(opt_df, max_rows_visible=2)

    with opt_right:
        st.markdown("#### Optimizer Snapshot")
        top_a, top_b = st.columns(2, gap="medium")
        bot_a, bot_b = st.columns(2, gap="medium")

        top_a.metric(
            "Max Sharpe Return",
            _fmt_pct(max_sharpe_stats["return"]),
            delta=f"Vol {_fmt_pct(max_sharpe_stats['volatility'])}",
            delta_color="normal",
        )
        top_b.metric(
            "Max Sharpe Ratio",
            _fmt_ratio(max_sharpe_stats["sharpe"]),
            delta="Constrained",
            delta_color="normal",
        )
        bot_a.metric(
            "Min Vol Return",
            _fmt_pct(min_vol_stats["return"]),
            delta=f"Vol {_fmt_pct(min_vol_stats['volatility'])}",
            delta_color="normal",
        )
        bot_b.metric(
            "Current Portfolio",
            _fmt_ratio(current_stats["sharpe"]),
            delta=f"Sharpe | Return {_fmt_pct(current_stats['return'])}",
            delta_color="normal",
        )

    st.markdown("#### Current vs Optimized Comparison")
    table_caption("Side-by-side comparison of current, max-Sharpe, and minimum-volatility portfolios across return, drawdown, and tail-risk measures.")
    render_dataframe(comparison_df, max_rows_visible=3)

    section_header(
        "Risk Metrics",
        "Rolling diagnostics and tail-risk summaries that help evaluate stability, sensitivity, and downside exposure over time.",
    )

    render_risk_cards(portfolio_summary, benchmark_summary, benchmark_col)

    if rolling_df.empty:
        st.error(f"Not enough data for {ROLLING_WINDOW}-day rolling metrics.")
    else:
        rolling_fig = build_rolling_metrics_figure(rolling_df, benchmark_col)
        render_chart(rolling_fig, width_mode="standard")
        table_caption(f"Rolling {ROLLING_WINDOW}-day panels track annualized volatility, rolling Sharpe ratio, and rolling beta versus the selected benchmark.")

    section_header(
        "Correlation Analysis",
        "Pairwise return relationships across the selected holdings to help assess diversification quality and clustering risk.",
    )

    corr = asset_returns.corr()
    corr_fig = build_correlation_heatmap(corr, stocks)
    render_chart(corr_fig, width_mode="narrow")
    table_caption("Higher positive readings indicate more synchronous return behavior, while lower or negative readings signal better diversification potential.")


if __name__ == "__main__":
    try:
        main()
    except DashboardHaltError as exc:
        if _has_streamlit_context():
            raise
        print(f"Dashboard error: {exc}")