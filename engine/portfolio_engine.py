import logging

import numpy as np
import pandas as pd

from config import (
    INITIAL_CAPITAL,
    TRANSACTION_COST,
    TARGET_VOL,
    MAX_WEIGHT,
    MAX_LEVERAGE,
    VOL_LOOKBACK,
    VOL_MIN_PERIODS,
    RISK_FREE_RATE,
    RANDOM_SEED,
)

logger = logging.getLogger(__name__)

PERIODS_PER_YEAR = 12  # 21-day rebalance ~ monthly


# =====================================================
# Performance Metrics
# =====================================================

def compute_sharpe(monthly_returns, risk_free_rate=RISK_FREE_RATE):
    """Annualised Sharpe ratio of monthly returns (net of a risk-free rate)."""
    returns = pd.Series(monthly_returns, dtype=float)

    if len(returns) < 2:
        return 0.0

    annual_return = (1 + returns.mean()) ** PERIODS_PER_YEAR - 1
    annual_vol = returns.std() * np.sqrt(PERIODS_PER_YEAR)

    if not np.isfinite(annual_vol) or annual_vol == 0:
        return 0.0

    return (annual_return - risk_free_rate) / annual_vol


def compute_max_drawdown(monthly_returns):
    returns = pd.Series(monthly_returns, dtype=float)
    if len(returns) == 0:
        return 0.0
    equity = (1 + returns).cumprod()
    return float((equity / equity.cummax() - 1).min())


# =====================================================
# Volatility Targeting (causal, no leverage)
# =====================================================

def apply_vol_target(monthly_returns):
    """
    Scales each period's return by TARGET_VOL / (trailing realized vol).

    - Realized vol uses only *prior* periods (last VOL_LOOKBACK months), so
      there is no look-ahead.
    - Scale is capped at MAX_LEVERAGE (default 1.0): the overlay can reduce
      exposure but never lever up. Remaining capital is held in cash (0%).
    - Until VOL_MIN_PERIODS of history exist, returns are left unscaled.
    """
    returns = pd.Series(monthly_returns, dtype=float)
    scaled = returns.copy()

    for t in range(len(returns)):
        history = returns.iloc[max(0, t - VOL_LOOKBACK):t]

        if len(history) < VOL_MIN_PERIODS:
            scale = 1.0
        else:
            realized_vol = history.std() * np.sqrt(PERIODS_PER_YEAR)
            if not np.isfinite(realized_vol) or realized_vol <= 0:
                scale = 1.0
            else:
                scale = min(TARGET_VOL / realized_vol, MAX_LEVERAGE)

        scaled.iloc[t] = returns.iloc[t] * scale

    return scaled


# =====================================================
# Long Top-N Portfolio Simulation
# =====================================================

def simulate_long_top_n(alpha_df, top_n=3, verbose=True):
    """
    Each rebalance date: go long the top_n assets by probability, equal
    weight (capped at MAX_WEIGHT; any remainder is cash).

    Transaction cost = TRANSACTION_COST x turnover, where turnover is the sum
    of absolute weight changes vs the previous rebalance (the first rebalance
    buys from cash). Weight drift between rebalances is ignored.
    """

    dates, raw_returns, bench_returns, turnovers = [], [], [], []
    prev_weights = {}

    for date, group in alpha_df.groupby("date"):

        selected = group.sort_values(
            "probability", ascending=False, kind="mergesort"
        ).head(top_n)

        if len(selected) == 0:
            continue

        weight = min(1.0 / top_n, MAX_WEIGHT)
        new_weights = {t: weight for t in selected["ticker"]}

        turnover = sum(
            abs(new_weights.get(t, 0.0) - prev_weights.get(t, 0.0))
            for t in set(new_weights) | set(prev_weights)
        )

        gross_return = float((selected["forward_return"] * weight).sum())
        net_return = gross_return - TRANSACTION_COST * turnover

        dates.append(date)
        raw_returns.append(net_return)
        bench_returns.append(float(group["forward_return"].mean()))
        turnovers.append(turnover)
        prev_weights = new_weights

    if not raw_returns:
        return INITIAL_CAPITAL, pd.Series(dtype=float)

    monthly_returns = apply_vol_target(pd.Series(raw_returns, index=dates))

    capital = INITIAL_CAPITAL * float((1 + monthly_returns).prod())

    if verbose:
        total_return = (capital / INITIAL_CAPITAL - 1) * 100
        bench_series = pd.Series(bench_returns, index=dates)
        bench_capital = INITIAL_CAPITAL * float((1 + bench_series).prod())

        logger.info(f"Periods simulated: {len(monthly_returns)}")
        logger.info(f"Final Capital: {capital:,.2f}")
        logger.info(f"Total Return (%): {total_return:.2f}%")
        logger.info(f"Sharpe Ratio: {compute_sharpe(monthly_returns):.3f}")
        logger.info(f"Max Drawdown: {compute_max_drawdown(monthly_returns) * 100:.2f}%")
        logger.info(f"Average Turnover per Rebalance: {np.mean(turnovers):.2f}")
        logger.info(
            "Equal-Weight Universe Benchmark (no costs): "
            f"Final Capital {bench_capital:,.2f}, "
            f"Sharpe {compute_sharpe(bench_series):.3f}"
        )

    return capital, monthly_returns


# =====================================================
# Permutation Test
# =====================================================

def permutation_test(alpha_df, top_n=3, n_tests=500):
    """
    Tests whether the model's stock *selection* beats chance.

    Within each date, forward returns are randomly reassigned among the
    assets, which keeps each date's market-level return intact but breaks any
    link between the model's probabilities and the returns. The p-value is
    (1 + #shuffled Sharpes >= real Sharpe) / (n_tests + 1).
    """

    rng = np.random.default_rng(RANDOM_SEED)

    alpha_df = alpha_df.reset_index(drop=True)

    _, real_returns = simulate_long_top_n(
        alpha_df, top_n=top_n, verbose=False
    )
    real_sharpe = compute_sharpe(real_returns)

    date_groups = list(alpha_df.groupby("date").indices.values())
    base_returns = alpha_df["forward_return"].values

    shuffled_sharpes = []

    for _ in range(n_tests):

        shuffled = base_returns.copy()
        for idx in date_groups:
            shuffled[idx] = rng.permutation(base_returns[idx])

        shuffled_df = alpha_df.copy()
        shuffled_df["forward_return"] = shuffled

        _, shuffled_returns = simulate_long_top_n(
            shuffled_df, top_n=top_n, verbose=False
        )
        shuffled_sharpes.append(compute_sharpe(shuffled_returns))

    shuffled_sharpes = np.array(shuffled_sharpes)

    p_value = (1 + np.sum(shuffled_sharpes >= real_sharpe)) / (n_tests + 1)

    logger.info(f"Real Sharpe: {real_sharpe:.3f}")
    logger.info(f"Average Shuffled Sharpe: {shuffled_sharpes.mean():.3f}")
    logger.info(f"P-value: {p_value:.4f}")

    return real_sharpe, shuffled_sharpes, p_value
