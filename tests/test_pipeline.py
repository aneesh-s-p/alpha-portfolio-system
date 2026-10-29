"""
Offline tests: no network access needed (price data is synthetic).
They check the methodology fixes (no look-ahead, no overlap, leverage cap,
turnover-based costs, scale-free features) and that the pipeline runs end to end.
"""
import numpy as np
import pandas as pd
import pytest

import config
from config import MAX_LEVERAGE, TRANSACTION_COST
from engine.portfolio_engine import (
    apply_vol_target, simulate_long_top_n, permutation_test, compute_sharpe,
)
from src.regime_detection import detect_market_regime
from src.feature_engineering import (
    add_technical_indicators, add_relative_strength,
    add_momentum_features, add_forward_return,
)
from src.model import train_models, NON_FEATURE_COLUMNS


def make_prices(n=1500, seed=0, market=None):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=n)
    close = 50 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, n)))
    if market is None:
        market = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n)))
    return pd.DataFrame({
        "Open": close, "High": close * 1.01, "Low": close * 0.99,
        "Close": close, "Volume": rng.integers(1_000_000, 2_000_000, n),
        "market_close": market,
    }, index=idx)


def test_horizon_does_not_overlap_rebalance():
    assert config.HORIZON <= config.REBALANCE_STEP


def test_vol_target_is_causal():
    r = np.random.default_rng(1).normal(0.01, 0.05, 40)
    a = apply_vol_target(r)
    r2 = r.copy()
    r2[-1] += 0.5                      # change only the last period
    b = apply_vol_target(r2)
    assert np.allclose(a.iloc[:-1], b.iloc[:-1])


def test_vol_target_never_levers_up():
    r = np.random.default_rng(2).normal(0.01, 0.001, 40)   # very low vol
    s = apply_vol_target(r)
    assert (np.abs(s.values) <= np.abs(r) * MAX_LEVERAGE + 1e-12).all()


def test_regime_volatility_is_causal():
    rng = np.random.default_rng(3)
    n = 400
    df = pd.DataFrame({
        "Close": 100 + rng.normal(size=n).cumsum(),
        "sma_200": 100.0,
        "volatility_20": rng.random(n),
    })
    base = detect_market_regime(df)["vol_regime"]
    df2 = df.copy()
    df2.loc[df2.index[-1], "volatility_20"] = 1e6   # change only the future
    alt = detect_market_regime(df2)["vol_regime"]
    assert (base.iloc[:-1] == alt.iloc[:-1]).all()


def test_model_excludes_raw_price_levels():
    df = make_prices(900)
    df = add_technical_indicators(df)
    df = add_relative_strength(df)
    df = add_momentum_features(df)
    df = add_forward_return(df, horizon=config.HORIZON)
    df = detect_market_regime(df).dropna()
    *_, cols = train_models(df)
    assert not set(NON_FEATURE_COLUMNS) & set(cols)
    assert "price_to_sma_200" in cols and "macd_norm" in cols


def test_transaction_cost_is_turnover_based():
    # Same two assets chosen every month, zero returns: only the first
    # rebalance (buying from cash) should cost anything.
    dates = pd.bdate_range("2020-01-31", periods=12, freq="21B")
    rows = []
    for d in dates:
        for t, p in [("A", 0.9), ("B", 0.8), ("C", 0.1)]:
            rows.append({"date": d, "ticker": t, "probability": p, "forward_return": 0.0})
    _, rets = simulate_long_top_n(pd.DataFrame(rows), top_n=2, verbose=False)
    weight = min(1 / 2, config.MAX_WEIGHT)
    assert rets.iloc[0] == pytest.approx(-TRANSACTION_COST * 2 * weight)
    assert np.allclose(rets.iloc[1:], 0.0)


def test_permutation_pvalue_is_valid():
    rng = np.random.default_rng(4)
    dates = pd.bdate_range("2018-01-31", periods=40, freq="21B")
    rows = [{"date": d, "ticker": f"T{k}", "probability": rng.random(),
             "forward_return": rng.normal(0.01, 0.05)}
            for d in dates for k in range(6)]
    real, shuffled, p = permutation_test(pd.DataFrame(rows), top_n=2, n_tests=30)
    assert 1 / 31 <= p <= 1.0
    assert len(shuffled) == 30


def test_full_pipeline_runs_on_synthetic_data(monkeypatch):
    import engine.alpha_engine as ae
    from engine.validation_engine import evaluate_alpha

    market = make_prices(1500, seed=99)["Close"].values
    seeds = {t: i for i, t in enumerate(["AAA", "BBB", "CCC", "DDD"])}
    monkeypatch.setattr(
        ae, "download_stock_data",
        lambda t, *a, **k: make_prices(1500, seed=seeds[t], market=market),
    )

    alpha_df = ae.generate_alpha_dataset(list(seeds))
    assert len(alpha_df) > 0
    assert alpha_df["probability"].between(0, 1).all()

    evaluate_alpha(alpha_df)
    capital, rets = simulate_long_top_n(alpha_df, top_n=2, verbose=False)
    assert capital > 0 and len(rets) > 0
    assert np.isfinite(compute_sharpe(rets))
