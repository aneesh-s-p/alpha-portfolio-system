import pandas as pd
import numpy as np

from config import REGIME_THRESHOLDS, HORIZON


def backtest_strategy_regime(df, ensemble_prob, horizon=HORIZON):

    df = df.copy()
    df = df.iloc[-len(ensemble_prob):].copy()
    df["prob"] = ensemble_prob

    trades = []
    i = 0

    while i < len(df) - horizon:

        regime = df.iloc[i]["market_regime"]

        # Regime-specific thresholds (defined in config.py)
        threshold = REGIME_THRESHOLDS.get(regime, REGIME_THRESHOLDS["-1_-1"])

        prob = df.iloc[i]["prob"]

        if prob > threshold:
            entry_price = df.iloc[i]["Close"]
            exit_price = df.iloc[i + horizon]["Close"]

            trade_return = (exit_price / entry_price) - 1
            trades.append(trade_return)

        i += horizon

    if len(trades) == 0:
        return 0, 0

    strategy_return = np.prod([1 + r for r in trades]) - 1

    market_start = df.iloc[0]["Close"]
    market_end = df.iloc[-1]["Close"]
    market_return = (market_end / market_start) - 1

    return strategy_return, market_return
