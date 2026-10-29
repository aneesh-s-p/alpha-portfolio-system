import pandas as pd

from config import REGIME_VOL_MIN_PERIODS


def detect_market_regime(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detects market regime based on:
    - Trend (Close vs 200 SMA)
    - Volatility (relative to the *expanding* median, so each date only uses
      information available up to that date - no look-ahead)

    Outputs:
    - trend_regime (int)
    - vol_regime (int)
    - market_regime (string, e.g., "1_-1")
    """

    df = df.copy()

    df["trend_regime"] = 0
    df.loc[df["Close"] > df["sma_200"], "trend_regime"] = 1
    df.loc[df["Close"] <= df["sma_200"], "trend_regime"] = -1

    vol_median = df["volatility_20"].expanding(
        min_periods=REGIME_VOL_MIN_PERIODS
    ).median()

    df["vol_regime"] = 0
    df.loc[df["volatility_20"] <= vol_median, "vol_regime"] = 1
    df.loc[df["volatility_20"] > vol_median, "vol_regime"] = -1

    df["market_regime"] = (
        df["trend_regime"].astype(str) + "_" +
        df["vol_regime"].astype(str)
    )

    return df
