import numpy as np
import pandas as pd
import ta


def add_technical_indicators(df: pd.DataFrame):
    """
    Adds technical indicators to dataframe.

    Raw price-level columns (sma_*, macd) are kept for regime detection, but
    scale-free versions (ratios to price) are added for use as model features.
    """

    # Trend
    df["sma_20"] = df["Close"].rolling(window=20).mean()
    df["sma_50"] = df["Close"].rolling(window=50).mean()
    df["sma_200"] = df["Close"].rolling(window=200).mean()

    df["price_to_sma_20"] = df["Close"] / df["sma_20"] - 1
    df["price_to_sma_50"] = df["Close"] / df["sma_50"] - 1
    df["price_to_sma_200"] = df["Close"] / df["sma_200"] - 1

    # Momentum
    df["rsi"] = ta.momentum.RSIIndicator(df["Close"], window=14).rsi()
    macd = ta.trend.MACD(df["Close"])
    df["macd"] = macd.macd()
    df["macd_norm"] = df["macd"] / df["Close"]

    # Volatility (std of daily returns, comparable across assets and time)
    df["volatility_20"] = df["Close"].pct_change().rolling(window=20).std()

    # Volume (relative to its own 20-day average)
    df["volume_ratio_20"] = (
        df["Volume"] / df["Volume"].rolling(window=20).mean() - 1
    ).replace([np.inf, -np.inf], np.nan)

    return df


def add_relative_strength(df: pd.DataFrame):
    """
    Adds relative strength feature vs S&P 500.
    """

    df["stock_return_20"] = df["Close"].pct_change(20)
    df["market_return_20"] = df["market_close"].pct_change(20)

    df["relative_strength_20"] = (
        df["stock_return_20"] - df["market_return_20"]
    )

    return df


def add_forward_return(df: pd.DataFrame, horizon=21):
    """
    Adds forward return target variable.
    """

    df["forward_return"] = df["Close"].shift(-horizon) / df["Close"] - 1
    df["target"] = (df["forward_return"] > 0).astype(int)

    return df


def add_momentum_features(df: pd.DataFrame):
    """
    Adds time-series momentum features.
    """

    # 3, 6, 12 month returns
    df["ret_3m"] = df["Close"].pct_change(63)
    df["ret_6m"] = df["Close"].pct_change(126)
    df["ret_12m"] = df["Close"].pct_change(252)

    # Rolling z-score of 3m return
    df["ret_3m_z"] = (
        df["ret_3m"] - df["ret_3m"].rolling(126).mean()
    ) / df["ret_3m"].rolling(126).std()

    # Rolling z-score of 6m return
    df["ret_6m_z"] = (
        df["ret_6m"] - df["ret_6m"].rolling(126).mean()
    ) / df["ret_6m"].rolling(126).std()

    return df
