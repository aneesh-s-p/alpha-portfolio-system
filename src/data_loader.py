from functools import lru_cache

import pandas as pd
import yfinance as yf

from config import START_DATE, END_DATE


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """yfinance may return MultiIndex columns; keep only the field level."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


@lru_cache(maxsize=4)
def _download_market(start: str, end: str) -> pd.DataFrame:
    """S&P 500 data. Cached so it is downloaded once, not once per ticker."""
    market = yf.download(
        "^GSPC", start=start, end=end, auto_adjust=True, progress=False
    )
    return _flatten_columns(market)


def download_stock_data(ticker: str, start=START_DATE, end=END_DATE):
    """
    Downloads (split/dividend-adjusted) stock data and aligns the S&P 500
    close as `market_close`. Returns an empty DataFrame if nothing was found.
    """

    stock = yf.download(
        ticker, start=start, end=end, auto_adjust=True, progress=False
    )
    stock = _flatten_columns(stock)

    if stock.empty:
        return stock

    market = _download_market(start, end)

    stock["market_close"] = market["Close"]
    stock.dropna(inplace=True)

    return stock
