from __future__ import annotations
import pandas as pd
import yfinance as yf

def fetch_prices(
    tickers: list[str],
    start: str = "2018-01-01",
    end: str | None = None,
) -> pd.DataFrame:
    """
    Fetch Adjusted Close prices for given tickers.
    Returns a DataFrame indexed by date with one column per ticker.
    """
    if not tickers:
        raise ValueError("tickers must be a non-empty list")

    df = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
        group_by="column",
    )

    # yfinance returns MultiIndex columns when multiple tickers are requested
    if isinstance(df.columns, pd.MultiIndex):
        if ("Adj Close" not in df.columns.get_level_values(0)):
            raise ValueError("Could not find 'Adj Close' in downloaded data.")
        prices = df["Adj Close"].copy()
    else:
        # single ticker case
        if "Adj Close" not in df.columns:
            raise ValueError("Could not find 'Adj Close' in downloaded data.")
        prices = df[["Adj Close"]].copy()
        prices.columns = [tickers[0]]

    prices = prices.dropna(how="all")
    return prices
