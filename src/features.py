from __future__ import annotations
import numpy as np
import pandas as pd

TRADING_DAYS = 252

def compute_returns(prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      simple_returns: pct change
      log_returns: log(1 + simple_returns)
    """
    if prices.isna().all().all():
        raise ValueError("Prices are all NaN.")

    simple = prices.pct_change().dropna()
    log = np.log1p(simple)
    return simple, log

def summary_stats(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Basic annualized stats:
      mean return, vol, Sharpe (rf=0), min/max daily return
    """
    mu_daily = returns.mean()
    vol_daily = returns.std()

    mu_ann = mu_daily * TRADING_DAYS
    vol_ann = vol_daily * np.sqrt(TRADING_DAYS)
    sharpe = (mu_ann / vol_ann).replace([np.inf, -np.inf], np.nan)

    out = pd.DataFrame({
        "mean_ann": mu_ann,
        "vol_ann": vol_ann,
        "sharpe_rf0": sharpe,
        "min_daily": returns.min(),
        "max_daily": returns.max(),
    })
    return out.sort_values("sharpe_rf0", ascending=False)

def rolling_vol(returns: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Annualized rolling volatility."""
    return returns.rolling(window).std() * np.sqrt(TRADING_DAYS)
