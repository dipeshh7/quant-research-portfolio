from __future__ import annotations
import numpy as np
import pandas as pd

TRADING_DAYS = 252

def moving_average_crossover_signals(
    prices: pd.DataFrame,
    short_window: int = 20,
    long_window: int = 100,
) -> pd.DataFrame:
    """
    Returns a signal DataFrame with values:
      1 = long
      0 = out of market (cash)
    Signal is based on short MA > long MA.
    """
    if short_window >= long_window:
        raise ValueError("short_window must be < long_window")

    short_ma = prices.rolling(short_window).mean()
    long_ma = prices.rolling(long_window).mean()

    signal = (short_ma > long_ma).astype(int)
    return signal

def positions_from_signals(signal: pd.DataFrame) -> pd.DataFrame:
    """
    Convert signals into positions applied on NEXT day to avoid look-ahead bias.
    """
    return signal.shift(1).fillna(0)

def backtest_long_only(
    prices: pd.DataFrame,
    positions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Basic long-only backtest:
    - Compute daily asset returns
    - Strategy return = position * asset_return
    Returns a DataFrame of strategy returns per asset.
    """
    asset_ret = prices.pct_change().fillna(0)
    strat_ret = positions * asset_ret
    return strat_ret

def equity_curve(returns: pd.DataFrame, start: float = 1.0) -> pd.DataFrame:
    """
    Convert returns to equity curve (cumulative growth).
    """
    return start * (1.0 + returns).cumprod()

def performance_metrics(strategy_returns: pd.DataFrame) -> pd.DataFrame:
    """
    Annualized mean, vol, Sharpe (rf=0) and max drawdown.
    """
    mu_daily = strategy_returns.mean()
    vol_daily = strategy_returns.std()

    mu_ann = mu_daily * TRADING_DAYS
    vol_ann = vol_daily * np.sqrt(TRADING_DAYS)
    sharpe = (mu_ann / vol_ann).replace([np.inf, -np.inf], np.nan)

    eq = equity_curve(strategy_returns).ffill()
    roll_max = eq.cummax()
    drawdown = (eq / roll_max) - 1.0
    max_dd = drawdown.min()

    out = pd.DataFrame({
        "mean_ann": mu_ann,
        "vol_ann": vol_ann,
        "sharpe_rf0": sharpe,
        "max_drawdown": max_dd,
    })
    return out.sort_values("sharpe_rf0", ascending=False)

def apply_transaction_costs(
    strategy_returns: pd.DataFrame,
    positions: pd.DataFrame,
    cost_per_trade: float = 0.0005,  # 5 bps per trade
) -> pd.DataFrame:
    """
    Deduct transaction costs when position changes.
    """
    trades = positions.diff().abs().fillna(0)
    costs = trades * cost_per_trade
    return strategy_returns - costs

