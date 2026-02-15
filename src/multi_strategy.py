from __future__ import annotations
import numpy as np
import pandas as pd

TRADING_DAYS = 252

def zscore(x: pd.Series, window: int) -> pd.Series:
    m = x.rolling(window).mean()
    s = x.rolling(window).std()
    return (x - m) / s

def trend_signal_ma(prices: pd.DataFrame, short_w: int = 20, long_w: int = 100) -> pd.DataFrame:
    short_ma = prices.rolling(short_w).mean()
    long_ma  = prices.rolling(long_w).mean()
    # 1 long, 0 cash
    sig = (short_ma > long_ma).astype(int)
    return sig

def mean_reversion_signal(prices: pd.DataFrame, window: int = 20, entry_z: float = 1.0) -> pd.DataFrame:
    """
    Simple mean reversion:
    - compute zscore of price vs rolling mean
    - if z < -entry_z => long (expect rebound)
    - if z > +entry_z => cash (or could short; we keep long/cash to stay simple)
    """
    sig = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)
    for c in prices.columns:
        z = zscore(prices[c], window)
        sig[c] = (z < -entry_z).astype(int)
    return sig

def positions_from_signal(sig: pd.DataFrame) -> pd.DataFrame:
    # apply next day to avoid lookahead
    return sig.shift(1).fillna(0)

def apply_transaction_costs(returns: pd.DataFrame, positions: pd.DataFrame, cost_per_trade: float = 0.0005) -> pd.DataFrame:
    trades = positions.diff().abs().fillna(0)
    costs = trades * cost_per_trade
    return returns - costs

def asset_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().fillna(0)

def vol_target_weights(returns: pd.DataFrame, target_ann_vol: float = 0.12, window: int = 20) -> pd.DataFrame:
    """
    Per-asset volatility targeting (simple):
    weight_t = target_daily_vol / rolling_std
    Clipped to [0, 2] to prevent crazy leverage.
    """
    target_daily = target_ann_vol / np.sqrt(TRADING_DAYS)
    vol = returns.rolling(window).std()
    w = target_daily / vol
    w = w.clip(lower=0.0, upper=2.0).fillna(0.0)
    return w

def combine_strategies(
    prices: pd.DataFrame,
    trend_params=(20, 100),
    mr_params=(20, 1.0),
    w_trend: float = 0.6,
    w_mr: float = 0.4,
    cost_per_trade: float = 0.0005,
    target_ann_vol: float = 0.12,
) -> pd.DataFrame:
    """
    Returns daily portfolio returns series as DataFrame with column 'Portfolio'.
    Pipeline:
    - build trend & mean reversion signals (long/cash)
    - convert to positions (next day)
    - compute returns * positions
    - apply transaction costs
    - volatility target each leg
    - combine legs by weights
    """
    rets = asset_returns(prices)

    s_short, s_long = trend_params
    mr_window, mr_entry = mr_params

    trend_sig = trend_signal_ma(prices, s_short, s_long)
    mr_sig = mean_reversion_signal(prices, mr_window, mr_entry)

    trend_pos = positions_from_signal(trend_sig)
    mr_pos = positions_from_signal(mr_sig)

    trend_leg = trend_pos * rets
    mr_leg = mr_pos * rets

    trend_leg = apply_transaction_costs(trend_leg, trend_pos, cost_per_trade=cost_per_trade)
    mr_leg = apply_transaction_costs(mr_leg, mr_pos, cost_per_trade=cost_per_trade)

    # vol targeting (per asset), then equal-weight across assets
    trend_w = vol_target_weights(trend_leg, target_ann_vol=target_ann_vol)
    mr_w = vol_target_weights(mr_leg, target_ann_vol=target_ann_vol)

    trend_port = (trend_leg * trend_w).mean(axis=1)
    mr_port = (mr_leg * mr_w).mean(axis=1)

    combo = (w_trend * trend_port) + (w_mr * mr_port)
    return combo.to_frame("Portfolio")
