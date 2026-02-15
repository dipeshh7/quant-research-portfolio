from __future__ import annotations
import numpy as np
import pandas as pd

TRADING_DAYS = 252

def month_end_index(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return pd.to_datetime(pd.Series(idx)).dt.to_period("M").dt.to_timestamp("M").unique()

def compute_momentum_scores(prices: pd.DataFrame, lookback_days: int = 126, skip_days: int = 21) -> pd.DataFrame:
    """
    Momentum score at time t:
      return from (t - skip - lookback) to (t - skip)
    skip_days avoids short-term reversal.
    """
    shifted = prices.shift(skip_days)
    score = shifted / shifted.shift(lookback_days) - 1.0
    return score

def build_cs_mom_weights(
    prices: pd.DataFrame,
    lookback_days: int = 126,
    skip_days: int = 21,
    top_n: int = 2,
    bottom_n: int = 2,
) -> pd.DataFrame:
    """
    Monthly rebalance:
    - rank assets by momentum score
    - long top_n (equal weight)
    - short bottom_n (equal weight)
    weights sum to 0 (market neutral-ish).
    """
    scores = compute_momentum_scores(prices, lookback_days, skip_days)

    w = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    months = prices.index.to_period("M")

    for m in months.unique():
        month_mask = months == m
        month_dates = prices.index[month_mask]
        if len(month_dates) == 0:
            continue
        # rebalance on first trading day of month using scores from that date
        reb_date = month_dates[0]
        s = scores.loc[reb_date].dropna()
        if len(s) < (top_n + bottom_n):
            continue

        ranked = s.sort_values(ascending=False)
        longs = ranked.head(top_n).index
        shorts = ranked.tail(bottom_n).index

        # equal weights
        w.loc[month_dates, longs] = 1.0 / top_n
        w.loc[month_dates, shorts] = -1.0 / bottom_n

    # Apply next day to avoid lookahead from same-day close-to-close
    return w.shift(1).fillna(0.0)

def apply_costs_from_weight_turnover(port_ret: pd.Series, weights: pd.DataFrame, cost_per_1x_turnover: float = 0.0005) -> pd.Series:
    """
    Turnover cost approx:
      cost_t = sum(|w_t - w_{t-1}|) * cost
    """
    turnover = weights.diff().abs().sum(axis=1).fillna(0.0)
    cost = turnover * cost_per_1x_turnover
    return port_ret - cost

def run_cs_momentum(
    prices: pd.DataFrame,
    lookback_days: int = 126,
    skip_days: int = 21,
    top_n: int = 2,
    bottom_n: int = 2,
    cost_per_1x_turnover: float = 0.0005,
) -> pd.DataFrame:
    rets = prices.pct_change().fillna(0.0)
    w = build_cs_mom_weights(prices, lookback_days, skip_days, top_n, bottom_n)
    port = (w * rets).sum(axis=1)
    port = apply_costs_from_weight_turnover(port, w, cost_per_1x_turnover)
    return port.to_frame("Portfolio")
