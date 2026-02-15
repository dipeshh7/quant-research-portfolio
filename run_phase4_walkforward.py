import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent / "src"))

from data import fetch_prices
from strategy import (
    moving_average_crossover_signals,
    positions_from_signals,
    backtest_long_only,
    equity_curve,
    performance_metrics,
    apply_transaction_costs,
)

# -----------------------
# CONFIG
# -----------------------
TICKERS = ["SPY", "AAPL", "MSFT", "NVDA"]
START = "2018-01-01"
COST_PER_TRADE = 0.0005

# Parameter grid (same idea as Phase 3)
SHORT_GRID = [10, 15, 20, 30, 40, 50]
LONG_GRID  = [60, 80, 100, 120, 150, 200]

# Walk-forward setup:
# Optimize using expanding history up to (year-1), trade during that year
FIRST_TRADE_YEAR = 2019
LAST_TRADE_YEAR = 2025

MIN_TRADES_OK = 3

# -----------------------
# Helpers
# -----------------------
def port_returns_with_costs(prices: pd.DataFrame, short_w: int, long_w: int):
    signal = moving_average_crossover_signals(prices, short_window=short_w, long_window=long_w)
    positions = positions_from_signals(signal)

    trades = positions.diff().abs().sum().sum()

    strat_ret = backtest_long_only(prices, positions)
    strat_ret_cost = apply_transaction_costs(strat_ret, positions, cost_per_trade=COST_PER_TRADE)

    port_ret = strat_ret_cost.mean(axis=1)  # equal-weight portfolio
    return port_ret, trades

def best_params_on_train(train_prices: pd.DataFrame):
    best = None
    best_row = None

    for s in SHORT_GRID:
        for l in LONG_GRID:
            if s >= l:
                continue

            r, trades = port_returns_with_costs(train_prices, s, l)
            if trades < MIN_TRADES_OK:
                continue

            m = performance_metrics(r.to_frame("Portfolio")).iloc[0]
            sharpe = float(m["sharpe_rf0"])

            row = {
                "short_window": s,
                "long_window": l,
                "sharpe": sharpe,
                "mean_ann": float(m["mean_ann"]),
                "vol_ann": float(m["vol_ann"]),
                "max_drawdown": float(m["max_drawdown"]),
                "trades": float(trades),
            }

            if (best is None) or (sharpe > best):
                best = sharpe
                best_row = row

    return best_row

# -----------------------
# Load data
# -----------------------
prices = fetch_prices(TICKERS, start=START)

# Benchmark (Buy & Hold SPY on same full timeline)
spy_ret_full = prices[["SPY"]].pct_change().fillna(0)
spy_eq_full = equity_curve(spy_ret_full).rename(columns={"SPY": "BuyHold_SPY"})

# -----------------------
# Walk-forward loop
# -----------------------
all_wfo_returns = []
chosen_params = []

for year in range(FIRST_TRADE_YEAR, LAST_TRADE_YEAR + 1):
    train_end = str(year - 1)
    test_year = str(year)

    train = prices.loc[:train_end]
    test = prices.loc[test_year:test_year]

    if len(test) < 50:
        continue  # not enough data

    best = best_params_on_train(train)
    if best is None:
        continue

    s = int(best["short_window"])
    l = int(best["long_window"])

    test_ret, test_trades = port_returns_with_costs(test, s, l)

    chosen_params.append({
        "trade_year": year,
        "train_end_year": year - 1,
        "short_window": s,
        "long_window": l,
        "train_best_sharpe": best["sharpe"],
        "test_trades": float(test_trades),
    })

    all_wfo_returns.append(test_ret.rename(f"WFO_{year}"))

# Combine yearly return segments into one continuous series
wfo_ret = pd.concat(all_wfo_returns).sort_index()
wfo_ret = wfo_ret[~wfo_ret.index.duplicated(keep="first")]
wfo_eq = equity_curve(wfo_ret.to_frame("WFO_Strategy"))

# Align benchmark to WFO timeline for fair comparison
spy_eq = spy_eq_full.reindex(wfo_eq.index).ffill()

comparison = pd.concat([wfo_eq, spy_eq], axis=1)

print("\n=== WALK-FORWARD PERFORMANCE (WFO Strategy) ===")
print(performance_metrics(wfo_ret.to_frame("Portfolio")).rename(index={"Portfolio": "WFO"}))

print("\nFinal Equity (WFO vs SPY) on WFO timeline:")
print(comparison.tail(1))

params_df = pd.DataFrame(chosen_params)
params_df.to_csv("phase4_wfo_chosen_params.csv", index=False)
wfo_ret.to_frame("WFO_Return").to_csv("phase4_wfo_returns.csv")

print("\nSaved:")
print("- phase4_wfo_chosen_params.csv")
print("- phase4_wfo_returns.csv")
