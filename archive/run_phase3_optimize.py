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
# Config
# -----------------------
TICKERS = ["SPY", "AAPL", "MSFT", "NVDA"]
START = "2018-01-01"

SHORT_GRID = [10, 15, 20, 30, 40, 50]
LONG_GRID  = [60, 80, 100, 120, 150, 200]

COST_PER_TRADE = 0.0005  # 5 bps
MIN_TRADES_OK = 3        # ignore “almost never trades” configs

# -----------------------
# Helpers
# -----------------------
def compute_portfolio_returns_with_costs(prices: pd.DataFrame, short_w: int, long_w: int) -> pd.Series:
    signal = moving_average_crossover_signals(prices, short_window=short_w, long_window=long_w)
    positions = positions_from_signals(signal)

    # sanity: count trades (how often positions change)
    trades = positions.diff().abs().sum().sum()

    strat_ret = backtest_long_only(prices, positions)
    strat_ret_cost = apply_transaction_costs(strat_ret, positions, cost_per_trade=COST_PER_TRADE)

    # equal-weight portfolio return (average across assets)
    port_ret = strat_ret_cost.mean(axis=1)

    return port_ret, trades

# -----------------------
# Load data
# -----------------------
prices = fetch_prices(TICKERS, start=START)

results = []

for s in SHORT_GRID:
    for l in LONG_GRID:
        if s >= l:
            continue

        port_ret, trades = compute_portfolio_returns_with_costs(prices, s, l)

        # ignore configs that basically don’t trade (can look artificially smooth)
        if trades < MIN_TRADES_OK:
            continue

        # metrics expects DataFrame
        m = performance_metrics(port_ret.to_frame("Portfolio")).iloc[0]

        results.append({
            "short_window": s,
            "long_window": l,
            "sharpe": float(m["sharpe_rf0"]),
            "mean_ann": float(m["mean_ann"]),
            "vol_ann": float(m["vol_ann"]),
            "max_drawdown": float(m["max_drawdown"]),
            "trades": float(trades),
        })

res = pd.DataFrame(results).sort_values(["sharpe", "mean_ann"], ascending=False)

print("\nTop 10 parameter sets (Portfolio, With Costs):")
print(res.head(10).to_string(index=False))

best = res.iloc[0]
print("\nBEST PARAMS:")
print(best.to_string())

# Build equity curve for best params
best_ret, best_trades = compute_portfolio_returns_with_costs(prices, int(best["short_window"]), int(best["long_window"]))
best_eq = equity_curve(best_ret.to_frame("Portfolio"))

# Benchmark (Buy & Hold SPY)
spy_ret = prices[["SPY"]].pct_change().fillna(0)
spy_eq = equity_curve(spy_ret).rename(columns={"SPY": "BuyHold_SPY"})

comparison = pd.concat(
    [best_eq.rename(columns={"Portfolio": "BestStrategy"}), spy_eq],
    axis=1
)

print("\nFinal Equity (BestStrategy vs SPY):")
print(comparison.tail(1))

# Save results to CSV for your portfolio repo
res.to_csv("phase3_grid_results.csv", index=False)
print("\nSaved grid search results to: phase3_grid_results.csv")
