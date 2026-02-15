import sys
from pathlib import Path
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
# CONFIG (Use your best params from grid search)
# -----------------------
TICKERS = ["SPY", "AAPL", "MSFT", "NVDA"]
SHORT_W = 20     # <-- replace with your best short
LONG_W  = 100    # <-- replace with your best long
COST_PER_TRADE = 0.0005

# -----------------------
# Load full data
# -----------------------
prices = fetch_prices(TICKERS, start="2018-01-01")

# -----------------------
# Split Train/Test
# -----------------------
train = prices.loc["2018":"2022"]
test  = prices.loc["2023":]

def run_strategy(prices):
    signal = moving_average_crossover_signals(prices, SHORT_W, LONG_W)
    positions = positions_from_signals(signal)

    strat_ret = backtest_long_only(prices, positions)
    strat_ret_cost = apply_transaction_costs(strat_ret, positions, COST_PER_TRADE)

    port_ret = strat_ret_cost.mean(axis=1).to_frame("Portfolio")
    eq = equity_curve(port_ret)

    metrics = performance_metrics(port_ret)

    return eq, metrics

# -----------------------
# Train Performance
# -----------------------
train_eq, train_metrics = run_strategy(train)

print("\n=== TRAIN (2018–2022) ===")
print(train_metrics)

# -----------------------
# Test Performance (UNSEEN DATA)
# -----------------------
test_eq, test_metrics = run_strategy(test)

print("\n=== TEST (2023–Present) ===")
print(test_metrics)

# -----------------------
# Benchmark comparison (TEST period)
# -----------------------
spy_test = test[["SPY"]].pct_change().fillna(0)
spy_eq_test = equity_curve(spy_test).rename(columns={"SPY": "BuyHold_SPY"})

comparison = pd.concat(
    [test_eq.rename(columns={"Portfolio": "Strategy"}), spy_eq_test],
    axis=1
)

print("\nFinal Equity on TEST:")
print(comparison.tail(1))
