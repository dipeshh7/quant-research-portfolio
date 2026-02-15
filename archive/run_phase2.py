import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "src"))

from data import fetch_prices
from strategy import (
    moving_average_crossover_signals,
    positions_from_signals,
    backtest_long_only,
    equity_curve,
    performance_metrics,
)
from plots import plot_signals_on_price, plot_equity_curve

tickers = ["SPY", "AAPL", "MSFT", "NVDA"]
prices = fetch_prices(tickers, start="2018-01-01")

# Benchmark: Buy & Hold SPY
spy_returns = prices[["SPY"]].pct_change().fillna(0)
spy_equity = equity_curve(spy_returns)

signal = moving_average_crossover_signals(prices, short_window=20, long_window=100)
positions = positions_from_signals(signal)

# --- Backtest ---
strategy_returns = backtest_long_only(prices, positions)

# --- Apply transaction costs ---
from strategy import apply_transaction_costs
strategy_returns_cost = apply_transaction_costs(strategy_returns, positions)

# --- Equal-weight portfolio ---
portfolio_returns = strategy_returns_cost.mean(axis=1).to_frame("Portfolio")
portfolio_equity = equity_curve(portfolio_returns)

# --- Metrics ---
metrics = performance_metrics(portfolio_returns)
print("\n=== PORTFOLIO (WITH COSTS) METRICS ===")
print(metrics)


# Visuals (pick one ticker to inspect signals)
plot_signals_on_price(prices, signal, ticker="SPY")
plot_equity_curve(portfolio_equity, title="Portfolio Equity Curve (With Costs)")

import pandas as pd

comparison = pd.concat(
    [
        portfolio_equity.rename(columns={"Portfolio": "Strategy"}),
        spy_equity.rename(columns={"SPY": "BuyHold_SPY"})
    ],
    axis=1
)

plot_equity_curve(comparison, title="Strategy vs Buy & Hold SPY")

print("\nFinal Equity (last row):")
print(comparison.tail(1))

print("\nTotal Return (%):")
total_return = (comparison.tail(1).iloc[0] - 1.0) * 100
print(total_return)

# Roughness proxy: daily volatility of equity curve changes
strategy_daily_vol = comparison["Strategy"].pct_change().std()
spy_daily_vol = comparison["BuyHold_SPY"].pct_change().std()

print("\nSmoothness check (lower daily vol = smoother):")
print("Strategy daily vol:", strategy_daily_vol)
print("SPY daily vol:", spy_daily_vol)
