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
    apply_transaction_costs,
    performance_metrics,
)

# -----------------------
# CONFIG
# -----------------------
TICKERS = ["SPY", "AAPL", "MSFT", "NVDA"]
START = "2018-01-01"
SHORT_W = 20
LONG_W = 100
COST_PER_TRADE = 0.0005

N_SIMS = 2000
SEED = 7
TRADING_DAYS = 252

# -----------------------
# Build your strategy returns (same as before)
# -----------------------
prices = fetch_prices(TICKERS, start=START)

signal = moving_average_crossover_signals(prices, short_window=SHORT_W, long_window=LONG_W)
positions = positions_from_signals(signal)

strat_ret = backtest_long_only(prices, positions)
strat_ret_cost = apply_transaction_costs(strat_ret, positions, cost_per_trade=COST_PER_TRADE)

port_ret = strat_ret_cost.mean(axis=1).dropna()

# Actual Sharpe (rf=0)
actual_metrics = performance_metrics(port_ret.to_frame("Portfolio")).iloc[0]
actual_sharpe = float(actual_metrics["sharpe_rf0"])

print("\n=== ACTUAL STRATEGY METRICS (Portfolio) ===")
print(actual_metrics.to_frame().T)

# -----------------------
# Bootstrap Monte Carlo
# -----------------------
rng = np.random.default_rng(SEED)

rets = port_ret.values
n = len(rets)

def sharpe_from_daily(daily_returns: np.ndarray) -> float:
    mu = daily_returns.mean()
    vol = daily_returns.std(ddof=0)
    if vol == 0:
        return np.nan
    return (mu * TRADING_DAYS) / (vol * np.sqrt(TRADING_DAYS))

sim_sharpes = np.empty(N_SIMS, dtype=float)

for i in range(N_SIMS):
    sample = rng.choice(rets, size=n, replace=True)
    sim_sharpes[i] = sharpe_from_daily(sample)

# Clean
sim_sharpes = sim_sharpes[~np.isnan(sim_sharpes)]

# Percentile of actual Sharpe among bootstraps
percentile = (sim_sharpes < actual_sharpe).mean() * 100.0

print("\n=== MONTE CARLO ROBUSTNESS (BOOTSTRAP) ===")
print(f"Actual Sharpe: {actual_sharpe:.4f}")
print(f"Bootstrap Sharpe mean: {sim_sharpes.mean():.4f}")
print(f"Bootstrap Sharpe std : {sim_sharpes.std():.4f}")
print(f"Actual Sharpe percentile vs bootstrap: {percentile:.2f}%")

# “p-value style” (how often bootstrap >= actual)
p_like = (sim_sharpes >= actual_sharpe).mean()
print(f"Fraction of bootstrap sims with Sharpe >= actual (p-like): {p_like:.4f}")

# Save distribution for plots/reporting
pd.DataFrame({"bootstrap_sharpe": sim_sharpes}).to_csv("phase4_montecarlo_bootstrap_sharpe.csv", index=False)
print("\nSaved: phase4_montecarlo_bootstrap_sharpe.csv")

from plots import save_hist_png

save_hist_png(
    sim_sharpes,
    "Bootstrap Sharpe Distribution",
    "Sharpe",
    "charts/montecarlo_sharpe_hist.png"
)
print("Saved chart: charts/montecarlo_sharpe_hist.png")
