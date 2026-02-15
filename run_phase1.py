import sys
from pathlib import Path

# Add src/ to python path
sys.path.append(str(Path(__file__).parent / "src"))

from data import fetch_prices
from features import compute_returns, summary_stats, rolling_vol
from plots import plot_prices, plot_returns, plot_rolling_vol

tickers = ["SPY", "AAPL", "MSFT", "NVDA"]
prices = fetch_prices(tickers, start="2018-01-01")

simple_ret, log_ret = compute_returns(prices)

stats = summary_stats(simple_ret)
print("\n=== SUMMARY STATS (simple returns) ===")
print(stats)

rvol20 = rolling_vol(simple_ret, window=20)

plot_prices(prices)
plot_returns(simple_ret)
plot_rolling_vol(rvol20)
