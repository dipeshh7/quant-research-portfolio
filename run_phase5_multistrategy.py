import sys
from pathlib import Path
import pandas as pd

sys.path.append(str(Path(__file__).parent / "src"))

from data import fetch_prices
from strategy import equity_curve, performance_metrics
from multi_strategy import combine_strategies

TICKERS = ["SPY", "AAPL", "MSFT", "NVDA"]
prices = fetch_prices(TICKERS, start="2018-01-01")

# Baseline: your previous best-ish MA params
port_ret = combine_strategies(
    prices,
    trend_params=(20, 100),
    mr_params=(20, 1.0),
    w_trend=0.75,
    w_mr=0.25,
    cost_per_trade=0.0005,
    target_ann_vol=0.14,
)

metrics = performance_metrics(port_ret)
print("\n=== MULTI-STRATEGY METRICS ===")
print(metrics)

eq = equity_curve(port_ret).rename(columns={"Portfolio": "MultiStrategy"})

# Benchmark SPY
spy_ret = prices[["SPY"]].pct_change().fillna(0)
spy_eq = equity_curve(spy_ret).rename(columns={"SPY": "BuyHold_SPY"})

comparison = pd.concat([eq, spy_eq], axis=1)

print("\nFinal Equity:")
print(comparison.tail(1))

comparison.to_csv("phase5_multistrategy_equity.csv")
print("\nSaved: phase5_multistrategy_equity.csv")

from plots import save_equity_png

save_equity_png(
    comparison,
    "MultiStrategy vs SPY",
    "charts/equity_multistrategy_vs_spy.png"
)
print("Saved chart: charts/equity_multistrategy_vs_spy.png")
