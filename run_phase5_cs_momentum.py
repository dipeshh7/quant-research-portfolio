import sys
from pathlib import Path
import pandas as pd

sys.path.append(str(Path(__file__).parent / "src"))

from data import fetch_prices
from strategy import equity_curve, performance_metrics
from cross_sectional_mom import run_cs_momentum

TICKERS = [
    "SPY","QQQ","IWM","DIA",
    "XLK","XLF","XLE","XLY","XLP","XLV","XLI","XLU","XLB","XLRE",
    "TLT","IEF","SHY",
    "GLD","SLV",
    "USO","UNG",
    "VNQ",
    "EEM","EFA",
    "ARKK"
]
prices = fetch_prices(TICKERS, start="2018-01-01")

# Example: 6M lookback, 1M skip, long 2 short 2
port_ret = run_cs_momentum(
    prices,
    lookback_days=126,
    skip_days=21,
    top_n=5,
    bottom_n=5,
    cost_per_1x_turnover=0.0005,
)

metrics = performance_metrics(port_ret)
print("\n=== CROSS-SECTIONAL MOMENTUM METRICS ===")
print(metrics)

eq = equity_curve(port_ret).rename(columns={"Portfolio": "CS_Momentum"})

# Benchmark SPY
spy_ret = prices[["SPY"]].pct_change().fillna(0)
spy_eq = equity_curve(spy_ret).rename(columns={"SPY": "BuyHold_SPY"})

comparison = pd.concat([eq, spy_eq], axis=1)

print("\nFinal Equity:")
print(comparison.tail(1))

comparison.to_csv("phase5_cs_momentum_equity.csv")
print("\nSaved: phase5_cs_momentum_equity.csv")

from plots import save_equity_png

save_equity_png(
    comparison,
    "Cross-Sectional Momentum vs SPY",
    "charts/equity_cs_momentum_vs_spy.png"
)
print("Saved chart: charts/equity_cs_momentum_vs_spy.png")
