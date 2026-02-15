import sys
from pathlib import Path
import pandas as pd

sys.path.append(str(Path(__file__).parent / "src"))

from data import fetch_prices
from strategy import equity_curve, performance_metrics
from multi_strategy import combine_strategies
from cross_sectional_mom import run_cs_momentum

# -----------------------
# Universe (expanded)
# -----------------------
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

# -----------------------
# Strategy 1 + 2 (Trend + Mean Reversion)
# -----------------------
ts_ret = combine_strategies(
    prices,
    trend_params=(20, 100),
    mr_params=(20, 1.0),
    w_trend=0.7,
    w_mr=0.3,
    cost_per_trade=0.0005,
    target_ann_vol=0.14,
)["Portfolio"]

# -----------------------
# Strategy 3 (Cross-Sectional Momentum)
# -----------------------
cs_ret = run_cs_momentum(
    prices,
    lookback_days=126,
    skip_days=21,
    top_n=5,
    bottom_n=5,
    cost_per_1x_turnover=0.0005,
)["Portfolio"]

# -----------------------
# Combine (equal-weight across strategies)
# -----------------------
combo_ret = (ts_ret + cs_ret) / 2.0
combo_ret = combo_ret.to_frame("Portfolio")

metrics = performance_metrics(combo_ret)
print("\n=== FINAL MULTI-STRATEGY PORTFOLIO ===")
print(metrics)

eq = equity_curve(combo_ret).rename(columns={"Portfolio": "FullPortfolio"})

# Benchmark SPY
spy_ret = prices[["SPY"]].pct_change().fillna(0)
spy_eq = equity_curve(spy_ret).rename(columns={"SPY": "BuyHold_SPY"})

comparison = pd.concat([eq, spy_eq], axis=1)

print("\nFinal Equity:")
print(comparison.tail(1))

comparison.to_csv("phase6_full_portfolio_equity.csv")
print("\nSaved: phase6_full_portfolio_equity.csv")

from plots import save_equity_png

save_equity_png(
    comparison,
    "Full Portfolio vs SPY",
    "charts/equity_full_portfolio_vs_spy.png"
)
print("Saved chart: charts/equity_full_portfolio_vs_spy.png")
