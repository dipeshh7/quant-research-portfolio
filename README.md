Quant Research Portfolio — Systematic Trading Framework (Python)
Overview

This project implements a research-grade systematic trading framework in Python for evaluating multi-asset quantitative strategies with realistic assumptions such as transaction costs, walk-forward optimization, and robustness testing. The objective is to focus on risk-adjusted performance, robustness, and proper research methodology rather than curve-fitted returns.

Strategies Implemented:

1. Time-Series Trend (Moving Average Crossover)
Long/cash strategy using short and long moving averages with next-day execution to avoid look-ahead bias.

2. Mean Reversion (Z-Score Based)
Long when price deviates significantly below rolling mean; improves stability in sideways markets.

3. Cross-Sectional Momentum
Monthly ranking of assets by momentum; long winners and short losers across diversified ETF universe.

4. Multi-Strategy Portfolio
Combined Trend + Mean Reversion + Cross-Sectional Momentum with volatility targeting and transaction cost modeling.

Evaluation Methodology:
Transaction cost modeling
Train/Test split (2018–2022 train, 2023–present test)
Walk-Forward Optimization (yearly re-optimization)
Monte Carlo robustness testing (Sharpe stability)
Risk metrics: Return, Volatility, Sharpe Ratio, Max Drawdown

Results Summary:
Baseline trend strategy Sharpe ≈ 1.0
Walk-forward Sharpe ≈ 1.04, confirming stable out-of-sample behavior
Multi-strategy portfolio:
    Sharpe ≈ 0.62
    Max Drawdown ≈ −16%
    Stable low-volatility equity curve
Demonstrates robust systematic research emphasizing risk control, validation, and avoiding overfitting

📊 Visual Results:
Full Portfolio vs Market
Multi-Strategy Portfolio
Cross-Sectional Momentum
Monte Carlo Robustness (Sharpe Distribution)
Parameter Grid Search (Sharpe Heatmap)

📈 Performance Files:
outputs/equity_full_portfolio.csv
outputs/equity_multistrategy.csv
outputs/equity_cs_momentum.csv
outputs/wfo_returns.csv
outputs/wfo_chosen_params.csv
outputs/montecarlo_bootstrap_sharpe.csv
outputs/grid_search_results.csv

Key Learnings:
Importance of transaction costs and realistic execution
Avoiding overfitting via walk-forward validation
Diversification across signals improves stability
Cross-sectional strategies require sufficiently large asset universe
Risk-adjusted metrics more meaningful than raw returns

How to Run:
python run_phase4_walkforward.py
python run_phase4_montecarlo.py
python run_phase5_multistrategy.py
python run_phase5_cs_momentum.py
python run_phase6_full_portfolio.py

Future Improvements:
Expand universe to 100+ assets
Dynamic strategy weighting
Regime detection (trend vs sideways markets)
Risk parity allocation
Machine learning alpha signals