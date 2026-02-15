from __future__ import annotations
import pandas as pd
import matplotlib.pyplot as plt

def plot_prices(prices: pd.DataFrame, title: str = "Adjusted Close Prices") -> None:
    ax = prices.plot(figsize=(10, 5))
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    plt.tight_layout()
    plt.show()

def plot_returns(returns: pd.DataFrame, title: str = "Daily Returns") -> None:
    ax = returns.plot(figsize=(10, 5), alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Return")
    plt.tight_layout()
    plt.show()

def plot_rolling_vol(rvol: pd.DataFrame, title: str = "Rolling Volatility (Annualized)") -> None:
    ax = rvol.plot(figsize=(10, 5), alpha=0.9)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Volatility")
    plt.tight_layout()
    plt.show()

def plot_signals_on_price(prices: pd.DataFrame, signal: pd.DataFrame, ticker: str) -> None:
    import matplotlib.pyplot as plt

    px = prices[ticker].dropna()
    sig = signal[ticker].reindex(px.index).fillna(0)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(px.index, px.values, label="Price")

    buy_points = sig.diff().fillna(0) == 1
    sell_points = sig.diff().fillna(0) == -1

    ax.scatter(px.index[buy_points], px[buy_points], marker="^")
    ax.scatter(px.index[sell_points], px[sell_points], marker="v")

    ax.set_title(f"{ticker}: Price with MA Crossover Signals")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    plt.tight_layout()
    plt.show()

def plot_equity_curve(equity: pd.DataFrame, title: str = "Equity Curve") -> None:
    ax = equity.plot(figsize=(10, 5))
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity")
    import matplotlib.pyplot as plt
    plt.tight_layout()
    plt.show()

def save_equity_png(df: pd.DataFrame, title: str, filepath: str) -> None:
    ax = df.plot(figsize=(10, 5))
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity")
    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close()

def save_hist_png(values, title: str, xlabel: str, filepath: str) -> None:
    plt.figure(figsize=(10, 5))
    plt.hist(values, bins=50)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close()