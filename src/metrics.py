import numpy as np
import pandas as pd
from .config import FEE, THRESHOLD

#Builds equity curve and finds worst drawdown
def max_drawdown(r: pd.Series) -> float:
    eq = (1 + r).cumprod()
    return float((eq / eq.cummax() - 1).min())

#Converts probabilities into positions
# position x: days return minus fee if position changes
#signal on day t is used to trade on day t+1 --> use next-day returns for backtesting
def backtest(proba: pd.Series, next_ret: pd.Series, thr: float = THRESHOLD) -> pd.Series:
    pos = (proba > thr).astype(int)
    gross = pos * next_ret.reindex(proba.index)
    trades = pos.diff().abs().fillna(pos.abs())
    return (gross - trades * FEE).dropna()

#Computes total return, max drawdown, Sharpe ratio, Sortino ratio and months beating Buy&Hold
#computed on a daily return series r, and optionally compared to Buy&Hold returns bh
def perf(r: pd.Series, bh: pd.Series | None = None) -> dict:
    # output dictionary with total return, max drawdown, Sharpe ratio, Sortino ratio, and months beating Buy&Hold if bh is provided
    out = {"total_return": (1 + r).prod() - 1, "max_drawdown": max_drawdown(r)}    # If Buy&Hold returns are provided, compute the fraction of months where the strategy beats Buy&Hold
    if bh is not None:
        #Resample both the strategy returns and Buy&Hold returns to monthly frequency, compute the cumulative return for each month, and calculate the fraction of months where the strategy outperforms Buy&Hold
        m = pd.DataFrame({"s": r, "b": bh.reindex(r.index)}).resample("MS") \
              .apply(lambda x: (1 + x).prod() - 1)
        #Add the fraction of months where the strategy beats Buy&Hold to the output dict
        out["months_beating_BH"] = float((m["s"] > m["b"]).mean())
    return out