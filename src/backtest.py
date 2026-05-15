from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    benchmark_curve: pd.Series
    total_return: float
    benchmark_return: float
    sharpe_ratio: float
    max_drawdown: float
    num_trades: int
    win_rate: float
    trades: pd.DataFrame


def run_backtest(
    signals: pd.Series,
    prices: pd.Series,
    initial_capital: float = 10_000.0,
    transaction_cost: float = 0.001,
) -> BacktestResult:
    """Simulate trading from Buy/Hold/Sell signals (2/1/0).

    Rules
    -----
    - Buy signal (2) while flat  → buy at close, pay transaction_cost on notional
    - Sell signal (0) while long → sell at close, pay transaction_cost on notional
    - Hold signal (1)            → do nothing
    - Only one position at a time; no shorting
    """
    common = signals.index.intersection(prices.index)
    signals = signals.loc[common].sort_index()
    prices = prices.loc[common].sort_index()

    cash = initial_capital
    shares = 0.0
    in_position = False
    entry_price = 0.0

    equity_values = []
    trades = []

    for date in signals.index:
        sig = int(signals.loc[date])
        price = float(prices.loc[date])

        if sig == 2 and not in_position:
            fee = cash * transaction_cost
            shares = (cash - fee) / price
            cash = 0.0
            entry_price = price
            in_position = True
            trades.append({"date": date, "action": "BUY", "price": price, "pnl_pct": np.nan})

        elif sig == 0 and in_position:
            gross = shares * price
            fee = gross * transaction_cost
            cash = gross - fee
            pnl = (price - entry_price) / entry_price
            trades.append({"date": date, "action": "SELL", "price": price, "pnl_pct": pnl})
            shares = 0.0
            in_position = False

        equity_values.append(cash + shares * price)

    equity_curve = pd.Series(equity_values, index=signals.index)
    benchmark_curve = initial_capital * prices / prices.iloc[0]

    # Sharpe (annualised)
    ret = equity_curve.pct_change().dropna()
    sharpe = float(ret.mean() / ret.std() * np.sqrt(252)) if ret.std() > 0 else 0.0

    # Max drawdown
    roll_max = equity_curve.cummax()
    max_dd = float(((equity_curve - roll_max) / roll_max).min())

    trade_df = pd.DataFrame(trades) if trades else pd.DataFrame(
        columns=["date", "action", "price", "pnl_pct"]
    )
    sells = trade_df[trade_df["action"] == "SELL"] if not trade_df.empty else pd.DataFrame()
    win_rate = float((sells["pnl_pct"] > 0).mean()) if len(sells) > 0 else 0.0
    num_trades = int((trade_df["action"] == "BUY").sum()) if not trade_df.empty else 0

    return BacktestResult(
        equity_curve=equity_curve,
        benchmark_curve=benchmark_curve,
        total_return=float(equity_curve.iloc[-1] / initial_capital - 1),
        benchmark_return=float(prices.iloc[-1] / prices.iloc[0] - 1),
        sharpe_ratio=sharpe,
        max_drawdown=max_dd,
        num_trades=num_trades,
        win_rate=win_rate,
        trades=trade_df,
    )
