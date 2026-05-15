import numpy as np
import pandas as pd
import yfinance as yf

FUNDAMENTAL_COLS = [
    "PE_ratio",
    "PB_ratio",
    "Revenue_growth",
    "Profit_margin",
    "Debt_to_equity",
    "ROE",
]


def _try_col(df: pd.DataFrame, *names) -> "pd.Series | None":
    for n in names:
        if n in df.columns:
            return df[n]
    return None


def _to_daily(quarterly: pd.Series, daily_index: pd.DatetimeIndex) -> pd.Series:
    """Forward-fill a quarterly series onto a daily index."""
    if quarterly is None or quarterly.empty:
        return pd.Series(np.nan, index=daily_index)
    q = quarterly.copy()
    q.index = pd.to_datetime(q.index).tz_localize(None)
    daily_naive = (
        daily_index.tz_localize(None) if daily_index.tzinfo is not None else daily_index
    )
    combined = q.reindex(q.index.union(daily_naive)).sort_index().ffill()
    result = combined.reindex(daily_naive)
    result.index = daily_index
    return result


def fetch_fundamentals(ticker: str, price_df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame of fundamental features aligned to price_df.index.
    All NaN values are filled with 0 so the model always receives a number."""
    idx = price_df.index
    out = pd.DataFrame(0.0, index=idx, columns=FUNDAMENTAL_COLS)

    try:
        t = yf.Ticker(ticker)
        close = price_df["Close"]

        try:
            qfin = t.quarterly_financials.T.sort_index()
        except Exception:
            qfin = pd.DataFrame()

        try:
            qbal = t.quarterly_balance_sheet.T.sort_index()
        except Exception:
            qbal = pd.DataFrame()

        # Revenue growth (quarter-over-quarter)
        rev = _try_col(qfin, "Total Revenue", "Revenue")
        if rev is not None and len(rev) > 1:
            out["Revenue_growth"] = _to_daily(rev.pct_change(), idx)

        # Profit margin (net income / revenue)
        net = _try_col(qfin, "Net Income", "Net Income Common Stockholders")
        if net is not None and rev is not None:
            margin = net / rev.replace(0, np.nan)
            out["Profit_margin"] = _to_daily(margin, idx)

        # P/E ratio (daily close / annualised quarterly EPS)
        eps = _try_col(qfin, "Basic EPS", "Diluted EPS")
        if eps is not None:
            eps_annual = _to_daily(eps * 4, idx)
            pe = close / eps_annual.replace(0, np.nan)
            out["PE_ratio"] = pe.clip(-500, 500)

        # Debt / Equity
        debt = _try_col(qbal, "Total Debt", "Long Term Debt")
        equity = _try_col(
            qbal,
            "Stockholders Equity",
            "Common Stock Equity",
            "Total Equity Gross Minority Interest",
        )
        if debt is not None and equity is not None:
            de = debt / equity.replace(0, np.nan)
            out["Debt_to_equity"] = _to_daily(de, idx).clip(-50, 50)

        # ROE (annualised quarterly net income / equity)
        if net is not None and equity is not None:
            net_q = net.reindex(equity.index, method="nearest")
            roe = (net_q * 4) / equity.replace(0, np.nan)
            out["ROE"] = _to_daily(roe, idx).clip(-20, 20)

        # P/B ratio (daily close / book value per share)
        shares = _try_col(
            qbal, "Share Issued", "Ordinary Shares Number", "Common Stock"
        )
        if equity is not None and shares is not None:
            bvps = equity / shares.replace(0, np.nan)
            pb = close / _to_daily(bvps, idx).replace(0, np.nan)
            out["PB_ratio"] = pb.clip(0, 100)

    except Exception:
        pass  # return all-zero frame on any failure

    return out.fillna(0.0)
