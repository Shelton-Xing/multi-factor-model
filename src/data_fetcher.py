"""Data fetcher — factor computation module."""
import akshare as ak
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

_CACHE = Path(__file__).parent.parent / 'cache'
_CACHE.mkdir(exist_ok=True)


def fetch_prices(codes, start_date=None, end_date=None):
    """Fetch daily price data for a list of stock codes."""
    if end_date is None:
        end_date = datetime.now().strftime('%Y%m%d')
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=270)).strftime('%Y%m%d')
    
    price_data = {}
    for i, code in enumerate(codes):
        try:
            df = ak.stock_zh_a_hist(symbol=code, period="daily",
                                     start_date=start_date, end_date=end_date,
                                     adjust="qfq")
            if not df.empty and len(df) > 130:
                price_data[code] = df
        except:
            pass
        if (i + 1) % 50 == 0:
            print(f"    Prices: {i+1}/{len(codes)} ({len(price_data)} valid)")
    
    return price_data


def fetch_fund_flow(codes):
    """Fetch individual stock fund flow data."""
    fund_flow = {}
    for i, code in enumerate(codes):
        market = "sh" if code.startswith('6') else "sz"
        try:
            df = ak.stock_individual_fund_flow(stock=code, market=market)
            if not df.empty:
                fund_flow[code] = df
        except:
            pass
        if (i + 1) % 10 == 0:
            print(f"    Fund flow: {i+1}/{len(codes)} ({len(fund_flow)} OK)")
    
    return fund_flow


def compute_sdl(price_data, fund_flow):
    """Compute SDL factor: Normalized(MainForceFlow / Close)."""
    rows = []
    for code, df in fund_flow.items():
        if code not in price_data:
            continue
        for _, r in df.iterrows():
            date = str(r['日期'])[:10]
            try:
                main_flow_col = [c for c in r.index if '主力净流入' in c and '净额' in c][0]
                mf = float(r[main_flow_col])
                close = float(r['收盘价'])
                if close > 0:
                    rows.append({'date': date, 'code': code, 'sdl_raw': mf / close})
            except:
                pass
    
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df['sdl'] = df.groupby('date')['sdl_raw'].transform(
        lambda x: (x - x.mean()) / x.std()
    )
    return df[['date', 'code', 'sdl']]


def compute_momentum(price_data, horizon=20):
    """Compute momentum factor: N-day return z-score."""
    rows = []
    for code, df in price_data.items():
        closes = df['收盘'].values.astype(float)
        dates = [str(d)[:10] for d in pd.to_datetime(df['日期']).values]
        for t in range(horizon, len(closes)):
            mom = closes[t] / closes[t - horizon] - 1
            rows.append({'date': dates[t], 'code': code, 'mom': mom})
    
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df['mom_z'] = df.groupby('date')['mom'].transform(
        lambda x: (x - x.mean()) / x.std()
    )
    return df[['date', 'code', 'mom_z']]


def compute_reversal(price_data, horizon=5):
    """Compute short-term reversal factor: -1× N-day return."""
    rows = []
    for code, df in price_data.items():
        closes = df['收盘'].values.astype(float)
        dates = [str(d)[:10] for d in pd.to_datetime(df['日期']).values]
        for t in range(horizon, len(closes)):
            rev = -(closes[t] / closes[t - horizon] - 1)
            rows.append({'date': dates[t], 'code': code, 'rev': rev})
    
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df['rev_z'] = df.groupby('date')['rev'].transform(
        lambda x: (x - x.mean()) / x.std()
    )
    return df[['date', 'code', 'rev_z']]


def fetch_all_data():
    """Fetch all data: universe → prices → fund flow."""
    import akshare as ak
    
    print("[1] Stock Universe...")
    df_idx = ak.index_stock_cons(symbol="000300")
    codes = df_idx['品种代码'].astype(str).str.zfill(6).tolist()[:200]
    print(f"  CSI 300 → {len(codes)} stocks")
    
    print("\n[2] Price Data...")
    price_data = fetch_prices(codes)
    print(f"  {len(price_data)} stocks with valid price data")
    
    print("\n[3] Fund Flow Data...")
    fund_flow = fetch_fund_flow(list(price_data.keys()))
    print(f"  {len(fund_flow)} stocks with fund flow data")
    
    return price_data, fund_flow


def compute_factors(price_data, fund_flow):
    """Compute all factors."""
    print("\n[4] Computing Factors...")
    
    sdl = compute_sdl(price_data, fund_flow)
    print(f"  SDL: {len(sdl):,} rows")
    
    mom = compute_momentum(price_data, 20)
    print(f"  Momentum: {len(mom):,} rows")
    
    rev = compute_reversal(price_data, 5)
    print(f"  Reversal: {len(rev):,} rows")
    
    return sdl, mom, rev
