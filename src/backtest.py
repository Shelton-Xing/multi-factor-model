"""Backtest engine — walk-forward LGBM + portfolio simulation."""
import numpy as np
import pandas as pd
from pathlib import Path

_ROOT = Path(__file__).parent.parent


def build_price_lookup(price_data):
    """Build fast price lookup: code -> {date: close}."""
    lookup = {}
    for code, df in price_data.items():
        closes = df['收盘'].values.astype(float)
        dates = [str(d)[:10] for d in pd.to_datetime(df['日期']).values]
        lookup[code] = dict(zip(dates, closes))
    return lookup


def run_backtest(signal_df, price_lookup, holding_days=3, top_pct=0.2, cost=0.001):
    """
    Market-neutral backtest: long top 20%, short bottom 20%.
    Rebalances every `holding_days` trading days.
    """
    dates = sorted(signal_df['date'].unique())
    periods = []
    prev_tickers = set()
    
    for i, date in enumerate(dates):
        if i % holding_days != 0:
            continue
        
        day = signal_df[signal_df['date'] == date]
        if len(day) < 20:
            continue
        
        n = max(10, int(len(day) * top_pct))
        top_tickers = day.nlargest(n, 'score')['code'].tolist()
        bot_tickers = day.nsmallest(n, 'score')['code'].tolist()
        
        weights = {}
        for c in top_tickers:
            weights[c] = 1.0 / n
        for c in bot_tickers:
            weights[c] = -1.0 / n
        
        new_tickers = set(weights.keys())
        turnover = len(prev_tickers - new_tickers) + len(new_tickers - prev_tickers)
        txn_cost = cost * turnover / max(1, len(prev_tickers | new_tickers))
        prev_tickers = new_tickers
        
        exit_date = dates[min(i + holding_days, len(dates) - 1)]
        
        port_ret = 0.0
        n_valid = 0
        for ticker, wgt in weights.items():
            price_dict = price_lookup.get(ticker, {})
            if date in price_dict and exit_date in price_dict:
                ret = price_dict[exit_date] / price_dict[date] - 1
                port_ret += wgt * ret
                n_valid += 1
        
        port_ret = port_ret * len(weights) / n_valid if n_valid > 0 else 0.0
        port_ret -= txn_cost
        
        periods.append({
            'entry_date': date,
            'exit_date': exit_date,
            'ret': port_ret,
            'cost': txn_cost,
            'n_positions': len(weights),
        })
    
    perf_df = pd.DataFrame(periods)
    if perf_df.empty:
        return perf_df
    
    perf_df['cum'] = (1 + perf_df['ret']).cumprod()
    perf_df['peak'] = perf_df['cum'].cummax()
    perf_df['drawdown'] = perf_df['cum'] / perf_df['peak'] - 1
    
    return perf_df


def compute_metrics(perf_df, holding_days=3):
    """Compute performance metrics from backtest results."""
    if perf_df.empty:
        return {'sharpe': 0, 'ann_ret': 0, 'max_dd': 0, 'win_rate': 0, 'n_periods': 0}
    
    n_periods = len(perf_df)
    total_days = n_periods * holding_days
    total_ret = perf_df['cum'].iloc[-1] - 1
    
    ann_ret = (1 + total_ret) ** (252 / max(1, total_days)) - 1
    ann_vol = np.std(perf_df['ret'], ddof=1) * np.sqrt(252 / max(1, holding_days))
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
    
    return {
        'sharpe': float(sharpe),
        'ann_ret': float(ann_ret),
        'max_dd': float(perf_df['drawdown'].min()),
        'total_ret': float(total_ret),
        'win_rate': float(np.mean(perf_df['ret'] > 0)),
        'n_periods': n_periods,
    }


def run_walkforward_backtest(price_data, sdl_df, mom_df, rev_df):
    """
    Run walk-forward LGBM backtest.
    
    1. Merge factors into a panel
    2. Walk-forward train/test splits
    3. LGBM predicts OOS scores
    4. Backtest on OOS predictions
    """
    from lightgbm import LGBMRegressor
    
    price_lookup = build_price_lookup(price_data)
    
    # Build factor panel
    lookup = {}
    for df, name in [(sdl_df, 'sdl'), (mom_df, 'mom_z'), (rev_df, 'rev_z')]:
        val_col = [c for c in df.columns if c not in ['date', 'code']][0]
        for _, r in df.iterrows():
            key = f"{r['date']}_{r['code']}"
            if key not in lookup:
                lookup[key] = {}
            lookup[key][name] = r[val_col]
    
    # Build panel with forward return
    rows = []
    for key, factors in lookup.items():
        date, code = key.split('_', 1)
        price_dict = price_lookup.get(code, {})
        if date not in price_dict:
            continue
        all_dates = sorted(price_dict.keys())
        idx = all_dates.index(date)
        if idx + 5 >= len(all_dates):
            continue
        ret5 = price_dict[all_dates[idx + 5]] / price_dict[date] - 1
        if abs(ret5) > 0.5:
            continue
        rows.append({
            'date': date,
            'code': code,
            'sdl': factors.get('sdl', np.nan),
            'mom_z': factors.get('mom_z', np.nan),
            'rev_z': factors.get('rev_z', np.nan),
            'ret_5d': ret5,
        })
    
    panel = pd.DataFrame(rows)
    print(f"  Panel: {len(panel):,} rows, {panel['date'].nunique()} dates")
    
    # Walk-forward
    features = ['sdl', 'mom_z', 'rev_z']
    all_dates = sorted(panel['date'].unique())
    n_dates = len(all_dates)
    fold_size = n_dates // 4
    
    oos_predictions = []
    all_ics = []
    fold_importances = []
    
    for fold in range(4):
        test_start = (fold + 1) * fold_size
        test_end = min(test_start + fold_size, n_dates)
        if test_start >= n_dates - 10:
            continue
        
        train_dates = set(all_dates[:test_start])
        test_dates = set(all_dates[test_start:test_end])
        
        train = panel[panel['date'].isin(train_dates)].dropna(subset=features + ['ret_5d'])
        test = panel[panel['date'].isin(test_dates)].dropna(subset=features + ['ret_5d'])
        
        if len(train) < 200 or len(test) < 50:
            continue
        
        model = LGBMRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            num_leaves=16, min_child_samples=20, subsample=0.8,
            colsample_bytree=0.8, random_state=42 + fold, verbose=-1, n_jobs=1
        )
        model.fit(train[features].values, train['ret_5d'].values)
        
        y_pred = model.predict(test[features].values)
        test_copy = test.copy()
        test_copy['pred'] = y_pred
        
        ic = test_copy.groupby('date').apply(
            lambda g: g['pred'].corr(g['ret_5d'], method='spearman')
        ).dropna()
        
        all_ics.extend(ic.values)
        oos_predictions.append(test_copy[['date', 'code', 'pred']])
        fold_importances.append(model.feature_importances_)
        
        print(f"  Fold {fold}: train to {all_dates[test_start-1]}, "
              f"test {all_dates[test_start]}..{all_dates[min(test_end,n_dates)-1]}, "
              f"IC={ic.mean():+.4f}")
    
    # Combine OOS predictions
    oos_all = pd.concat(oos_predictions, ignore_index=True)
    
    # Metrics
    ic_series = pd.Series(all_ics)
    ic_mean = ic_series.mean()
    ic_t = ic_mean / ic_series.std() * np.sqrt(len(ic_series)) if ic_series.std() > 0 else 0
    
    # Feature importance
    imp_df = pd.DataFrame(fold_importances, columns=features)
    feature_importance = {f: float(imp_df[f].mean()) for f in features}
    
    print(f"\n  OOS Rank IC: {ic_mean:+.4f} (t={ic_t:.2f})")
    print(f"  Feature Importance: {feature_importance}")
    
    # Backtest with best holding period (3d)
    sig = oos_all.rename(columns={'pred': 'score'})
    perf_df = run_backtest(sig, price_lookup, holding_days=3)
    metrics = compute_metrics(perf_df, holding_days=3)
    
    print(f"\n  Backtest (3d): Sharpe={metrics['sharpe']:.3f}, "
          f"AnnRet={metrics['ann_ret']:+.2%}, MaxDD={metrics['max_dd']:.2%}")
    
    return {
        'performance': perf_df,
        'metrics': metrics,
        'oos_ic_mean': float(ic_mean),
        'oos_ic_t_stat': float(ic_t),
        'feature_importance': feature_importance,
        'oos_predictions': oos_all,
    }
