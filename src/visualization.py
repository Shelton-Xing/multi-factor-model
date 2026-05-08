"""Visualization module — publication-quality charts."""
import json, numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

_ROOT = Path(__file__).parent.parent
_CHART_DIR = _ROOT / 'results' / 'charts'
_CHART_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'font.size': 10, 'axes.titlesize': 12, 'axes.labelsize': 10,
    'figure.facecolor': '#FAFAFA', 'axes.facecolor': '#FAFAFA',
})


def plot_feature_importance(feature_importance):
    """Plot LGBM feature importance."""
    fig, ax = plt.subplots(figsize=(7, 4))
    features = list(feature_importance.keys())
    imps = [feature_importance[f] for f in features]
    labels = ['SDL\n(Main Force Flow)', 'Momentum\n(20d Return)', 'Reversal\n(5d Return)']
    colors = ['#2166AC', '#D6604D', '#4DAF4A']
    bars = ax.barh(labels, imps, color=colors, height=0.6, edgecolor='white', linewidth=1.5)
    for bar, val in zip(bars, imps):
        ax.text(bar.get_width() + 10, bar.get_y() + bar.get_height() / 2,
                f'{val:.0f}', va='center', fontsize=11, fontweight='bold')
    ax.set_xlabel('LightGBM Feature Importance', fontsize=11, fontweight='bold')
    ax.set_title('Factor Contribution (Walk-Forward OOS)', fontsize=13, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim(0, max(imps) * 1.25)
    plt.tight_layout()
    fig.savefig(_CHART_DIR / 'feature_importance.png', dpi=200, bbox_inches='tight')
    plt.close()


def plot_equity_curve(perf_df, metrics):
    """Plot equity curve with drawdown."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), gridspec_kw={'height_ratios': [3, 1]})
    entry_dates = pd.to_datetime(perf_df['entry_date'])
    ax1.plot(entry_dates, perf_df['cum'], color='#2166AC', linewidth=1.5)
    ax1.fill_between(entry_dates, perf_df['cum'], alpha=0.15, color='#2166AC')
    ax1.axhline(y=1, color='#888888', linewidth=0.5, linestyle='--')
    ax1.set_ylabel('Cumulative Return', fontsize=11, fontweight='bold')
    ax1.set_title(f'LightGBM Multi-Factor — Equity Curve\n'
                  f'Sharpe={metrics["sharpe"]:.2f}, AnnRet={metrics["ann_ret"]:+.1%}',
                  fontsize=13, fontweight='bold')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax2.fill_between(entry_dates, perf_df['drawdown'] * 100, 0, color='#D6604D', alpha=0.4)
    ax2.set_ylabel('Drawdown (%)', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Date', fontsize=11, fontweight='bold')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    plt.tight_layout()
    fig.savefig(_CHART_DIR / 'equity_curve.png', dpi=200, bbox_inches='tight')
    plt.close()


def plot_ic_analysis(oos_predictions, price_lookup):
    """Plot OOS Rank IC distribution and time series."""
    # Compute IC per date
    rows = []
    for _, r in oos_predictions.iterrows():
        price_dict = price_lookup.get(r['code'], {})
        if r['date'] in price_dict:
            dates = sorted(price_dict.keys())
            idx = dates.index(r['date'])
            if idx + 5 < len(dates):
                ret5 = price_dict[dates[idx + 5]] / price_dict[r['date']] - 1
                if abs(ret5) < 0.5:
                    rows.append({'date': r['date'], 'pred': r['pred'], 'ret_5d': ret5})
    
    df = pd.DataFrame(rows)
    if df.empty:
        return
    
    ics = df.groupby('date').apply(lambda g: g['pred'].corr(g['ret_5d'], method='spearman')).dropna()
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax = axes[0]
    ax.hist(ics, bins=20, color='#2166AC', alpha=0.7, edgecolor='white')
    ax.axvline(ics.mean(), color='#D6604D', linestyle='--', linewidth=1.5,
               label=f'Mean={ics.mean():+.4f}')
    ax.axvline(0, color='#888888', linewidth=0.8)
    ax.set_xlabel('Rank IC', fontsize=11, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax.set_title('OOS Rank IC Distribution', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax = axes[1]
    ax.plot(range(len(ics)), ics.values, color='#2166AC', linewidth=0.8, alpha=0.6)
    ax.axhline(ics.mean(), color='#D6604D', linestyle='--', linewidth=1.5,
               label=f'Mean={ics.mean():+.4f}')
    ax.axhline(0, color='#888888', linewidth=0.8)
    t_stat = ics.mean() / ics.std() * np.sqrt(len(ics)) if ics.std() > 0 else 0
    ax.set_xlabel('Trading Day', fontsize=11, fontweight='bold')
    ax.set_ylabel('Rank IC', fontsize=11, fontweight='bold')
    ax.set_title(f'Rank IC Time Series (t={t_stat:.2f})', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    fig.savefig(_CHART_DIR / 'ic_analysis.png', dpi=200, bbox_inches='tight')
    plt.close()


def plot_period_returns(perf_df):
    """Plot period returns bar chart."""
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = ['#4DAF4A' if r > 0 else '#D6604D' for r in perf_df['ret']]
    ax.bar(range(len(perf_df)), perf_df['ret'] * 100, color=colors, alpha=0.8, width=0.7)
    ax.axhline(y=0, color='#888888', linewidth=0.8)
    ax.set_xlabel('Trading Period', fontsize=11, fontweight='bold')
    ax.set_ylabel('Return (%)', fontsize=11, fontweight='bold')
    ax.set_title('Period Returns (3-day Holding)', fontsize=12, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    fig.savefig(_CHART_DIR / 'period_returns.png', dpi=200, bbox_inches='tight')
    plt.close()


def generate_all_charts(results):
    """Generate all charts from backtest results."""
    print("\n[Charts] Generating...")
    
    plot_feature_importance(results['feature_importance'])
    print("  feature_importance.png ✓")
    
    plot_equity_curve(results['performance'], results['metrics'])
    print("  equity_curve.png ✓")
    
    plot_period_returns(results['performance'])
    print("  period_returns.png ✓")
    
    print(f"  📁 Saved to {_CHART_DIR}")
