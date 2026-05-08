#!/usr/bin/env python3
"""
mf-factor-model: LightGBM-Powered Multi-Factor Equity Strategy

Usage:
    python run.py --demo       # Show pre-computed results
    python run.py --full       # Full pipeline (download + compute + train + backtest)
    python run.py --charts     # Regenerate charts from cached results
"""
import sys, warnings, json
from pathlib import Path
warnings.filterwarnings('ignore')

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))


def demo_mode():
    """Load and display pre-computed results."""
    from core import run_demo
    return run_demo()


def full_mode():
    """Run full pipeline."""
    from core import run_full
    return run_full()


def charts_only():
    """Regenerate charts from cached results."""
    import pandas as pd
    from src.backtest import compute_metrics
    from src.visualization import generate_all_charts
    from src.data_fetcher import build_price_lookup
    
    data_dir = ROOT / 'data'
    
    # Load backtest periods
    perf = pd.read_csv(data_dir / 'backtest_periods.csv')
    metrics = compute_metrics(perf, holding_days=3)
    
    # Load results
    res = json.loads((data_dir / 'results.json').read_text('utf-8'))
    
    # Load predictions for IC chart
    preds = pd.read_csv(data_dir / 'predictions.csv')
    
    # Build price_lookup from stored data
    price_data = json.loads((data_dir / 'price_data.json').read_text('utf-8'))
    # Convert to dict format
    price_lookup = price_data  # Already in {code: {date: close}} format
    
    results = {
        'performance': perf,
        'metrics': metrics,
        'oos_ic_mean': res['oos_ic_mean'],
        'oos_ic_t_stat': res['oos_ic_t_stat'],
        'feature_importance': res['feature_importance'],
    }
    
    from src.backtest import build_price_lookup as bpl
    # Need price data properly formatted for IC chart
    # Use the compact format directly
    
    generate_all_charts(results)
    return results


if __name__ == '__main__':
    if '--demo' in sys.argv:
        demo_mode()
    elif '-d' in sys.argv:
        demo_mode()
    elif '--full' in sys.argv:
        full_mode()
    elif '-f' in sys.argv:
        full_mode()
    elif '--charts' in sys.argv:
        charts_only()
    else:
        print(__doc__)
