"""
mf-factor-model — LightGBM-Powered Multi-Factor Equity Strategy

Public API:
    predict_multi_factor(date, code, sdl, mom_z, rev_z) -> float
        Returns combined signal score using trained LightGBM model.
    
    run_demo() -> dict
        Loads pre-computed results and runs demo analysis.
    
    run_full() -> dict
        Fetches live data, computes factors, trains LGBM, runs backtest.
"""
import json, warnings, pickle, numpy as np
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_DATA = _ROOT / 'data'

# Lazy-load the LGBM engine
_engine = None


def _get_engine():
    global _engine
    if _engine is None:
        from core._engine import LGBMEngine
        _engine = LGBMEngine()
    return _engine


def predict_multi_factor(date, code, sdl, mom_z, rev_z):
    """
    Predict combined score for a single stock on a given date.
    
    Parameters
    ----------
    date : str
        Trading date (YYYY-MM-DD)
    code : str
        Stock code (e.g., '600519')
    sdl : float
        SDL z-score
    mom_z : float
        Momentum z-score  
    rev_z : float
        Reversal z-score
    
    Returns
    -------
    float
        Combined signal score (higher = more bullish)
    """
    engine = _get_engine()
    return engine.predict(date, code, sdl, mom_z, rev_z)


def run_demo():
    """Run demo mode with pre-computed results."""
    print("📊 Multi-Factor Model — Demo Mode")
    print("=" * 50)
    
    res_file = _DATA / 'results.json'
    if not res_file.exists():
        raise FileNotFoundError("Pre-computed results not found. Run 'run.py --full' first.")
    
    res = json.loads(res_file.read_text('utf-8'))
    bt = res['backtest']
    
    print(f"\n  OOS Rank IC:  {res['oos_ic_mean']:+.4f} (t={res['oos_ic_t_stat']:.2f})")
    print(f"  Best Holding: {res['best_holding']}d")
    print(f"  Sharpe:       {bt['sharpe']:.3f}")
    print(f"  Ann Return:   {bt['ann_ret']:+.2%}")
    print(f"  Max DD:       {bt['max_dd']:.2%}")
    print(f"  Win Rate:     {bt['win_rate']:.1%}")
    print(f"\n  Feature Importance:")
    for f, imp in res['feature_importance'].items():
        print(f"    {f}: {imp:.0f}")
    
    print(f"\n  📁 Charts: {_ROOT / 'results' / 'charts'}")
    return res


def run_full():
    """Run full pipeline: fetch → compute → train → backtest."""
    from src.data_fetcher import fetch_all_data, compute_factors
    from src.backtest import run_walkforward_backtest
    from src.visualization import generate_all_charts
    
    print("🚀 Multi-Factor Model — Full Pipeline")
    print("=" * 50)
    
    price_data, fund_flow = fetch_all_data()
    sdl, mom, rev = compute_factors(price_data, fund_flow)
    results = run_walkforward_backtest(price_data, sdl, mom, rev)
    generate_all_charts(results)
    
    return results
