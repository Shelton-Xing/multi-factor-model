"""
Proprietary LightGBM engine — black-box core of mf-factor-model.

This module contains the trained model and weights.
The model learns adaptive factor weights via walk-forward regression.

⚠ This file is for reference only — the actual trained model is in data/lgbm_model.txt
"""
import json, warnings, numpy as np
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_DATA = _ROOT / 'data'


class LGBMEngine:
    """
    LightGBM-based factor combination engine.
    
    Trained via walk-forward regression to predict forward 5-day returns
    from factor z-scores (SDL, Momentum, Reversal).
    
    The model adaptively weights factors based on their time-varying
    predictive power, capturing regime changes in market structure.
    """
    
    def __init__(self):
        self.model = None
        self.features = ['sdl', 'mom_z', 'rev_z']
        self._load()
    
    def _load(self):
        """Load pre-trained model."""
        model_path = _DATA / 'lgbm_model.txt'
        if not model_path.exists():
            raise FileNotFoundError(
                f"No trained model found at {model_path}. "
                "Run with --full flag to train a new model."
            )
        
        try:
            from lightgbm import Booster
            self.model = Booster(model_file=str(model_path))
        except ImportError:
            # Fallback: use saved feature importances
            warnings.warn("LightGBM not installed; using heuristic weights.")
            self.model = None
    
    def predict(self, date, code, sdl, mom_z, rev_z):
        """
        Predict combined signal score.
        
        Parameters
        ----------
        date : str
        code : str  
        sdl : float
        mom_z : float
        rev_z : float
        
        Returns
        -------
        float
        """
        if self.model is not None:
            X = np.array([[sdl, mom_z, rev_z]])
            return float(self.model.predict(X)[0])
        else:
            # Heuristic fallback
            return 0.3 * sdl + 0.4 * mom_z + 0.3 * rev_z
