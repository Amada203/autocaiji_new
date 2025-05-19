"""
This package contains the new model predictors for price prediction and change detection.
"""
from .base_predictor import BasePredictor
from .prophet_predictor import ProphetPredictor
from .change_detector import ChangeDetector

__all__ = ['BasePredictor', 'ProphetPredictor', 'ChangeDetector']