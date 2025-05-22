# 使models目录成为Python包
from .model_trainer import ModelTrainer
from .predictors import BasePredictor, ProphetPredictor, ChangeDetector
from .fusion_model import PriceChangePredictor

# 新版本统一使用PriceChangePredictor
__all__ = [
    'ModelTrainer', 
    'BasePredictor', 
    'ProphetPredictor', 
    'ChangeDetector',
    'PriceChangePredictor'
]