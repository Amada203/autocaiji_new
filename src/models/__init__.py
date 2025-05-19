# 使models目录成为Python包
from .model_trainer import ModelTrainer
from .predictors import BasePredictor, ProphetPredictor, ChangeDetector
from .fusion_model import ProphetLGBMFusion

# 为保持向后兼容，同时导出FusionModel别名
FusionModel = ProphetLGBMFusion

__all__ = [
    'ModelTrainer', 
    'BasePredictor', 
    'ProphetPredictor', 
    'ChangeDetector',
    'ProphetLGBMFusion',
    'FusionModel'
]