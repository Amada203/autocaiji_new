import logging
from abc import ABC, abstractmethod
import pandas as pd

class BaseModel(ABC):
    """所有模型的基础抽象类"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """训练模型"""
        pass
        
    @abstractmethod 
    def predict(self, X: pd.DataFrame):
        """执行预测"""
        pass
        
    @abstractmethod
    def save(self, path: str):
        """保存模型"""
        pass
        
    @abstractmethod
    def load(self, path: str):
        """加载模型""" 
        pass
        
    def _validate_input(self, X: pd.DataFrame):
        """验证输入数据"""
        if X is None or X.empty:
            self.logger.error("输入数据为空")
            return False
        return True