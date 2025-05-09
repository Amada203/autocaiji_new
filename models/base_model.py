"""
基础模型类，定义所有模型的通用接口
"""
from abc import ABC, abstractmethod

class BaseModel(ABC):
    """所有预测模型的基类"""
    
    @abstractmethod
    def fit(self, X, y=None):
        """训练模型"""
        pass
    
    @abstractmethod
    def predict(self, X):
        """使用模型进行预测"""
        pass
    
    @abstractmethod
    def evaluate(self, X, y):
        """评估模型性能"""
        pass
    
    @abstractmethod
    def save(self, path):
        """保存模型到指定路径"""
        pass
    
    @abstractmethod
    def load(self, path):
        """从指定路径加载模型"""
        pass 