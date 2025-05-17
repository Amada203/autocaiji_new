"""
预测器基类，定义所有预测器的通用接口
迁移自旧版BaseModel
"""
from abc import ABC, abstractmethod

class BasePredictor(ABC):
    """所有预测器的基类"""
    
    @abstractmethod
    def fit(self, X, y=None):
        """训练模型
        
        Args:
            X: 特征数据
            y: 目标变量（可选）
        """
        pass
    
    @abstractmethod
    def predict(self, X):
        """使用模型进行预测
        
        Args:
            X: 特征数据
            
        Returns:
            预测结果
        """
        pass
    
    @abstractmethod
    def evaluate(self, X, y):
        """评估模型性能
        
        Args:
            X: 特征数据
            y: 真实标签
            
        Returns:
            评估指标字典
        """
        pass
    
    @abstractmethod
    def save(self, path):
        """保存模型到指定路径
        
        Args:
            path: 保存路径
        """
        pass
    
    @abstractmethod
    def load(self, path):
        """从指定路径加载模型
        
        Args:
            path: 模型路径
            
        Returns:
            self
        """
        pass