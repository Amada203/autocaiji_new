import abc
import logging
import pandas as pd

logger = logging.getLogger(__name__)

class BaseModel(abc.ABC):
    """
    模型基类，定义所有模型共有的接口
    """
    
    def __init__(self):
        self.fitted = False
    
    @abc.abstractmethod
    def fit(self, data, y=None):
        """
        训练模型
        
        Args:
            data: 训练数据
            y: 目标变量(可选)
            
        Returns:
            self
        """
        pass
    
    @abc.abstractmethod
    def predict(self, data):
        """
        生成预测
        
        Args:
            data: 预测数据
            
        Returns:
            预测结果
        """
        pass
    
    @abc.abstractmethod
    def evaluate(self, test_data, y=None):
        """
        评估模型
        
        Args:
            test_data: 测试数据
            y: 目标变量(可选)
            
        Returns:
            评估指标
        """
        pass
    
    def save(self, path):
        """
        保存模型到文件
        
        Args:
            path: 保存路径
        """
        raise NotImplementedError("save方法需要子类实现")
    
    def load(self, path):
        """
        从文件加载模型
        
        Args:
            path: 模型路径
            
        Returns:
            self
        """
        raise NotImplementedError("load方法需要子类实现")