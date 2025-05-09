"""
Prophet模型实现
"""
import pandas as pd
import numpy as np
from prophet import Prophet
from models.base_model import BaseModel
import pickle
import os

class ProphetModel(BaseModel):
    """Prophet时间序列预测模型包装器"""
    
    def __init__(self, **prophet_params):
        """
        初始化Prophet模型
        
        参数:
            prophet_params: Prophet模型参数
        """
        self.model = Prophet(**prophet_params)
        self.fitted = False
        
    def fit(self, df, y=None):
        """
        训练Prophet模型
        
        参数:
            df: 包含'ds'和'y'列的DataFrame
        
        返回:
            self
        """
        # 确保数据格式正确
        if 'ds' not in df.columns or 'y' not in df.columns:
            raise ValueError("数据必须包含'ds'和'y'列")
            
        self.model.fit(df)
        self.fitted = True
        return self
    
    def predict(self, future_df=None, periods=30):
        """
        生成预测
        
        参数:
            future_df: 包含'ds'列的未来日期DataFrame
            periods: 如果future_df为None，预测的天数
            
        返回:
            包含预测结果的DataFrame
        """
        if not self.fitted:
            raise ValueError("模型尚未训练，请先调用fit方法")
            
        if future_df is None:
            future_df = self.model.make_future_dataframe(periods=periods)
            
        return self.model.predict(future_df)
    
    def evaluate(self, test_df, metrics=None):
        """
        评估模型
        
        参数:
            test_df: 测试数据，包含'ds'和'y'列
            metrics: 评估指标列表
            
        返回:
            包含各项指标的字典
        """
        if not self.fitted:
            raise ValueError("模型尚未训练，请先调用fit方法")
            
        # 默认评估指标
        if metrics is None:
            metrics = ['mae', 'mse', 'rmse', 'mape']
            
        # 生成预测
        predictions = self.predict(test_df[['ds']])
        
        # 计算指标
        results = {}
        y_true = test_df['y'].values
        y_pred = predictions['yhat'].values
        
        for metric in metrics:
            if metric == 'mae':
                results[metric] = np.mean(np.abs(y_true - y_pred))
            elif metric == 'mse':
                results[metric] = np.mean((y_true - y_pred)**2)
            elif metric == 'rmse':
                results[metric] = np.sqrt(np.mean((y_true - y_pred)**2))
            elif metric == 'mape':
                mask = y_true != 0
                results[metric] = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
                
        return results
    
    def save(self, path):
        """保存模型到指定路径"""
        if not self.fitted:
            raise ValueError("模型尚未训练，无法保存")
            
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)
    
    def load(self, path):
        """从指定路径加载模型"""
        with open(path, 'rb') as f:
            self.model = pickle.load(f)
        self.fitted = True
        return self 