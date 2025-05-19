"""
Prophet + LightGBM 融合模型
基于残差处理方法，将Prophet模型的预测残差交给LightGBM继续学习
"""
import pandas as pd
import numpy as np
from prophet import Prophet
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import logging
import os
import pickle
from .base_model import BaseModel

logger = logging.getLogger(__name__)

class ProphetLGBMFusion(BaseModel):
    """
    Prophet + LightGBM 融合模型
    利用Prophet捕获时间序列的趋势和季节性，然后使用LightGBM预测残差
    """
    
    def __init__(self, prophet_params=None, lgbm_params=None):
        """
        初始化融合模型
        
        Args:
            prophet_params: Prophet模型参数字典
            lgbm_params: LightGBM模型参数字典
        """
        self.prophet_params = prophet_params or {}
        self.lgbm_params = lgbm_params or {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9
        }
        
        # 初始化子模型
        self.prophet_model = Prophet(**self.prophet_params)
        self.lgbm_model = None
        
        # 状态标志
        self.fitted = False
        self.feature_names = None
        
    def fit(self, data, y=None):
        """
        训练融合模型
        
        Args:
            data: 包含'ds'和'y'列的DataFrame，或者包含特征的DataFrame和目标变量y
            y: 目标变量，当data不包含'y'列时使用
            
        Returns:
            self
        """
        logger.info("开始训练Prophet+LightGBM融合模型")
        
        # 处理输入格式
        if y is not None:
            # 特征矩阵和目标变量分开传入的情况
            if 'ds' not in data.columns:
                raise ValueError("数据必须包含'ds'列")
            train_df = data.copy()
            train_df['y'] = y
        else:
            # ds和y在同一个DataFrame的情况
            if 'ds' not in data.columns or 'y' not in data.columns:
                raise ValueError("数据必须包含'ds'和'y'列")
            train_df = data.copy()
        
        # 确保日期格式正确
        train_df['ds'] = pd.to_datetime(train_df['ds'])
        
        # 1. 训练Prophet模型
        logger.info("训练Prophet模型...")
        self.prophet_model.fit(train_df[['ds', 'y']])
        
        # 2. 生成Prophet预测
        prophet_pred = self.prophet_model.predict(train_df[['ds']])
        
        # 3. 计算残差
        residuals = train_df['y'].values - prophet_pred['yhat'].values
        
        # 4. 准备LightGBM训练特征
        X_train = self._prepare_features(train_df, prophet_pred)
        self.feature_names = X_train.columns.tolist()
        
        # 5. 训练LightGBM模型预测残差
        logger.info("训练LightGBM模型预测残差...")
        self.lgbm_model = lgb.LGBMRegressor(**self.lgbm_params)
        self.lgbm_model.fit(X_train, residuals)
        
        self.fitted = True
        logger.info("Prophet+LightGBM融合模型训练完成")
        return self
    
    def predict(self, future_df):
        """
        生成预测
        
        Args:
            future_df: 包含'ds'列的DataFrame，表示要预测的未来时间点
            
        Returns:
            包含预测结果的DataFrame
        """
        if not self.fitted:
            raise ValueError("模型尚未训练，请先调用fit方法")
            
        # 确保日期格式正确
        future_df = future_df.copy()
        future_df['ds'] = pd.to_datetime(future_df['ds'])
        
        # 1. Prophet预测
        prophet_forecast = self.prophet_model.predict(future_df[['ds']])
        
        # 2. 准备LightGBM特征
        X_future = self._prepare_features(future_df, prophet_forecast)
        
        # 确保特征列匹配
        missing_cols = set(self.feature_names) - set(X_future.columns)
        for col in missing_cols:
            X_future[col] = 0
        X_future = X_future[self.feature_names]
        
        # 3. LightGBM预测残差
        residual_forecast = self.lgbm_model.predict(X_future)
        
        # 4. 组合预测结果
        final_forecast = prophet_forecast.copy()
        final_forecast['residual'] = residual_forecast
        final_forecast['yhat_original'] = final_forecast['yhat'].copy() 
        final_forecast['yhat'] = final_forecast['yhat'] + residual_forecast
        
        return final_forecast
    
    def evaluate(self, test_df, y=None):
        """
        评估模型
        
        Args:
            test_df: 包含'ds'和'y'列的测试数据，或者包含特征的DataFrame
            y: 当test_df不包含'y'列时的目标变量
            
        Returns:
            包含评估指标的字典
        """
        # 处理输入格式
        if y is not None:
            if 'ds' not in test_df.columns:
                raise ValueError("测试数据必须包含'ds'列")
            eval_df = test_df.copy()
            eval_df['y'] = y
        else:
            if 'ds' not in test_df.columns or 'y' not in test_df.columns:
                raise ValueError("测试数据必须包含'ds'和'y'列")
            eval_df = test_df.copy()
        
        # 生成预测
        predictions = self.predict(eval_df)
        
        # 计算指标
        y_true = eval_df['y'].values
        y_pred = predictions['yhat'].values
        y_pred_prophet = predictions['yhat_original'].values
        
        # 计算各种指标
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        # 计算Prophet单独的指标用于对比
        mae_prophet = mean_absolute_error(y_true, y_pred_prophet)
        mse_prophet = mean_squared_error(y_true, y_pred_prophet)
        rmse_prophet = np.sqrt(mse_prophet)
        r2_prophet = r2_score(y_true, y_pred_prophet)
        
        # 计算改进百分比
        improvement_mae = (mae_prophet - mae) / mae_prophet * 100
        improvement_rmse = (rmse_prophet - rmse) / rmse_prophet * 100
        
        metrics = {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'mae_prophet': mae_prophet,
            'mse_prophet': mse_prophet,
            'rmse_prophet': rmse_prophet,
            'r2_prophet': r2_prophet,
            'improvement_mae_percent': improvement_mae,
            'improvement_rmse_percent': improvement_rmse
        }
        
        return metrics
    
    def _prepare_features(self, data, prophet_predictions):
        """
        准备LightGBM特征
        
        Args:
            data: 包含时间戳的DataFrame
            prophet_predictions: Prophet预测结果
            
        Returns:
            特征DataFrame
        """
        features = pd.DataFrame()
        
        # 提取Prophet分解组件
        for component in ['trend', 'yearly', 'weekly']:
            if component in prophet_predictions:
                features[component] = prophet_predictions[component]
        
        # 时间特征
        features['day_of_week'] = pd.to_datetime(data['ds']).dt.dayofweek
        features['month'] = pd.to_datetime(data['ds']).dt.month
        features['day'] = pd.to_datetime(data['ds']).dt.day
        features['quarter'] = pd.to_datetime(data['ds']).dt.quarter
        features['is_weekend'] = features['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
        
        # 滞后特征：如果数据中有y值，可以添加滞后特征
        if 'y' in data.columns:
            # 对每个时间序列分别处理
            if 'sku_id' in data.columns:
                for lag in [1, 2, 3, 7]:
                    features[f'lag_{lag}'] = data.groupby('sku_id')['y'].shift(lag)
            else:
                # 单一时间序列
                for lag in [1, 2, 3, 7]:
                    features[f'lag_{lag}'] = data['y'].shift(lag)
                    
        # 移除缺失值
        features = features.fillna(0)
        
        return features
    
    def save(self, path):
        """
        保存模型到磁盘
        
        Args:
            path: 保存路径
        """
        if not self.fitted:
            raise ValueError("模型尚未训练，无法保存")
        
        # 创建目录
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # 保存模型
        model_dict = {
            'prophet_model': self.prophet_model,
            'lgbm_model': self.lgbm_model,
            'prophet_params': self.prophet_params,
            'lgbm_params': self.lgbm_params,
            'feature_names': self.feature_names,
            'fitted': self.fitted
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_dict, f)
            
        logger.info(f"模型已保存到 {path}")
    
    def load(self, path):
        """
        从磁盘加载模型
        
        Args:
            path: 模型路径
            
        Returns:
            self
        """
        with open(path, 'rb') as f:
            model_dict = pickle.load(f)
        
        self.prophet_model = model_dict['prophet_model']
        self.lgbm_model = model_dict['lgbm_model']
        self.prophet_params = model_dict['prophet_params']
        self.lgbm_params = model_dict['lgbm_params']
        self.feature_names = model_dict['feature_names']
        self.fitted = model_dict['fitted']
        
        logger.info(f"模型已从 {path} 加载")
        return self