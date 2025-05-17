"""
价格预测模型 - 新版本
基于Prophet捕获时间模式，LightGBM预测价格变化概率
迁移自旧版PriceModel
"""
import pandas as pd
import numpy as np
from prophet import Prophet
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score
import logging
import os
import pickle
from src.models.predictors.base_predictor import BasePredictor

logger = logging.getLogger(__name__)

class ProphetPredictor(BasePredictor):
    """
    价格变化概率预测模型
    利用Prophet捕获时间序列的趋势和季节性，然后使用LightGBM预测变化概率
    迁移自旧版PriceModel
    """
    
    def __init__(self, prophet_params=None, lgbm_params=None, change_threshold=0.01):
        """
        初始化模型
        
        Args:
            prophet_params: Prophet模型参数字典
            lgbm_params: LightGBM模型参数字典
            change_threshold: 价格变化阈值（相对变化比例），超过该阈值认为价格发生变化
        """
        # 确保启用必要的季节性组件
        self.prophet_params = {
            'yearly_seasonality': True,
            'weekly_seasonality': True,
            'daily_seasonality': False,
            **(prophet_params or {})
        }
        self.lgbm_params = lgbm_params or {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'n_estimators': 100
        }
        self.change_threshold = change_threshold
        
        # 初始化子模型
        self.prophet_model = Prophet(**self.prophet_params)
        self.lgbm_model = None
        
        # 状态标志
        self.fitted = False
        self.feature_names = None
    
    def _extract_prophet_features(self, data):
        """
        健壮的Prophet特征提取方法
        
        Args:
            data: 包含时间戳的DataFrame
            
        Returns:
            Prophet特征DataFrame
        """
        # 使用Prophet模型进行预测
        prophet_forecast = self.prophet_model.predict(data[['ds']])
        
        # 初始化特征DataFrame
        features = data[['ds']].copy()
        
        # 确保所有必要特征都有值
        features['trend'] = prophet_forecast.get('trend', 0)
        features['yearly'] = prophet_forecast.get('yearly', 0)
        features['weekly'] = prophet_forecast.get('weekly', 0)
        features['holiday_effect'] = prophet_forecast.get('holidays', 0)
        
        return features

    # [保留其他原有方法不变...]
    
    def _build_features(self, data, prophet_features):
        """
        构建完整特征集（保持原始数据顺序）
        
        Args:
            data: 原始数据
            prophet_features: Prophet提取的特征
            
        Returns:
            特征DataFrame
        """
        # 创建特征副本保持原始索引
        features = data.copy()
        
        # 合并Prophet特征（不改变顺序）
        for col in prophet_features.columns:
            if col != 'ds':  # 避免重复合并ds列
                features[col] = prophet_features[col]
        
        # [保留其他特征工程代码不变...]
        
        return features

    def fit(self, *args):
        """
        训练Prophet和LightGBM模型
        
        参数形式:
        1. fit(X, y) - X: 特征DataFrame, y: 目标Series
        2. fit(data) - data: 包含特征和目标列的DataFrame(默认目标列为'change_flag')
        """
        try:
            if len(args) == 1:
                # 单参数调用: data包含特征和目标
                data = args[0]
                X = data.drop(columns=['change_flag'], errors='ignore')
                y = data['change_flag']
            elif len(args) == 2:
                # 双参数调用: X和y分开
                X, y = args
            else:
                raise ValueError("fit()接受1或2个参数")
            
            # 准备Prophet训练数据
            prophet_df = X[['ds']].copy()
            prophet_df['y'] = y
            
            # 训练Prophet模型
            self.prophet_model.fit(prophet_df)
            
            # 提取Prophet特征
            prophet_features = self._extract_prophet_features(X)
            
            # 确保所有特征为数值类型
            numeric_features = prophet_features.copy()
            for col in numeric_features.columns:
                if np.issubdtype(numeric_features[col].dtype, np.datetime64):
                    numeric_features[col] = numeric_features[col].astype(np.int64) // 10**9  # 转换为Unix时间戳
                elif not np.issubdtype(numeric_features[col].dtype, np.number):
                    numeric_features[col] = pd.to_numeric(numeric_features[col], errors='coerce')
            
            # 检查并处理NaN值
            if numeric_features.isnull().any().any():
                logger.warning("特征中存在NaN值，将进行填充")
                numeric_features = numeric_features.fillna(0)
            
            # 训练LightGBM模型
            self.lgbm_model = lgb.LGBMClassifier(**self.lgbm_params)
            self.lgbm_model.fit(numeric_features, y)
            
            self.fitted = True
            logger.info("模型训练完成")
        except Exception as e:
            logger.error(f"训练失败: {str(e)}")
            raise

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """生成预测结果"""
        if not self.fitted:
            raise RuntimeError("模型未训练，请先调用fit()方法")
            
        prophet_features = self._extract_prophet_features(X)
        return self.lgbm_model.predict_proba(prophet_features)[:, 1]

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """评估模型性能"""
        preds = self.predict(X)
        return {
            'roc_auc': roc_auc_score(y, preds),
            'average_precision': average_precision_score(y, preds),
            'threshold': self.change_threshold
        }

    def save(self, filepath: str):
        """保存模型到文件"""
        if not self.fitted:
            raise RuntimeError("模型未训练，无法保存")
            
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump({
                'prophet_model': self.prophet_model,
                'lgbm_model': self.lgbm_model,
                'params': {
                    'prophet_params': self.prophet_params,
                    'lgbm_params': self.lgbm_params,
                    'change_threshold': self.change_threshold
                }
            }, f)
        logger.info(f"模型已保存到 {filepath}")

    def load(self, filepath: str):
        """从文件加载模型"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            
        self.prophet_model = data['prophet_model']
        self.lgbm_model = data['lgbm_model']
        self.prophet_params = data['params']['prophet_params']
        self.lgbm_params = data['params']['lgbm_params']
        self.change_threshold = data['params']['change_threshold']
        self.fitted = True
        logger.info(f"模型已从 {filepath} 加载")