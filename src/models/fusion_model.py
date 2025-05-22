"""
Prophet + LightGBM 融合模型
基于并行预测架构，同时使用Prophet和LightGBM预测价格变动概率
"""
import pandas as pd
import numpy as np
from prophet import Prophet
import lightgbm as lgb
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix
)
import logging
import os
import pickle
from typing import Dict, List
from .base_model import BaseModel

logger = logging.getLogger(__name__)

class PriceChangePredictor(BaseModel):
    """
    价格变动预测模型
    使用Prophet预测时间序列模式，LightGBM预测价格变动概率
    """
    
    def __init__(self, prophet_params=None, lgbm_params=None):
        """
        初始化预测模型
        
        Args:
            prophet_params: Prophet模型参数字典
            lgbm_params: LightGBM模型参数字典
        """
        self.prophet_params = prophet_params or {
            'yearly_seasonality': True,
            'weekly_seasonality': True,
            'daily_seasonality': False
        }
        self.lgbm_params = lgbm_params or {
            'objective': 'binary',
            'metric': 'auc',
            'learning_rate': 0.05,
            'num_leaves': 31,
            'feature_fraction': 0.8,
            'scale_pos_weight': 3  # 处理样本不平衡
        }
        
        # 模型组件
        self.prophet_models = {}  # 每个SKU一个Prophet模型
        self.lgbm_model = None    # 全局LightGBM分类器
        self.threshold = 0.5      # 初始分类阈值
        
        # 特征配置
        self.features = [
            # 时间特征
            'day_of_week', 'is_weekend', 'month', 'day',
            
            # 历史价格特征
            'price_mean_7d', 'price_std_7d', 'price_mean_14d', 'price_std_14d',
            
            # 价格变动模式
            'days_since_last_change', 'change_freq_7d', 'change_freq_30d',
            
            # 价格趋势特征
            'price_trend_7d', 'price_trend_14d',
            
            # Prophet特征
            'prophet_trend', 'prophet_yhat'
        ]
        
        # 状态标志
        self.fitted = False
        
    def fit(self, train_df, val_df=None):
        """
        训练价格变动预测模型
        
        Args:
            train_df: 训练数据DataFrame，需包含:
                - sku_id: SKU标识
                - date: 日期
                - discount_price: 折扣价格
            val_df: 验证数据DataFrame（可选），结构与train_df相同
            
        Returns:
            self
        """
        logger.info("开始训练价格变动预测模型")
        
        # 数据预处理
        train_df = self._preprocess_data(train_df)
        if val_df is not None:
            val_df = self._preprocess_data(val_df)
        
        # 1. 训练Prophet模型（按SKU分组）
        logger.info("训练Prophet模型...")
        for sku_id, group in train_df.groupby('sku_id'):
            if len(group) >= 30:  # 数据充足才训练
                prophet_data = group[['date', 'discount_price']].rename(
                    columns={'date': 'ds', 'discount_price': 'y'}
                )
                model = Prophet(**self.prophet_params)
                model.fit(prophet_data)
                self.prophet_models[sku_id] = model
        
        # 2. 生成Prophet特征
        logger.info("生成Prophet特征...")
        train_df = self._add_prophet_features(train_df)
        if val_df is not None:
            val_df = self._add_prophet_features(val_df)
        
        # 3. 训练LightGBM模型
        logger.info("训练LightGBM分类器...")
        self.lgbm_model = lgb.LGBMClassifier(**self.lgbm_params)
        
        if val_df is not None:
            self.lgbm_model.fit(
                train_df[self.features],
                train_df['price_change'],
                eval_set=[(val_df[self.features], val_df['price_change'])],
                early_stopping_rounds=50,
                verbose=100
            )
            
            # 4. 优化分类阈值（确保召回率≥95%）
            logger.info("优化分类阈值...")
            self.threshold = self._optimize_threshold(val_df)
        else:
            self.lgbm_model.fit(
                train_df[self.features],
                train_df['price_change']
            )
        
        self.fitted = True
        logger.info("价格变动预测模型训练完成")
        return self
    
    def _preprocess_data(self, df):
        """数据预处理：计算价格变动标签和特征"""
        # 确保按SKU和日期排序
        df = df.sort_values(['sku_id', 'date'])
        
        # 计算价格变动标签
        df['prev_price'] = df.groupby('sku_id')['discount_price'].shift(1)
        df['price_change'] = (df['discount_price'] != df['prev_price']).astype(int)
        
        # 填充第一个记录的NaN值
        df['price_change'] = df['price_change'].fillna(0)
        
        return df
    
    def _add_prophet_features(self, df):
        """为DataFrame添加Prophet预测特征"""
        result_dfs = []
        
        for sku_id, group in df.groupby('sku_id'):
            if sku_id in self.prophet_models:
                # 预测
                future = pd.DataFrame({'ds': group['date']})
                forecast = self.prophet_models[sku_id].predict(future)
                
                # 添加特征
                group_with_features = group.copy()
                group_with_features['prophet_trend'] = forecast['trend'].values
                group_with_features['prophet_yhat'] = forecast['yhat'].values
                
                result_dfs.append(group_with_features)
            else:
                # 没有Prophet模型的SKU
                group_with_features = group.copy()
                group_with_features['prophet_trend'] = group['discount_price']
                group_with_features['prophet_yhat'] = group['discount_price']
                result_dfs.append(group_with_features)
        
        return pd.concat(result_dfs)
    
    def _optimize_threshold(self, val_df):
        """在验证集上优化分类阈值"""
        # 预测验证集概率
        val_probs = self.lgbm_model.predict_proba(val_df[self.features])[:, 1]
        
        # 寻找满足召回率≥95%的最高阈值
        best_threshold = 0.5
        best_precision = 0
        min_recall = 0.95
        
        for threshold in np.linspace(0.01, 0.99, 99):
            val_preds = (val_probs >= threshold).astype(int)
            recall = recall_score(val_df['price_change'], val_preds)
            
            if recall >= min_recall:
                precision = precision_score(val_df['price_change'], val_preds)
                if precision > best_precision:
                    best_precision = precision
                    best_threshold = threshold
        
        logger.info(f"最优阈值: {best_threshold:.4f} (召回率≥{min_recall:.0%}, 精确度={best_precision:.4f})")
        return best_threshold
    
    def predict(self, df):
        """
        预测价格变动概率和分类结果
        
        Args:
            df: 包含预测数据的DataFrame，需包含:
                - sku_id: SKU标识
                - date: 日期
                - discount_price: 折扣价格
                
        Returns:
            Dict[str, np.ndarray]: 包含以下键的字典:
                - 'probability': 价格变动概率数组 (0-1)
                - 'predicted_change': 预测是否变动 (0或1)
        """
        if not self.fitted:
            raise RuntimeError("模型尚未训练，请先调用fit方法")
            
        # 添加Prophet特征
        df = self._add_prophet_features(df)
        
        # 确保所有特征都存在
        missing_features = set(self.features) - set(df.columns)
        for feature in missing_features:
            df[feature] = 0  # 填充默认值
            
        # 预测概率
        probs = self.lgbm_model.predict_proba(df[self.features])[:, 1]
        
        # 应用阈值生成分类结果
        preds = (probs >= self.threshold).astype(int)
        
        return {
            'probability': probs,
            'predicted_change': preds
        }
    
    def evaluate(self, test_df):
        """
        评估模型性能
        
        Args:
            test_df: 测试数据DataFrame，需包含:
                - sku_id: SKU标识
                - date: 日期
                - discount_price: 折扣价格
                
        Returns:
            Dict[str, float]: 包含评估指标的字典，包括:
                - 标准分类指标: accuracy, precision, recall, f1, auc
                - 业务指标: sampling_reduction, capture_rate
        """
        # 预处理测试数据
        test_df = self._preprocess_data(test_df)
        test_df = self._add_prophet_features(test_df)
        
        # 获取真实标签和预测结果
        y_true = test_df['price_change'].values
        predictions = self.predict(test_df)
        y_prob = predictions['probability']
        y_pred = predictions['predicted_change']
        
        # 计算标准分类指标
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred),
            'auc': roc_auc_score(y_true, y_prob),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
        }
        
        # 计算业务指标
        total_samples = len(y_true)
        true_changes = sum(y_true)
        predicted_changes = sum(y_pred)
        
        metrics.update({
            'sampling_reduction': 1 - predicted_changes/total_samples,
            'capture_rate': sum((y_true == 1) & (y_pred == 1)) / true_changes,
            'threshold': self.threshold
        })
        
        # 记录评估结果
        logger.info("模型评估结果:")
        for name, value in metrics.items():
            if name != 'confusion_matrix':
                logger.info(f"{name}: {value:.4f}")
        
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