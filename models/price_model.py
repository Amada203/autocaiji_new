"""
价格变化概率预测模型
基于Prophet捕获时间模式，LightGBM预测价格变化概率
"""
import pandas as pd
import numpy as np
from prophet import Prophet
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score
import logging
import os
import pickle

logger = logging.getLogger(__name__)

class PriceModel:
    """
    价格变化概率预测模型
    利用Prophet捕获时间序列的趋势和季节性，然后使用LightGBM预测变化概率
    """
    
    def __init__(self, prophet_params=None, lgbm_params=None, change_threshold=0.01):
        """
        初始化模型
        
        Args:
            prophet_params: Prophet模型参数字典
            lgbm_params: LightGBM模型参数字典
            change_threshold: 价格变化阈值（相对变化比例），超过该阈值认为价格发生变化
        """
        self.prophet_params = prophet_params or {}
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
    
    def fit(self, data, y=None):
        """
        训练模型
        
        Args:
            data: 包含'ds'、'y'和'sku_id'列的DataFrame，或者包含特征的DataFrame和目标变量y
            y: 目标变量（价格变化标志），当data不包含时使用
            
        Returns:
            self
        """
        logger.info("开始训练价格变化概率预测模型")
        
        # 处理输入格式
        if y is not None:
            # 特征矩阵和目标变量分开传入的情况
            if 'ds' not in data.columns:
                raise ValueError("数据必须包含'ds'列")
            train_df = data.copy()
            train_df['change_flag'] = y
        else:
            # ds和y在同一个DataFrame的情况
            if 'ds' not in data.columns or 'y' not in data.columns:
                raise ValueError("数据必须包含'ds'和'y'列")
            train_df = data.copy()
            
            # 计算价格变化标志
            if 'change_flag' not in train_df.columns:
                self._calculate_change_flags(train_df)
        
        # 确保日期格式正确
        train_df['ds'] = pd.to_datetime(train_df['ds'])
        
        # 1. 训练Prophet模型捕获时间模式
        logger.info("训练Prophet模型捕获时间模式...")
        self.prophet_model.fit(train_df[['ds', 'y']])
        
        # 2. 生成Prophet特征
        prophet_features = self._extract_prophet_features(train_df)
        
        # 3. 构建完整特征集
        X_train = self._build_features(train_df, prophet_features)
        self.feature_names = X_train.columns.tolist()
        
        # 4. 训练LightGBM模型预测价格变化概率
        logger.info("训练LightGBM模型预测价格变化概率...")
        self.lgbm_model = lgb.LGBMClassifier(**self.lgbm_params)
        self.lgbm_model.fit(X_train, train_df['change_flag'])
        
        self.fitted = True
        logger.info("价格变化概率预测模型训练完成")
        return self
    
    def predict(self, data):
        """
        预测价格变化标志（0/1）
        
        Args:
            data: 包含'ds'列的DataFrame，表示要预测的时间点
            
        Returns:
            变化标志(0/1)的numpy数组
        """
        if not self.fitted:
            raise ValueError("模型尚未训练，请先调用fit方法")
            
        # 获取预测概率
        proba_df = self.predict_probability(data)
        
        # 使用0.5作为默认阈值
        threshold = 0.5
        predictions = (proba_df['change_probability'] >= threshold).astype(int).values
        
        return predictions
    
    def predict_probability(self, future_df):
        """
        预测价格变化概率
        
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
        
        # 1. 提取Prophet特征
        prophet_features = self._extract_prophet_features(future_df)
        
        # 2. 构建完整特征集
        X_future = self._build_features(future_df, prophet_features)
        
        # 确保特征列匹配
        missing_cols = set(self.feature_names) - set(X_future.columns)
        for col in missing_cols:
            X_future[col] = 0
        X_future = X_future[self.feature_names]
        
        # 3. 预测变化概率
        change_probability = self.lgbm_model.predict_proba(X_future)[:, 1]
        
        # 4. 构建结果DataFrame
        result = future_df[['ds']].copy()
        if 'sku_id' in future_df.columns:
            result['sku_id'] = future_df['sku_id']
        result['change_probability'] = change_probability
        
        return result
    
    def evaluate(self, test_df, y=None):
        """
        评估模型
        
        Args:
            test_df: 包含'ds'和'y'列的测试数据，或者包含特征的DataFrame
            y: 当test_df不包含'change_flag'列时的目标变量
            
        Returns:
            包含评估指标的字典
        """
        # 处理输入格式
        if y is not None:
            if 'ds' not in test_df.columns:
                raise ValueError("测试数据必须包含'ds'列")
            eval_df = test_df.copy()
            eval_df['change_flag'] = y
        else:
            if 'ds' not in test_df.columns or 'y' not in test_df.columns:
                raise ValueError("测试数据必须包含'ds'和'y'列")
            eval_df = test_df.copy()
            
            # 计算价格变化标志
            if 'change_flag' not in eval_df.columns:
                self._calculate_change_flags(eval_df)
        
        # 生成预测
        predictions = self.predict_probability(eval_df)
        
        # 计算指标
        y_true = eval_df['change_flag'].values
        y_prob = predictions['change_probability'].values
        
        # 计算AUC
        auc = roc_auc_score(y_true, y_prob)
        
        # 计算PR曲线下面积
        ap = average_precision_score(y_true, y_prob)
        
        # 找到最佳阈值（根据F1分数）
        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
        f1_scores = 2 * recall * precision / (recall + precision + 1e-10)
        best_threshold = thresholds[np.argmax(f1_scores[:-1])]
        
        # 计算最佳阈值下的捕获率（召回率）
        y_pred = (y_prob >= best_threshold).astype(int)
        capture_rate = np.sum((y_true == 1) & (y_pred == 1)) / np.sum(y_true == 1)
        
        metrics = {
            'auc': auc,
            'average_precision': ap,
            'best_threshold': best_threshold,
            'capture_rate': capture_rate
        }
        
        return metrics
    
    def _calculate_change_flags(self, df):
        """
        计算价格变化标志
        
        Args:
            df: 包含价格数据的DataFrame
            
        Returns:
            添加change_flag列的DataFrame
        """
        # 按SKU分组计算前一日价格（如果有sku_id列）
        if 'sku_id' in df.columns:
            df['prev_y'] = df.groupby('sku_id')['y'].shift(1)
        else:
            df['prev_y'] = df['y'].shift(1)
        
        # 计算相对变化比例（保留用于其他可能的分析）
        df['change_ratio'] = (df['y'] - df['prev_y']) / (df['prev_y'] + 1e-10)
        
        # 修改为严格不等判定
        df['change_flag'] = (df['y'] != df['prev_y']).astype(int)
        
        # 第一个数据点没有前值，设为0
        df['change_flag'] = df['change_flag'].fillna(0)
        
        return df
    
    def _extract_prophet_features(self, data):
        """
        提取Prophet模型特征
        
        Args:
            data: 包含时间戳的DataFrame
            
        Returns:
            Prophet特征DataFrame
        """
        # 使用Prophet模型进行预测，提取组件特征
        prophet_forecast = self.prophet_model.predict(data[['ds']])
        
        # 提取关键组件
        features = prophet_forecast[['ds', 'trend', 'yearly', 'weekly']]
        
        # 添加节假日效应（如果存在）
        if 'holidays' in prophet_forecast.columns:
            features['holiday_effect'] = prophet_forecast['holidays']
        
        return features
    
    def _build_features(self, data, prophet_features):
        """
        构建完整特征集
        
        Args:
            data: 原始数据
            prophet_features: Prophet提取的特征
            
        Returns:
            特征DataFrame
        """
        # 合并Prophet特征
        features = pd.merge(data, prophet_features, on='ds')
        
        # 添加时间特征
        features['day_of_week'] = features['ds'].dt.dayofweek
        features['month'] = features['ds'].dt.month
        features['day'] = features['ds'].dt.day
        features['quarter'] = features['ds'].dt.quarter
        features['is_weekend'] = (features['day_of_week'] >= 5).astype(int)
        features['is_month_start'] = features['ds'].dt.is_month_start.astype(int)
        features['is_month_end'] = features['ds'].dt.is_month_end.astype(int)
        
        # 计算历史变化频率特征（如果有足够数据）
        if 'change_flag' in data.columns and len(data) > 10:
            if 'sku_id' in data.columns:
                # 按SKU计算滚动变化频率
                for window in [7, 14, 30]:
                    features[f'change_freq_{window}d'] = features.groupby('sku_id')['change_flag'].transform(
                        lambda x: x.rolling(window, min_periods=1).mean()
                    )
            else:
                # 单一时间序列
                for window in [7, 14, 30]:
                    features[f'change_freq_{window}d'] = features['change_flag'].rolling(
                        window, min_periods=1
                    ).mean()
        
        # 如果有价格数据，计算价格波动特征
        if 'y' in data.columns:
            if 'sku_id' in data.columns:
                # 按SKU计算
                features['price_volatility_7d'] = features.groupby('sku_id')['y'].transform(
                    lambda x: x.rolling(7, min_periods=1).std() / (x.rolling(7, min_periods=1).mean() + 1e-10)
                )
            else:
                # 单一时间序列
                features['price_volatility_7d'] = features['y'].rolling(7, min_periods=1).std() / (
                    features['y'].rolling(7, min_periods=1).mean() + 1e-10
                )
        
        # 移除非特征列
        drop_cols = ['ds', 'y', 'prev_y', 'change_ratio', 'change_flag']
        feature_cols = [col for col in features.columns if col not in drop_cols]
        
        return features[feature_cols]
    
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
            'change_threshold': self.change_threshold,
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
        self.change_threshold = model_dict['change_threshold']
        self.fitted = model_dict['fitted']
        
        logger.info(f"模型已从 {path} 加载")
        return self 