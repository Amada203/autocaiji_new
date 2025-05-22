import pandas as pd
import numpy as np
from datetime import datetime
import logging

class FeatureEngineer:
    def __init__(self, logger=None):
        """初始化特征工程师"""
        self.logger = logger or logging.getLogger(__name__)
    
    def create_features(self, df):
        """
        生成所有特征并检查防泄漏
        参数:
            df: 经过预处理的DataFrame
        返回:
            包含所有特征的DataFrame
        """
        try:
            self._check_input(df)
            self._check_no_future_leakage(df)
            
            # 1. 时间特征
            df = self._add_time_features(df)
            
            # 2. 滑动窗口特征
            df = self._add_window_features(df)
            
            # 3. 价格变动模式特征
            df = self._add_change_pattern_features(df)
            
            return df
        except Exception as e:
            self.logger.error(f"特征工程失败: {str(e)}", exc_info=True)
            raise
    
    def _check_input(self, df):
        """验证输入数据格式"""
        required_cols = {'sku_id', 'date', 'discount_price', 'price_change'}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"输入数据必须包含列: {required_cols}")
        if df.empty:
            raise ValueError("输入数据不能为空")
    
    def _check_no_future_leakage(self, df):
        """严格检查防泄漏"""
        max_date = df['date'].max()
        if not df.groupby('sku_id')['date'].transform('max').le(max_date).all():
            self.logger.error("检测到数据泄漏!")
            raise ValueError("存在数据泄漏: 使用了未来信息")
    
    def _add_time_features(self, df):
        """添加时间相关特征"""
        df = df.copy()
        df['day_of_week'] = df['date'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['quarter'] = df['date'].dt.quarter
        df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
        df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
        return df
    
    def _add_window_features(self, df):
        """添加滑动窗口统计特征"""
        df = df.copy()
        for window in [3, 7, 14, 30]:
            # 价格统计特征
            df[f'price_mean_{window}d'] = (df.groupby('sku_id')['discount_price']
                                         .transform(lambda x: x.rolling(window, min_periods=1).mean()))
            df[f'price_std_{window}d'] = (df.groupby('sku_id')['discount_price']
                                        .transform(lambda x: x.rolling(window, min_periods=1).std()))
            
            # 价格变动频率特征
            df[f'change_freq_{window}d'] = (df.groupby('sku_id')['price_change']
                                         .transform(lambda x: x.rolling(window, min_periods=1).mean()))
        
        return df
    
    def _add_change_pattern_features(self, df):
        """添加价格变动模式特征"""
        df = df.copy()
        # 距离上次变动的天数
        df['days_since_last_change'] = (df.groupby('sku_id')['price_change']
                                      .transform(lambda x: x.cumsum().groupby(x.cumsum()).cumcount()))
        
        # 价格趋势特征
        for window in [7, 14]:
            df[f'price_trend_{window}d'] = (df.groupby('sku_id')['discount_price']
                                         .transform(lambda x: x.rolling(window).apply(
                                             lambda y: np.polyfit(range(len(y)), y, 1)[0] if len(y) > 1 else 0)))
        
        # 价格变动方向一致性
        df['price_direction_consistency_7d'] = (df.groupby('sku_id')['price_change_direction']
                                              .transform(lambda x: x.rolling(7).apply(
                                                  lambda y: (y == y.iloc[0]).mean() if len(y) > 0 else 0)))
        
        return df