import pandas as pd
import numpy as np
from typing import List, Optional
from datetime import datetime, timedelta

class FeatureEngineering:
    def __init__(self, df: pd.DataFrame):
        """
        初始化特征工程类
        
        Args:
            df: 预处理后的数据框
        """
        self.df = df.copy()
        
        # 处理日期列（支持ds或date列名）
        if 'ds' in self.df.columns:
            try:
                self.df['date'] = pd.to_datetime(self.df['ds'])
            except Exception as e:
                raise ValueError(f"无法将ds列转换为日期类型: {str(e)}")
        elif 'date' not in self.df.columns:
            raise ValueError("输入数据必须包含日期列（'ds'或'date'）")
        else:
            try:
                self.df['date'] = pd.to_datetime(self.df['date'])
            except Exception as e:
                raise ValueError(f"无法将date列转换为日期类型: {str(e)}")
            
        # 重命名y列为price（如果存在）
        if 'y' in self.df.columns and 'price' not in self.df.columns:
            self.df['price'] = self.df['y']
            
        # 计算价格变化相关指标（如果不存在）
        if 'price' in self.df.columns:
            # 基本价格变化指标
            if 'price_change_flag' not in self.df.columns:
                self.df['price_change_flag'] = (self.df.groupby('sku_id')['price'].diff() != 0).astype(int)
            
            if 'price_change_amount' not in self.df.columns:
                self.df['price_change_amount'] = self.df.groupby('sku_id')['price'].diff()
            
            if 'price_change_ratio' not in self.df.columns:
                self.df['price_change_ratio'] = self.df.groupby('sku_id')['price'].pct_change()
                
            # 价格变化方向
            self.df['price_change_direction'] = np.where(self.df['price_change_amount'] > 0, 1,
                                                       np.where(self.df['price_change_amount'] < 0, -1, 0))
            
            # 价格变化类型
            self.df['price_change_type'] = np.where(self.df['price_change_flag'] == 0, 'no_change',
                                                   np.where(self.df['price_change_direction'] > 0, 'increase', 'decrease'))
        
    def build_time_features(self) -> pd.DataFrame:
        """
        构建时间特征
        
        Returns:
            pd.DataFrame: 添加时间特征后的数据框
        """
        # 星期几
        self.df['weekday'] = self.df['date'].dt.weekday
        
        # 是否周末
        self.df['is_weekend'] = self.df['weekday'].isin([5, 6]).astype(int)
        
        # 是否月初（1-3号）
        self.df['is_month_start'] = self.df['date'].dt.day.isin([1, 2, 3]).astype(int)
        
        # 是否月末（28-31号）
        self.df['is_month_end'] = self.df['date'].dt.day.isin([28, 29, 30, 31]).astype(int)
        
        return self.df
    
    def build_price_trend_features(self, window_sizes: List[int] = [3, 7, 14, 30]) -> pd.DataFrame:
        """
        构建价格趋势特征
        
        Args:
            window_sizes: 滑动窗口大小列表
            
        Returns:
            pd.DataFrame: 添加价格趋势特征后的数据框
        """
        for window in window_sizes:
            # 计算前N天价格变动频率
            self.df[f'price_change_freq_{window}d'] = self.df.groupby('sku_id')['price_change_flag'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
            
            # 计算前N天价格变动幅度
            self.df[f'price_change_amount_{window}d'] = self.df.groupby('sku_id')['price_change_amount'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
            
            # 计算前N天价格变动比例
            self.df[f'price_change_ratio_{window}d'] = self.df.groupby('sku_id')['price_change_ratio'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
            
        return self.df
    
    def build_product_features(self) -> pd.DataFrame:
        """
        构建商品特征
        
        Returns:
            pd.DataFrame: 添加商品特征后的数据框
        """
        # 计算商品价格区间
        self.df['price_range'] = pd.qcut(self.df['price'], q=5, labels=['very_low', 'low', 'medium', 'high', 'very_high'])
        
        # 计算商品价格稳定性（使用价格标准差）
        self.df['price_stability'] = self.df.groupby('sku_id')['price'].transform(
            lambda x: x.rolling(window=30, min_periods=1).std()
        )
        
        # 计算商品价格变动频率
        self.df['price_change_frequency'] = self.df.groupby('sku_id')['price_change_flag'].transform('mean')
        
        return self.df
    
    def build_cross_features(self) -> pd.DataFrame:
        """
        构建交叉特征
        
        Returns:
            pd.DataFrame: 添加交叉特征后的数据框
        """
        # 价格区间与时间特征的交叉
        self.df['price_range_weekend'] = self.df['price_range'].astype(str) + '_' + self.df['is_weekend'].astype(str)
        
        # 价格变动方向与时间特征的交叉
        self.df['price_direction_weekend'] = self.df['price_change_direction'].astype(str) + '_' + self.df['is_weekend'].astype(str)
        
        # 价格变动类型与时间特征的交叉
        self.df['price_type_weekend'] = self.df['price_change_type'] + '_' + self.df['is_weekend'].astype(str)
        
        return self.df
    
    def process(self) -> pd.DataFrame:
        """
        处理数据并构建所有特征
        
        Returns:
            pd.DataFrame: 处理后的数据框，包含所有特征
        """
        try:
            # 构建所有特征
            self.build_time_features()
            self.build_price_trend_features()
            self.build_product_features()
            self.build_cross_features()
            
            # 移除包含空值的行
            self.df = self.df.dropna()
            
            # 确保数据类型正确
            self._ensure_data_types()
            
            return self.df
            
        except Exception as e:
            raise RuntimeError(f"特征工程处理失败: {str(e)}")
            
    def _ensure_data_types(self):
        """确保数据类型正确"""
        # 数值型特征
        numeric_features = [
            'price_change_freq_3d', 'price_change_freq_7d', 'price_change_freq_14d', 'price_change_freq_30d',
            'price_change_amount_3d', 'price_change_amount_7d', 'price_change_amount_14d', 'price_change_amount_30d',
            'price_change_ratio_3d', 'price_change_ratio_7d', 'price_change_ratio_14d', 'price_change_ratio_30d',
            'price_stability', 'price_change_frequency'
        ]
        
        # 类别型特征
        categorical_features = [
            'weekday', 'is_weekend', 'is_month_start', 'is_month_end',
            'price_range', 'price_range_weekend', 'price_direction_weekend', 'price_type_weekend'
        ]
        
        # 转换数值型特征
        for col in numeric_features:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
                
        # 转换类别型特征
        for col in categorical_features:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype('category')
    
    def get_feature_columns(self) -> List[str]:
        """
        获取所有特征列名
        
        Returns:
            List[str]: 特征列名列表
        """
        # 排除目标变量和ID列
        exclude_cols = ['sku_id', 'date', 'price', 'prev_price', 'price_change_flag']
        return [col for col in self.df.columns if col not in exclude_cols] 