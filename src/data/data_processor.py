import pandas as pd
import numpy as np
from typing import Optional, Tuple
from datetime import datetime, timedelta

class DataProcessor:
    def __init__(self, data_path: str):
        """
        初始化数据处理器
        
        Args:
            data_path: 原始数据文件路径
        """
        self.data_path = data_path
        self.df = None
        
    def load_data(self) -> pd.DataFrame:
        """
        加载原始数据
        
        Returns:
            pd.DataFrame: 加载的数据
        """
        self.df = pd.read_csv(self.data_path)
        return self.df
    
    def preprocess_data(self) -> pd.DataFrame:
        """
        数据预处理
        
        Returns:
            pd.DataFrame: 预处理后的数据
        """
        if self.df is None:
            self.load_data()
            
        # 确保日期列格式正确
        self.df['date'] = pd.to_datetime(self.df['date'])
        
        # 按SKU和日期排序
        self.df = self.df.sort_values(['sku_id', 'date'])
        
        # 计算价格变动相关特征
        self._calculate_price_changes()
        
        return self.df
    
    def _calculate_price_changes(self):
        """
        计算价格变动相关特征
        """
        # 按SKU分组计算前一日价格
        self.df['prev_price'] = self.df.groupby('sku_id')['price'].shift(1)
        
        # 计算价格变动标志
        self.df['price_change_flag'] = (self.df['price'] != self.df['prev_price']).astype(int)
        
        # 计算价格变动类型
        self.df['price_change_type'] = 'unchanged'
        self.df.loc[self.df['price'] > self.df['prev_price'], 'price_change_type'] = 'up'
        self.df.loc[self.df['price'] < self.df['prev_price'], 'price_change_type'] = 'down'
        
        # 计算价格变动金额
        self.df['price_change_amount'] = self.df['price'] - self.df['prev_price']
        
        # 计算价格变动比例
        self.df['price_change_ratio'] = self.df['price_change_amount'] / self.df['prev_price']
        
        # 计算价格变动方向
        self.df['price_change_direction'] = 0
        self.df.loc[self.df['price'] > self.df['prev_price'], 'price_change_direction'] = 1
        self.df.loc[self.df['price'] < self.df['prev_price'], 'price_change_direction'] = -1
        
        # 计算连续变动天数
        self.df['price_change_streak'] = self.df.groupby('sku_id')['price_change_flag'].transform(
            lambda x: x.groupby((x != x.shift()).cumsum()).cumsum()
        )
        
    def split_data(self, test_size: float = 0.15, val_size: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        划分训练集、验证集和测试集
        
        Args:
            test_size: 测试集比例
            val_size: 验证集比例
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: 训练集、验证集和测试集
        """
        if self.df is None:
            self.preprocess_data()
            
        # 按时间顺序划分数据
        total_size = len(self.df)
        test_idx = int(total_size * (1 - test_size))
        val_idx = int(test_idx * (1 - val_size))
        
        train_df = self.df.iloc[:val_idx]
        val_df = self.df.iloc[val_idx:test_idx]
        test_df = self.df.iloc[test_idx:]
        
        return train_df, val_df, test_df
    
    def save_processed_data(self, output_path: str):
        """
        保存处理后的数据
        
        Args:
            output_path: 输出文件路径
        """
        if self.df is None:
            self.preprocess_data()
            
        self.df.to_csv(output_path, index=False) 