import pandas as pd
import numpy as np
from datetime import datetime
import logging

class DataProcessor:
    def __init__(self, raw_df):
        """初始化数据处理器"""
        self.df = raw_df.copy()
        self.logger = logging.getLogger(__name__)
        
    def process(self):
        """执行完整的数据处理流程"""
        try:
            # 验证输入数据
            if not self._validate_input():
                self.logger.error("输入数据验证失败")
                return pd.DataFrame()
                
            self.logger.info(f"开始处理数据，初始形状: {self.df.shape}")
            
            # 1. 数据清洗
            self._clean_data()
            self.logger.info(f"数据清洗后形状: {self.df.shape}")
            
            # 2. 特征工程
            self._feature_engineering()
            self.logger.info(f"特征工程后形状: {self.df.shape}")
            
            # 3. 计算价格变动
            self._calculate_price_changes()
            self.logger.info(f"最终处理完成，数据形状: {self.df.shape}")
            
            return self.df
            
        except Exception as e:
            self.logger.error(f"数据处理失败: {str(e)}", exc_info=True)
            # 返回已处理的部分数据而非空DataFrame
            return self.df if hasattr(self, 'df') else pd.DataFrame()
    
    def _validate_input(self):
        """验证输入数据"""
        if self.df is None or self.df.empty:
            self.logger.error("输入数据为空")
            raise ValueError("输入数据为空")
            
        required_columns = ['sku_id', 'dt', 'page_price', 'discount_price']
        missing_cols = [col for col in required_columns if col not in self.df.columns]
        
        if missing_cols:
            self.logger.error(f"缺少必要列: {missing_cols}")
            raise ValueError(f"缺少必要列: {missing_cols}")
            
        return True
    
    def _clean_data(self):
        """数据清洗"""
        # 记录原始行数
        original_rows = len(self.df)
        
        # 转换数据类型（更安全的转换方式）
        self.df['dt'] = pd.to_datetime(self.df['dt'], errors='coerce')
        self.df['page_price'] = pd.to_numeric(self.df['page_price'], errors='coerce')
        self.df['discount_price'] = pd.to_numeric(self.df['discount_price'], errors='coerce')
        
        # 填充空值而不是直接删除
        price_median = self.df['page_price'].median()
        discount_median = self.df['discount_price'].median()
        
        self.df['page_price'] = self.df['page_price'].fillna(price_median)
        self.df['discount_price'] = self.df['discount_price'].fillna(discount_median)
        self.df['dt'] = self.df['dt'].fillna(pd.to_datetime('today'))
        
        # 只删除dt和page_price都为空的记录
        self.df = self.df.dropna(subset=['dt', 'page_price'], how='all')
        
        # 记录处理情况
        removed_rows = original_rows - len(self.df)
        if removed_rows > 0:
            self.logger.warning(f"移除了 {removed_rows} 条无效记录")
        
    def _feature_engineering(self):
        """特征工程"""
        # 添加时间特征
        self.df['day_of_week'] = self.df['dt'].dt.dayofweek
        self.df['month'] = self.df['dt'].dt.month
        
        # 添加价格差异特征（处理可能的除零情况）
        with np.errstate(divide='ignore', invalid='ignore'):
            price_diff = self.df['page_price'] - self.df['discount_price']
            self.df['price_diff'] = np.where(
                (self.df['page_price'] > 0) & (self.df['discount_price'] > 0),
                price_diff,
                np.nan
            )
        
    def _calculate_price_changes(self):
        """计算价格变动"""
        if len(self.df) == 0:
            self.logger.warning("无有效数据可计算价格变动")
            return
            
        # 按SKU分组计算价格变动
        self.df = self.df.sort_values(['sku_id', 'dt'])
        
        # 更安全的pct_change计算
        self.df['price_change'] = self.df.groupby('sku_id')['page_price'].apply(
            lambda x: x.pct_change().fillna(0)
        )
        
        # 标记显著变动(>5%)
        self.df['significant_change'] = (self.df['price_change'].abs() > 0.05).astype(int)