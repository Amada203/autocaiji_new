import pandas as pd
from datetime import datetime

class DataProcessor:
    def __init__(self, raw_df):
        """初始化数据处理器"""
        self.df = raw_df.copy()
        
    def process(self):
        """执行完整的数据处理流程"""
        try:
            # 1. 数据清洗
            self._clean_data()
            
            # 2. 特征工程
            self._feature_engineering()
            
            # 3. 计算价格变动
            self._calculate_price_changes()
            
            return self.df
            
        except Exception as e:
            print(f"数据处理失败: {str(e)}")
            return pd.DataFrame()
    
    def _clean_data(self):
        """数据清洗"""
        # 去除空值
        self.df = self.df.dropna(subset=['page_price', 'dt'])
        
        # 转换数据类型
        self.df['dt'] = pd.to_datetime(self.df['dt'])
        self.df['page_price'] = self.df['page_price'].astype(float)
        
    def _feature_engineering(self):
        """特征工程"""
        # 添加时间特征
        self.df['day_of_week'] = self.df['dt'].dt.dayofweek
        self.df['month'] = self.df['dt'].dt.month
        
        # 添加价格差异特征
        self.df['price_diff'] = self.df['page_price'] - self.df['discount_price']
        
    def _calculate_price_changes(self):
        """计算价格变动"""
        # 按SKU分组计算价格变动
        self.df = self.df.sort_values(['sku_id', 'dt'])
        self.df['price_change'] = self.df.groupby('sku_id')['page_price'].pct_change()
        
        # 标记显著变动(>5%)
        self.df['significant_change'] = (self.df['price_change'].abs() > 0.05).astype(int)