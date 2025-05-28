import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import warnings
import logging

logger = logging.getLogger("feature_engineering")

class FeatureEngineer:
    """价格变动预测特征工程"""
    
    def __init__(self, cutoff_date: Optional[str] = None):
        """
        初始化特征工程
        
        Args:
            cutoff_date: 截止日期(YYYY-MM-DD)，确保不使用未来数据
        """
        self.cutoff_date = pd.to_datetime(cutoff_date) if cutoff_date else None
        self.feature_columns = []  # 记录生成的特征列
        
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        执行特征工程转换
        
        Args:
            df: 输入数据框，需包含:
                - sku_id: 商品ID
                - date: 日期
                - discount_price: 折扣价格
                
        Returns:
            包含所有特征的数据框
        """
        df = df.copy()
        
        # 1. 基础验证
        self._validate_input(df)
        
        # 2. 预处理
        df = self._preprocess_data(df)
        
        # 3. 生成特征
        df = self._build_time_features(df)
        df = self._build_price_features(df)
        df = self._build_trend_features(df)
        
        # 4. 后处理
        df = self._postprocess_features(df)
        
        return df
    
    def _validate_input(self, df: pd.DataFrame):
        """验证输入数据"""
        required_cols = ['sku_id', 'date', 'discount_price']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"缺少必要列: {missing_cols}")
            
        # 验证日期不超过cutoff_date
        if self.cutoff_date:
            max_date = pd.to_datetime(df['date']).max()
            if max_date > self.cutoff_date:
                raise ValueError(f"数据包含未来日期({max_date})，超过截止日期({self.cutoff_date})")
    
    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """数据预处理"""
        # 采集次数统计（在填充前完成）
        collect_counts = df.groupby(['sku_id', 'date']).size().reset_index(name='collect_count')
        df = df.merge(collect_counts, on=['sku_id', 'date'], how='left')
        # 确保日期类型
        df['date'] = pd.to_datetime(df['date'])
        # 按SKU和日期排序
        df = df.sort_values(['sku_id', 'date'])
        # 填充价格（前向优先，后向补充）
        df['discount_price'] = df.groupby('sku_id')['discount_price'].transform(lambda x: x.ffill().bfill())
        # 填充促销标志
        if 'is_promotion' in df.columns:
            df['is_promotion'] = df['is_promotion'].fillna(0)
        else:
            df['is_promotion'] = 0
        # 填充后断言
        missing = df.isnull().sum()
        logger.info(f"填充后缺失统计: {{'discount_price': {missing['discount_price']}, 'is_promotion': {missing['is_promotion']}}}")
        assert missing['discount_price'] == 0, "价格填充后仍有缺失"
        # 计算基础价格变动特征
        df['prev_price'] = df.groupby('sku_id')['discount_price'].shift(1)
        df['price_change'] = (df['discount_price'] != df['prev_price']).astype(int)
        df['price_change_amount'] = df['discount_price'] - df['prev_price']
        df['price_change_ratio'] = df['price_change_amount'] / df['prev_price']
        # 缺失变动特征填0
        df = df.fillna({
            'prev_price': df['discount_price'],
            'price_change': 0,
            'price_change_amount': 0,
            'price_change_ratio': 0
        })
        return df
    
    def _build_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """构建时间特征"""
        # 星期几 (0=周一, 6=周日)
        df['day_of_week'] = df['date'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # 月份和日
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        
        self.feature_columns.extend(['day_of_week', 'is_weekend', 'month', 'day'])
        return df
    
    def _build_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """构建价格相关特征"""
        # 滑动窗口统计
        windows = [3, 7, 14, 30]
        for w in windows:
            df[f'price_mean_{w}d'] = df.groupby('sku_id')['discount_price'].transform(
                lambda x: x.rolling(w, min_periods=1).mean()
            )
            df[f'price_std_{w}d'] = df.groupby('sku_id')['discount_price'].transform(
                lambda x: x.rolling(w, min_periods=1).std()
            )
        # 价格变动频率
        for w in [7, 30]:
            df[f'change_freq_{w}d'] = df.groupby('sku_id')['price_change'].transform(
                lambda x: x.rolling(w, min_periods=1).mean()
            )
        # 距离上次价格变动的天数
        df['days_since_last_change'] = df.groupby('sku_id').apply(
            lambda group: group['price_change'].cumsum().groupby(
                group['price_change'].cumsum()).cumcount()
        ).reset_index(level=0, drop=True)
        self.feature_columns.extend(
            [f'price_mean_{w}d' for w in windows] +
            [f'price_std_{w}d' for w in windows] +
            [f'change_freq_{w}d' for w in [7, 30]] +
            ['days_since_last_change']
        )
        # --- 防泄漏断言：每个SKU随机抽查1-2个点 ---
        sku_list = df['sku_id'].unique()
        sample_skus = np.random.choice(sku_list, min(5, len(sku_list)), replace=False)
        for sku in sample_skus:
            sku_df = df[df['sku_id'] == sku]
            if len(sku_df) == 0:
                continue
            sample_idx = np.random.choice(sku_df.index, min(2, len(sku_df)), replace=False)
            for idx in sample_idx:
                row = sku_df.loc[idx]
                for w in windows:
                    window_data = sku_df.loc[:idx, 'discount_price'].tail(w)
                    calc = window_data.mean()
                    val = row.get(f'price_mean_{w}d', None)
                    if val is not None and not np.isclose(val, calc, atol=1e-6):
                        logger.error(f"滑窗特征泄漏: sku={sku}, idx={idx}, window={w}, val={val}, calc={calc}")
                        raise AssertionError("滑窗特征泄漏未来数据")
        logger.info("滑窗防泄漏检查通过")
        return df
    
    def _build_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """构建趋势特征"""
        # 价格趋势 (线性回归斜率)
        for w in [7, 14]:
            df[f'price_trend_{w}d'] = df.groupby('sku_id')['discount_price'].transform(
                lambda x: x.rolling(w, min_periods=1).apply(
                    lambda y: np.polyfit(range(len(y)), y, 1)[0] if len(y) > 1 else 0
                )
            )
            
        self.feature_columns.extend([f'price_trend_{w}d' for w in [7, 14]])
        # 合并 price_direction_consistency_7d 特征
        if 'price_change_direction' in df.columns:
            df['price_direction_consistency_7d'] = (
                df.groupby('sku_id')['price_change_direction']
                .transform(lambda x: x.rolling(7, min_periods=1).apply(
                    lambda y: (y == y.iloc[0]).mean() if len(y) > 0 else 0
                ))
            )
            self.feature_columns.append('price_direction_consistency_7d')
        return df
    
    def _postprocess_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """特征后处理"""
        # 确保所有特征列都存在
        missing_features = set(self.feature_columns) - set(df.columns)
        for f in missing_features:
            df[f] = 0
            warnings.warn(f"自动填充缺失特征: {f}")
            
        # 填充剩余NA
        df = df.fillna(0)
        
        return df
    
    def get_feature_names(self) -> List[str]:
        """获取所有特征列名"""
        return self.feature_columns
        
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