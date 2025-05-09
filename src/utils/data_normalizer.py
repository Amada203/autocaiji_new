"""
数据标准化和归一化工具类
提供多种数据标准化和归一化方法，支持异常值处理。
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import logging

logger = logging.getLogger(__name__)

class DataNormalizer:
    """数据标准化和归一化工具类"""
    
    def __init__(self, method='z-score', params=None):
        """
        初始化数据标准化器
        
        Args:
            method: 标准化方法, 'z-score', 'minmax', 或 'combined'
            params: 额外参数，例如minmax的特征范围
        """
        self.method = method
        self.params = params or {}
        self.scaler = None
        self._fitted = False
        
    def fit_transform(self, data, cols=None):
        """
        拟合并转换数据
        
        Args:
            data: DataFrame或numpy数组
            cols: 要处理的列名列表，仅在data为DataFrame时有效
            
        Returns:
            标准化/归一化后的数据
        """
        if self.method not in ['z-score', 'minmax', 'combined']:
            raise ValueError(f"不支持的标准化方法: {self.method}, 请使用'z-score', 'minmax', 或 'combined'")
            
        # 处理DataFrame的情况
        if isinstance(data, pd.DataFrame):
            if cols is None:
                cols = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
            
            data_copy = data.copy()
            if len(cols) == 0:
                logger.warning("没有找到数值列，返回原始数据")
                return data_copy
                
            # 提取要处理的数据
            data_to_transform = data_copy[cols].values
            
            # 标准化/归一化
            transformed_data = self._transform_array(data_to_transform)
            
            # 将结果放回DataFrame
            data_copy[cols] = transformed_data
            result = data_copy
        else:
            # 数组直接处理
            result = self._transform_array(data)
            
        return result
    
    def _transform_array(self, data_array):
        """
        转换数组数据
        
        Args:
            data_array: numpy数组
            
        Returns:
            转换后的数组
        """
        if self.method == 'z-score':
            self.scaler = StandardScaler()
            transformed = self.scaler.fit_transform(data_array)
            
        elif self.method == 'minmax':
            feature_range = self.params.get('feature_range', (0, 1))
            self.scaler = MinMaxScaler(feature_range=feature_range)
            transformed = self.scaler.fit_transform(data_array)
            
        elif self.method == 'combined':
            # 先标准化再归一化
            std_scaler = StandardScaler()
            minmax_scaler = MinMaxScaler(feature_range=self.params.get('feature_range', (0, 1)))
            
            interim = std_scaler.fit_transform(data_array)
            transformed = minmax_scaler.fit_transform(interim)
            
            self.scaler = {'standard': std_scaler, 'minmax': minmax_scaler}
            
        self._fitted = True
        return transformed
    
    def inverse_transform(self, normalized_data, cols=None):
        """
        将标准化/归一化后的数据转换回原始尺度
        
        Args:
            normalized_data: 标准化后的数据
            cols: 要处理的列名列表，仅在数据为DataFrame时有效
            
        Returns:
            转换回原始尺度的数据
        """
        if not self._fitted:
            raise ValueError("请先调用fit_transform方法")
            
        # 处理DataFrame的情况
        if isinstance(normalized_data, pd.DataFrame):
            if cols is None:
                cols = normalized_data.select_dtypes(include=['float64', 'int64']).columns.tolist()
            
            data_copy = normalized_data.copy()
            if len(cols) == 0:
                logger.warning("没有找到数值列，返回原始数据")
                return data_copy
                
            # 提取要处理的数据
            data_to_inverse = data_copy[cols].values
            
            # 转换回原始尺度
            inverse_data = self._inverse_transform_array(data_to_inverse)
            
            # 将结果放回DataFrame
            data_copy[cols] = inverse_data
            result = data_copy
        else:
            # 数组直接处理
            result = self._inverse_transform_array(normalized_data)
            
        return result
    
    def _inverse_transform_array(self, data_array):
        """
        将数组转换回原始尺度
        
        Args:
            data_array: numpy数组
            
        Returns:
            转换回原始尺度的数组
        """
        if self.method in ['z-score', 'minmax']:
            return self.scaler.inverse_transform(data_array)
            
        elif self.method == 'combined':
            # 先反归一化再反标准化
            interim = self.scaler['minmax'].inverse_transform(data_array)
            return self.scaler['standard'].inverse_transform(interim)
            
    def detect_and_handle_outliers(self, data, cols=None, method='zscore', threshold=3):
        """
        检测并处理异常值
        
        Args:
            data: DataFrame或numpy数组
            cols: 要处理的列名列表，仅在data为DataFrame时有效
            method: 检测方法，'zscore'或'iqr'
            threshold: 阈值，zscore方法下默认为3，iqr方法下默认为1.5
            
        Returns:
            处理后的数据
        """
        # 处理DataFrame的情况
        if isinstance(data, pd.DataFrame):
            if cols is None:
                cols = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
            
            data_copy = data.copy()
            if len(cols) == 0:
                logger.warning("没有找到数值列，返回原始数据")
                return data_copy
                
            # 对每列处理异常值
            for col in cols:
                data_copy[col] = self._handle_outliers_series(data_copy[col], method, threshold)
                
            return data_copy
        else:
            # 暂不支持直接处理numpy数组
            raise ValueError("异常值处理功能仅支持DataFrame")
    
    def _handle_outliers_series(self, series, method, threshold):
        """
        处理Series中的异常值
        
        Args:
            series: pandas Series
            method: 检测方法
            threshold: 阈值
            
        Returns:
            处理后的Series
        """
        series_copy = series.copy()
        
        if method == 'zscore':
            # Z-score方法
            z_scores = np.abs((series_copy - series_copy.mean()) / series_copy.std())
            outliers = z_scores > threshold
            # 将异常值替换为边界值
            series_copy[outliers] = series_copy.mean() + threshold * series_copy.std() * np.sign(series_copy[outliers] - series_copy.mean())
            
        elif method == 'iqr':
            # IQR方法
            Q1 = series_copy.quantile(0.25)
            Q3 = series_copy.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            # 截断超出边界的值
            series_copy = series_copy.clip(lower=lower_bound, upper=upper_bound)
            
        else:
            raise ValueError(f"不支持的异常值检测方法: {method}")
            
        return series_copy 