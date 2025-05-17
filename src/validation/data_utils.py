"""
数据处理工具模块
"""
import pandas as pd
import random
from src.data.data_fetcher import DataFetcher

def prepare_test_data():
    """准备测试数据集（根据原始计划）
    
    Returns:
        tuple: (train_data, test_data) 训练集和测试集
    """
    # 1. 获取完整数据集
    fetcher = DataFetcher()
    full_data = fetcher.fetch_training_data()
    
    # 2. 确保数据包含必要字段（接受ds或date作为日期列）
    required_cols = ['y', 'sku_id']
    optional_cols = ['is_promotion']
    date_col = 'ds' if 'ds' in full_data.columns else 'date'
    
    if date_col not in full_data.columns:
        raise ValueError("输入数据缺少日期列（需要'ds'或'date'）")
    required_cols.append(date_col)
    
    # 检查必要列
    missing_cols = [col for col in required_cols if col not in full_data.columns]
    if missing_cols:
        raise ValueError(f"输入数据缺少必要列: {', '.join(missing_cols)}")
        
    # 检查可选列
    for col in optional_cols:
        if col not in full_data.columns:
            logger.warning(f"可选列 {col} 不存在，模型可能无法使用该特征")
    
    # 3. 按原始计划划分数据集
    full_data[date_col] = pd.to_datetime(full_data[date_col])
    train_data = full_data[full_data[date_col] <= '2024-12-31']
    test_data = full_data[full_data[date_col] > '2024-12-31']
    
    # 4. 计算变化标志（如果不存在）
    if 'change_flag' not in train_data.columns:
        train_data = _calculate_change_flags(train_data)
        test_data = _calculate_change_flags(test_data)
    
    # 重命名日期列为ds
    train_data = train_data.rename(columns={'date': 'ds'})
    test_data = test_data.rename(columns={'date': 'ds'})
    
    return train_data, test_data

def _calculate_change_flags(df):
    """计算价格变化标志（辅助函数）"""
    # 按SKU分组计算前一日价格
    if 'sku_id' in df.columns:
        df['prev_y'] = df.groupby('sku_id')['y'].shift(1)
    else:
        df['prev_y'] = df['y'].shift(1)
    
    # 计算变化标志
    df['change_flag'] = (df['y'] != df['prev_y']).astype(int)
    df['change_flag'] = df['change_flag'].fillna(0)
    
    return df

def split_for_validation(data):
    """划分验证集
    
    Args:
        data: DataFrame 输入数据集
        
    Returns:
        tuple: (train_data, val_data) 训练集和验证集
    """
    # 随机选择20%的SKU作为验证集
    all_skus = list(data['sku_id'].unique())
    val_skus = random.sample(all_skus, k=int(len(all_skus) * 0.2))
    
    val_data = data[data['sku_id'].isin(val_skus)]
    train_data = data[~data['sku_id'].isin(val_skus)]
    
    return train_data, val_data