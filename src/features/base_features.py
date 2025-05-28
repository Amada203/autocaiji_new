import pandas as pd
import numpy as np

def standardize_date_column(df):
    """
    标准化日期列，将'dt'重命名为'date'，并转为datetime类型。
    """
    if 'dt' in df.columns:
        df = df.rename(columns={'dt': 'date'})
    df['date'] = pd.to_datetime(df['date'])
    return df

def fill_missing_prices(df):
    """
    对每个SKU的价格进行前向优先、后向补充填充。
    """
    df['discount_price'] = df.groupby('sku_id')['discount_price'].transform(lambda x: x.ffill().bfill())
    return df

def add_price_change_features(df):
    """
    生成价格变动相关特征，包括：
    prev_price, price_change, price_change_amount, price_change_ratio, price_change_direction
    """
    df['prev_price'] = df.groupby('sku_id')['discount_price'].shift(1)
    df['price_change'] = (df['discount_price'] != df['prev_price']).astype(int)
    df['price_change_amount'] = df['discount_price'] - df['prev_price']
    df['price_change_ratio'] = df['price_change_amount'] / df['prev_price'].replace(0, np.nan)
    df['price_change_direction'] = np.sign(df['price_change_amount'])
    df = df.fillna({
        'prev_price': df['discount_price'],
        'price_change': 0,
        'price_change_amount': 0,
        'price_change_ratio': 0,
        'price_change_direction': 0
    })
    return df

def add_collect_count_features(df, windows=[1, 3, 7, 14, 30]):
    """
    统计采集天数及其滑动窗口特征。
    - collect_count: 每天每SKU是否有采集（有为1，无为0）
    - collect_count_mean_{w}d: 滑动窗口w天的采集均值
    - collect_count_sum_{w}d: 滑动窗口w天的采集天数
    """
    # 标记原始采集天数（有采集为1，无采集为0）
    collect_flag = df.groupby(['sku_id', 'date']).size().reset_index(name='tmp_count')
    collect_flag['collect_count'] = 1  # 只要有记录就为1
    df = df.merge(collect_flag[['sku_id', 'date', 'collect_count']], on=['sku_id', 'date'], how='left')
    # 填充缺失为0（补全生成的日期）
    df['collect_count'] = df['collect_count'].fillna(0)
    # 滑动窗口特征
    for w in windows:
        df[f'collect_count_mean_{w}d'] = df.groupby('sku_id')['collect_count'].transform(lambda x: x.rolling(w, min_periods=1).mean())
        df[f'collect_count_sum_{w}d'] = df.groupby('sku_id')['collect_count'].transform(lambda x: x.rolling(w, min_periods=1).sum())
    return df 