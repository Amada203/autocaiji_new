import pandas as pd
import numpy as np
import logging

def standardize_date_column(df):
    """
    标准化日期列，将'dt'重命名为'date'，并转为datetime类型。
    删除重复列，确保date唯一。
    """
    if 'dt' in df.columns:
        df = df.rename(columns={'dt': 'date'})
    # 删除重复列
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()]
    df['date'] = pd.to_datetime(df['date'])
    return df

def fill_missing_dates_and_prices(df):
    """
    对每个SKU补全所有日期，价格前向优先、后向补充，促销标志填充为0。
    """
    all_sku = df['sku_id'].unique()
    min_date = df['date'].min()
    max_date = df['date'].max()
    all_dates = pd.date_range(min_date, max_date)
    dfs = []
    for sku in all_sku:
        sku_df = df[df['sku_id'] == sku].copy()
        sku_df = sku_df.set_index('date')
        sku_df = sku_df.reindex(all_dates)
        sku_df['sku_id'] = sku
        sku_df['date'] = sku_df.index
        # 价格填充
        sku_df['discount_price'] = sku_df['discount_price'].ffill().bfill()
        # 促销标志填充
        if 'is_promotion' in sku_df.columns:
            sku_df['is_promotion'] = sku_df['is_promotion'].fillna(0)
        else:
            sku_df['is_promotion'] = 0
        dfs.append(sku_df.reset_index(drop=True))
    df_full = pd.concat(dfs, ignore_index=True)
    return df_full

def remove_outliers(df, price_col='discount_price', min_price=0, max_price=1e6):
    """
    移除价格异常值。
    """
    df = df[(df[price_col] > min_price) & (df[price_col] < max_price)]
    return df

def preprocess_data(df):
    """
    数据预处理主流程：标准化日期、补全日期和价格、填充促销、去除异常。
    """
    logger = logging.getLogger("data_processor")
    df = standardize_date_column(df)
    logger.info(f"原始数据行数: {len(df)}")
    df = fill_missing_dates_and_prices(df)
    logger.info(f"补全后数据行数: {len(df)}")
    df = remove_outliers(df)
    logger.info(f"去除异常后数据行数: {len(df)}")
    return df 