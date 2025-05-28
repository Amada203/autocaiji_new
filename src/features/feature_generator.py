import pandas as pd
import numpy as np
from typing import List
import logging

def fetch_raw_data(sku: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    拉取指定sku和日期范围的原始数据
    这里仅为接口定义，实际实现可对接数据库或文件
    """
    # TODO: 实现数据拉取
    pass

def generate_features(
    raw_df: pd.DataFrame,
    predict_date: str,
    windows: List[int] = [1, 3, 7, 14, 30]
) -> pd.DataFrame:
    """
    输入原始数据，补全日期，统计采集次数，生成滑窗特征等
    返回特征DataFrame
    """
    # 1. 日期补全
    sku = raw_df['sku'].iloc[0]
    raw_df['date'] = pd.to_datetime(raw_df['date'])
    all_dates = pd.date_range(raw_df['date'].min(), predict_date)
    df = pd.DataFrame({'date': all_dates})
    df['sku'] = sku

    # 2. 采集次数统计（一天算一次，补全为0）
    collect_days = raw_df['date'].dt.date.unique()
    df['collect_count'] = df['date'].dt.date.apply(lambda d: 1 if d in collect_days else 0)

    # 3. 滑窗特征
    for w in windows:
        df[f'collect_count_mean_{w}d'] = (
            df['collect_count'].rolling(window=w, min_periods=1).mean()
        )
        df[f'collect_count_sum_{w}d'] = (
            df['collect_count'].rolling(window=w, min_periods=1).sum()
        )

    # 4. 价格前向优先，后向补充
    price_map = raw_df.set_index('date')['price']
    df['price'] = df['date'].map(price_map)
    df['price'] = df['price'].ffill().bfill()

    # 5. 促销标志，缺失默认为0
    if 'promotion' in raw_df.columns:
        promo_map = raw_df.set_index('date')['promotion']
        df['promotion'] = df['date'].map(promo_map).fillna(0)
    else:
        df['promotion'] = 0

    # 6. 价格变动标签（t日价格 != t-1日价格）
    df['price_change_label'] = (df['price'] != df['price'].shift(1)).astype(int)
    df['price_change_label'] = df['price_change_label'].fillna(0)

    # 7. 只保留到预测日
    df = df[df['date'] <= pd.to_datetime(predict_date)]

    # 8. 返回
    return df.reset_index(drop=True)

def batch_generate_features(
    sku_list: List[str], 
    predict_date: str, 
    start_date: str, 
    end_date: str
) -> pd.DataFrame:
    """
    支持批量sku、批量日期的特征生成
    """
    all_features = []
    for sku in sku_list:
        raw_df = fetch_raw_data(sku, start_date, end_date)
        features = generate_features(raw_df, predict_date)
        all_features.append(features)
    return pd.concat(all_features, ignore_index=True)

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
    collect_flag = df.groupby(['sku_id', 'date']).size().reset_index(name='tmp_count')
    collect_flag['collect_count'] = 1
    df = df.merge(collect_flag[['sku_id', 'date', 'collect_count']], on=['sku_id', 'date'], how='left')
    df['collect_count'] = df['collect_count'].fillna(0)
    for w in windows:
        df[f'collect_count_mean_{w}d'] = df.groupby('sku_id')['collect_count'].transform(lambda x: x.rolling(w, min_periods=1).mean())
        df[f'collect_count_sum_{w}d'] = df.groupby('sku_id')['collect_count'].transform(lambda x: x.rolling(w, min_periods=1).sum())
    return df

def add_time_features(df):
    """
    添加时间相关特征：星期、是否周末、月份、日等
    """
    df['day_of_week'] = df['date'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    return df

def add_price_mean_features(df, window_sizes=[3, 7, 14, 30]):
    """
    添加价格滑窗均值特征 price_mean_{w}d
    """
    for w in window_sizes:
        df[f'price_mean_{w}d'] = df.groupby('sku_id')['discount_price'].transform(
            lambda x: x.rolling(window=w, min_periods=1).mean()
        )
    return df

def rolling_slope(series, window):
    """滑动窗口线性回归斜率，向量化实现"""
    y = series.values
    n = len(y)
    slopes = np.full(n, np.nan)
    for i in range(window-1, n):
        y_window = y[i-window+1:i+1]
        x = np.arange(window)
        if np.any(np.isnan(y_window)):
            continue
        A = np.vstack([x, np.ones(window)]).T
        m, _ = np.linalg.lstsq(A, y_window, rcond=None)[0]
        slopes[i] = m
    return slopes

def add_price_trend_features(df, window_sizes=[3, 7, 14, 30]):
    """
    添加价格趋势特征 price_trend_{w}d（线性回归斜率，向量化实现）
    """
    for w in window_sizes:
        df[f'price_trend_{w}d'] = (
            df.groupby('sku_id')['discount_price']
            .transform(lambda x: pd.Series(rolling_slope(x, w), index=x.index))
        )
    return df

def add_price_std_features(df, window_sizes=[3, 7, 14, 30]):
    """
    添加价格滑窗标准差特征 price_std_{w}d
    """
    for w in window_sizes:
        df[f'price_std_{w}d'] = df.groupby('sku_id')['discount_price'].transform(
            lambda x: x.rolling(window=w, min_periods=1).std()
        )
    return df

def add_days_since_last_change(df):
    """
    添加距离上次价格变动的天数特征 days_since_last_change
    """
    df = df.sort_values(['sku_id', 'date'])
    df['days_since_last_change'] = (
        df.groupby('sku_id')['price_change'].cumsum().groupby(df['sku_id'], group_keys=False).apply(lambda x: x.groupby(x).cumcount())
    ).reset_index(level=0, drop=True)
    return df

def add_product_features(df):
    """
    构建商品特征：价格区间、价格稳定性、价格变动频率
    """
    if 'discount_price' in df.columns:
        df['price_range'] = pd.qcut(df['discount_price'], q=5, labels=['very_low', 'low', 'medium', 'high', 'very_high'])
        df['price_stability'] = df.groupby('sku_id')['discount_price'].transform(
            lambda x: x.rolling(window=30, min_periods=1).std()
        )
    if 'price_change' in df.columns:
        df['price_change_frequency'] = df.groupby('sku_id')['price_change'].transform('mean')
    return df

def add_cross_features(df):
    """
    构建交叉特征：价格区间与时间、变动方向与时间等
    """
    if 'price_range' in df.columns and 'is_weekend' in df.columns:
        df['price_range_weekend'] = df['price_range'].astype(str) + '_' + df['is_weekend'].astype(str)
    if 'price_change_direction' in df.columns and 'is_weekend' in df.columns:
        df['price_direction_weekend'] = df['price_change_direction'].astype(str) + '_' + df['is_weekend'].astype(str)
    return df

def assert_no_leakage(df, cutoff_date=None):
    """
    防止未来数据泄漏，断言所有特征不使用cutoff_date之后的数据
    """
    if cutoff_date is not None:
        max_date = pd.to_datetime(df['date']).max()
        assert max_date <= pd.to_datetime(cutoff_date), f"数据包含未来日期({max_date})，超过截止日期({cutoff_date})"

def add_price_change_freq_features(df, window_sizes=[3, 7, 14, 30]):
    """
    添加价格变动频率滑窗特征 price_change_freq_{w}d
    """
    for w in window_sizes:
        df[f'price_change_freq_{w}d'] = df.groupby('sku_id')['price_change'].transform(
            lambda x: x.rolling(window=w, min_periods=1).mean()
        )
    return df

def generate_all_features(df, windows=[1, 3, 7, 14, 30], trend_windows=[3, 7, 14, 30], cutoff_date=None, predict_date=None):
    """
    统一特征工程主流程，整合所有基础与高级特征
    """
    # 字段名清理，去除空格
    df.columns = df.columns.str.strip()
    if 'dt' in df.columns:
        df = df.rename(columns={'dt': 'date'})
    if 'datetime' in df.columns:
        df = df.rename(columns={'datetime': 'date'})
    if 'date' not in df.columns:
        raise ValueError(f'输入数据缺少date列！实际字段：{list(df.columns)}')
    df['date'] = pd.to_datetime(df['date'])
    # 补全到预测日，支持未来日期
    if predict_date is not None:
        max_date = pd.to_datetime(df['date']).max()
        predict_date = pd.to_datetime(predict_date)
        if predict_date > max_date:
            sku_ids = df['sku_id'].unique()
            all_dates = pd.date_range(df['date'].min(), predict_date)
            full_index = pd.MultiIndex.from_product([sku_ids, all_dates], names=['sku_id', 'date'])
            df = df.set_index(['sku_id', 'date']).reindex(full_index).reset_index()
            # 用历史最后一天的特征向后填充
            df = df.sort_values(['sku_id', 'date'])
            for col in ['discount_price', 'is_promotion']:
                if col in df.columns:
                    df[col] = df.groupby('sku_id')[col].ffill()
            # 对价格类特征做趋势外推（线性外推）
            for sku in sku_ids:
                sku_df = df[df['sku_id'] == sku]
                hist = sku_df[sku_df['date'] <= max_date]
                future = sku_df[sku_df['date'] > max_date]
                if not hist.empty and not future.empty:
                    # 线性拟合历史价格
                    x = (hist['date'] - hist['date'].min()).dt.days.values
                    y = hist['discount_price'].values
                    if len(x) > 1:
                        coef = np.polyfit(x, y, 1)
                        for i, row in future.iterrows():
                            day = (row['date'] - hist['date'].min()).days
                            pred_price = coef[0] * day + coef[1]
                            df.loc[(df['sku_id'] == sku) & (df['date'] == row['date']), 'discount_price'] = pred_price
    df = standardize_date_column(df)
    df = fill_missing_prices(df)
    df = add_price_change_features(df)
    df = add_price_change_freq_features(df, window_sizes=trend_windows)
    df = add_collect_count_features(df, windows=windows)
    df = add_time_features(df)
    df = add_price_mean_features(df, window_sizes=trend_windows)
    df = add_price_trend_features(df, window_sizes=trend_windows)
    df = add_price_std_features(df, window_sizes=trend_windows)
    df = add_days_since_last_change(df)
    df = add_product_features(df)
    df = add_cross_features(df)
    df = convert_feature_types(df)
    assert_no_leakage(df, cutoff_date)
    # 统一命名，兼容 change_freq_7d/30d
    for w in [7, 30]:
        if f'price_change_freq_{w}d' in df.columns:
            df[f'change_freq_{w}d'] = df[f'price_change_freq_{w}d']
    logger = None
    try:
        logger = logging.getLogger("feature_generator")
    except:
        pass
    if logger:
        logger.info(f"特征工程生成的所有特征: {list(df.columns)}")
    else:
        print("特征工程生成的所有特征:", list(df.columns))
    return df

def convert_feature_types(df):
    """
    自动转换数值型和类别型特征，保证与原有一致
    """
    numeric_features = [
        col for col in df.columns if any(s in col for s in ['mean', 'std', 'sum', 'freq', 'amount', 'ratio', 'stability', 'days_since'])
    ]
    categorical_features = [
        'day_of_week', 'is_weekend', 'month', 'day', 'price_range', 'price_range_weekend', 'price_direction_weekend'
    ]
    for col in numeric_features:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    for col in categorical_features:
        if col in df.columns:
            df[col] = df[col].astype('category')
    return df

# 其余高级特征、主流程等后续补充 