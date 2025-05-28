import pandas as pd
import logging
from datetime import datetime, timedelta
from src.data.data_fetcher import DataFetcher

logger = logging.getLogger("data_fetcher")

def fetch_sku_history(sku_list, start_date, end_date):
    """
    拉取指定sku列表在指定日期范围内的历史数据，字段兼容，便于后续处理。
    """
    fetcher = DataFetcher()
    try:
        days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days
        df = fetcher.fetch_latest_data_by_skus(sku_list, days=days, end_date=end_date)
        if 'dt' in df.columns:
            df['date'] = df['dt']
        # 删除重复列
        if 'dt' in df.columns and 'date' in df.columns:
            df = df.drop(columns=['dt'])
        df = df.loc[:, ~df.columns.duplicated()]
        if 'sku' not in df.columns and 'sku_id' in df.columns:
            df['sku'] = df['sku_id']
        if 'discount_price' not in df.columns and 'price' in df.columns:
            df['discount_price'] = df['price']
        if 'promotion' not in df.columns and 'is_promotion' in df.columns:
            df['promotion'] = df['is_promotion']
        return df
    except Exception as e:
        logger.error(f"拉取数据失败: {str(e)}")
        return pd.DataFrame()

def fetch_training_data(train_end, val_end, test_end):
    """
    拉取训练/验证/测试集数据，按时间轴划分，字段兼容。
    """
    fetcher = DataFetcher()
    try:
        datasets = fetcher.fetch_training_data(
            train_end=train_end,
            val_end=val_end,
            test_end=test_end
        )
        for name, df in datasets.items():
            if 'dt' in df.columns:
                df['date'] = df['dt']
            # 删除重复列
            if 'dt' in df.columns and 'date' in df.columns:
                df = df.drop(columns=['dt'])
            df = df.loc[:, ~df.columns.duplicated()]
            if 'sku' not in df.columns and 'sku_id' in df.columns:
                df['sku'] = df['sku_id']
            if 'discount_price' not in df.columns and 'price' in df.columns:
                df['discount_price'] = df['price']
            if 'promotion' not in df.columns and 'is_promotion' in df.columns:
                df['promotion'] = df['is_promotion']
            datasets[name] = df
        return datasets
    except Exception as e:
        logger.error(f"拉取训练数据失败: {str(e)}")
        return {} 