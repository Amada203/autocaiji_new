from typing import List, Dict
from collections import Counter
import os
import pandas as pd
from src.api.schemas import TopSkuItem, CompareResult

# 假设历史数据和预测数据的获取方式如下（需根据实际情况替换）
HISTORY_DATA_PATH = os.getenv("HISTORY_DATA_PATH", "data/history.csv")
PREDICT_LOG_PATH = os.getenv("PREDICT_LOG_PATH", "data/predict_log.csv")
THRESHOLD_PATH = os.getenv("MODEL_THRESHOLD_PATH", "models/price_model_threshold.txt")

# 内存计数器，记录实时预测查询次数
top_query_counter = Counter()

def record_predict_query(sku: str):
    top_query_counter[sku] += 1

def get_top_query_skus(n=20) -> List[TopSkuItem]:
    return [TopSkuItem(sku=sku, count=count) for sku, count in top_query_counter.most_common(n)]

def get_top_change_skus(n=20) -> List[TopSkuItem]:
    if not os.path.exists(HISTORY_DATA_PATH):
        return []
    df = pd.read_csv(HISTORY_DATA_PATH)
    if 'sku' not in df.columns or 'price_change' not in df.columns:
        return []
    change_counts = df.groupby('sku')['price_change'].apply(lambda x: (x != 0).sum())
    top = change_counts.sort_values(ascending=False).head(n)
    return [TopSkuItem(sku=sku, count=int(count)) for sku, count in top.items()]

def compare_real_pred(sku: str, date: str) -> CompareResult:
    # 真实值
    if not os.path.exists(HISTORY_DATA_PATH) or not os.path.exists(PREDICT_LOG_PATH):
        return CompareResult(sku=sku, date=date, real=-1, pred=-1)
    df_real = pd.read_csv(HISTORY_DATA_PATH)
    df_pred = pd.read_csv(PREDICT_LOG_PATH)
    real_row = df_real[(df_real['sku'] == sku) & (df_real['date'] == date)]
    pred_row = df_pred[(df_pred['sku'] == sku) & (df_pred['date'] == date)]
    real = float(real_row['price'].iloc[0]) if not real_row.empty else -1
    pred = float(pred_row['predict'].iloc[0]) if not pred_row.empty else -1
    return CompareResult(sku=sku, date=date, real=real, pred=pred)

def get_model_threshold() -> float:
    if os.path.exists(THRESHOLD_PATH):
        with open(THRESHOLD_PATH) as f:
            return float(f.read().strip())
    return 0.5 