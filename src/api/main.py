from fastapi import FastAPI, HTTPException
from .schemas import PredictRequest, PredictResponse, PredictItem, HistoryItem
from src.data.fetcher import fetch_sku_history
from src.data.processor import preprocess_data
from src.features.feature_generator import generate_all_features
from src.model.predictor import ModelPredictor
import os
import pandas as pd
import logging
from typing import List
import traceback
from datetime import timedelta

app = FastAPI()
logger = logging.getLogger("api_main")

MODEL_PATH = os.getenv("MODEL_PATH", "models/price_model.pkl")
FEATURES_PATH = MODEL_PATH.replace('.pkl', '_features.json')
predictor = ModelPredictor(MODEL_PATH, FEATURES_PATH)

def _predict_single(item: PredictItem) -> PredictResponse:
    """单个SKU预测的内部函数"""
    # 1. 拉取历史数据（默认取预测日前30天）
    end_date = pd.to_datetime(item.date)
    start_date = (end_date - pd.Timedelta(days=30)).strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')
    raw_df = fetch_sku_history([item.sku], start_date=start_date, end_date=end_date_str)
    if raw_df.empty:
        raise HTTPException(status_code=404, detail=f"未找到SKU {item.sku}的历史数据")
    
    # 2. 预处理
    processed_df = preprocess_data(raw_df)
    
    # 3. 特征工程
    features = generate_all_features(processed_df, predict_date=item.date)
    
    # 4. 只取预测日特征
    features_pred = features[features['date'] == pd.to_datetime(item.date)]
    if features_pred.empty:
        raise HTTPException(status_code=400, detail=f"SKU {item.sku}预测日特征生成失败")
    
    # 5. 预测
    result_df = predictor.predict(features_pred)
    proba = float(result_df['predict_proba'].iloc[0])
    
    # 6. 采样计划（阈值可配置）
    threshold = float(os.getenv("SAMPLING_THRESHOLD", 0.5))
    sampling_plan = "采集" if proba > threshold else "不采集"
    
    return PredictResponse(
        sku=item.sku,
        date=item.date,
        predict_proba=proba,
        sampling_plan=sampling_plan
    )

@app.post("/predict", response_model=List[PredictResponse])
def predict_api(request: PredictRequest):
    """统一的预测接口，支持单SKU和批量SKU预测"""
    results = []
    for item in request.items:
        try:
            result = _predict_single(item)
            results.append(result)
        except HTTPException as e:
            # 对于404和400错误，返回错误信息
            results.append(PredictResponse(
                sku=item.sku,
                date=item.date,
                predict_proba=-1,
                sampling_plan=f"错误: {e.detail}"
            ))
        except Exception as e:
            # 对于其他错误，记录日志并返回通用错误信息
            logger.error(f"预测异常: {str(e)}")
            logger.error(traceback.format_exc())
            results.append(PredictResponse(
                sku=item.sku,
                date=item.date,
                predict_proba=-1,
                sampling_plan="预测失败"
            ))
    return results

@app.get("/history", response_model=List[HistoryItem])
def history_api(sku: str, start_date: str, end_date: str):
    try:
        raw_df = fetch_sku_history([sku], start_date, end_date)
        if raw_df.empty:
            return []
        processed_df = preprocess_data(raw_df)
        # 只保留需要的字段
        out = processed_df[['date', 'discount_price', 'price_change']].copy()
        out.rename(columns={'discount_price': 'price'}, inplace=True)
        out['date'] = out['date'].astype(str)
        return [HistoryItem(**row) for row in out.to_dict(orient='records')]
    except Exception as e:
        logger.error(f"历史接口异常: {str(e)}")
        return [] 