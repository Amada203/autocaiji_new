from pydantic import BaseModel
from typing import List, Union

class PredictItem(BaseModel):
    sku: str
    date: str  # 预测日期，格式YYYY-MM-DD

class PredictResponse(BaseModel):
    sku: str
    date: str
    predict_proba: float
    sampling_plan: str  # "采集" 或 "不采集"

class PredictRequest(BaseModel):
    items: List[PredictItem]

class PredictBatchRequest(BaseModel):
    items: List[PredictRequest]

class PredictBatchResponseItem(BaseModel):
    sku: str
    date: str
    predict_proba: float
    sampling_plan: str

class HistoryItem(BaseModel):
    date: str
    price: float
    price_change: int 