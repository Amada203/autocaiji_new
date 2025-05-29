from fastapi import APIRouter, Query
from typing import List
from src.api.schemas import TopSkuItem, CompareResult, ModelThresholdResponse
from src.api.services.stats_service import (
    get_top_change_skus, get_top_query_skus, compare_real_pred, get_model_threshold
)

router = APIRouter()

@router.get("/top-change-skus", response_model=List[TopSkuItem])
def top_change_skus(n: int = Query(20, description="返回前N个SKU")):
    return get_top_change_skus(n)

@router.get("/top-query-skus", response_model=List[TopSkuItem])
def top_query_skus(n: int = Query(20, description="返回前N个SKU")):
    return get_top_query_skus(n)

@router.get("/compare-prediction", response_model=CompareResult)
def compare_prediction(sku: str, date: str):
    return compare_real_pred(sku, date)

@router.get("/model-threshold", response_model=ModelThresholdResponse)
def model_threshold():
    return ModelThresholdResponse(threshold=get_model_threshold()) 