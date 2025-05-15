from fastapi import APIRouter, Depends, HTTPException
from typing import List, Optional
from datetime import date
from api.services.prediction_service import PredictionService
from api.routers.auth import get_current_active_user

router = APIRouter()
prediction_service = PredictionService()

@router.get("/", response_model=List[dict])
async def get_predictions(
    limit: Optional[int] = 100,
    current_user: dict = Depends(get_current_active_user)
):
    """获取预测数据列表，包含SKU采样计划和训练记录"""
    try:
        predictions = prediction_service.get_predictions(limit=limit)
        training_logs = prediction_service.get_training_logs()
        
        return [{
            "prediction": pred,
            "sampling_plan": prediction_service.generate_sampling_plan(pred["sku"]),
            "training_history": training_logs.get(pred["sku"], []),
            "last_trained": max(training_logs.get(pred["sku"], []), default=None)
        } for pred in predictions]
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"获取预测数据失败: {str(e)}"
        )