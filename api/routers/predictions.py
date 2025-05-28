from fastapi import APIRouter, Depends, HTTPException, Body
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

@router.post("/realtime", response_model=List[dict])
async def realtime_predict(
    sku_list: List[str] = Body(..., embed=True),
    days: int = Body(1, embed=True),
    end_date: Optional[str] = Body(None, embed=True),
    current_user: dict = Depends(get_current_active_user)
):
    """前端实时触发预测，支持批量SKU、天数、指定起点日期"""
    try:
        result = prediction_service.predict_realtime(sku_list, days=days, end_date=end_date)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"实时预测失败: {str(e)}"
        )

@router.get("/latest")
async def get_latest_predictions(limit: int = 100, current_user = Depends(get_current_active_user)):
    """获取最新预测结果列表，适配前端表格"""
    try:
        service = prediction_service
        results = service.get_predictions(limit=limit)
        # 适配前端字段
        formatted = []
        for r in results:
            formatted.append({
                'sku_id': r.get('sku_id') or r.get('sku'),
                'sku_name': r.get('sku_name', f"商品_{r.get('sku_id') or r.get('sku')}") if r.get('sku_id') or r.get('sku') else '-',
                'current_price': r.get('discount_price') or r.get('current_price') or 0,
                'predicted_prob': r.get('probability') or r.get('predicted_prob') or 0,
                'last_updated': r.get('date') or r.get('last_updated')
            })
        return formatted
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取最新预测失败: {str(e)}")

@router.get("/stats")
async def get_prediction_stats(current_user: dict = Depends(get_current_active_user)):
    try:
        results = prediction_service.get_predictions(limit=10000)
        total = len(results)
        high_risk = sum(1 for r in results if (r.get('change_probability') or r.get('probability') or 0) > 0.8)
        return {
            "total_predictions": total,
            "high_prob_count": high_risk,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取预测统计失败: {str(e)}")