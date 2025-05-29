from fastapi import APIRouter, Depends, HTTPException, Body
from typing import List, Optional
from datetime import date
from api.services.prediction_service import PredictionService
from api.routers.auth import get_current_active_user
from src.api.services.stats_service import get_top_query_skus, get_top_change_skus, get_model_threshold

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
        # 集成统计与阈值信息
        top_query = get_top_query_skus()
        top_change = get_top_change_skus()
        threshold = get_model_threshold()
        return {
            "result": result,
            "top_query_skus": [item.dict() for item in top_query],
            "top_change_skus": [item.dict() for item in top_change],
            "model_threshold": threshold
        }
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
    """
    获取预测统计信息，包括总数、高风险数、时间分布。
    time_distribution: [{hour: int, count: int}, ...]
    """
    try:
        results = prediction_service.get_predictions(limit=10000)
        total = len(results)
        high_risk = sum(1 for r in results if (r.get('change_probability') or r.get('probability') or 0) > 0.8)
        # 统计每小时预测数量
        from collections import Counter
        import pandas as pd
        hours = []
        for r in results:
            dt = r.get('date') or r.get('last_updated')
            if dt:
                try:
                    hour = pd.to_datetime(dt).hour
                    hours.append(hour)
                except Exception:
                    continue
        hour_counts = Counter(hours)
        time_distribution = [
            {"hour": h, "count": hour_counts.get(h, 0)} for h in range(24)
        ]
        return {
            "total_predictions": total,
            "high_prob_count": high_risk,
            "time_distribution": time_distribution
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取预测统计失败: {str(e)}")

# 新增：获取SKU详情接口，供前端弹窗/详情页使用
@router.get("/sku/{sku_id}")
async def get_sku_detail(sku_id: str, current_user: dict = Depends(get_current_active_user)):
    """
    获取指定SKU的详情，包括当前预测、历史价格等。
    返回结构示例：
    {
        "sku_id": "1184675",
        "current_prediction": { ... },
        "price_history": [ ... ]
    }
    """
    try:
        # 当前预测
        preds = prediction_service.get_predictions(limit=10000)
        pred = next((r for r in preds if str(r.get('sku_id') or r.get('sku')) == str(sku_id)), None)
        if not pred:
            raise HTTPException(status_code=404, detail="未找到该SKU的预测数据")
        # 保证current_price字段存在且为数字
        try:
            pred['current_price'] = float(pred.get('current_price', 0) or pred.get('discount_price', 0) or 0)
        except Exception:
            pred['current_price'] = 0.0
        # 历史价格
        # 这里假设prediction_service有history接口，或可用数据库查找
        # 可根据实际情况调整
        price_history = []
        try:
            # 尝试用history_api
            from src.api.schemas import HistoryItem
            import datetime
            today = datetime.date.today()
            start_date = (today - datetime.timedelta(days=30)).strftime('%Y-%m-%d')
            end_date = today.strftime('%Y-%m-%d')
            history = history_api(sku=sku_id, start_date=start_date, end_date=end_date)
            # history_api 可能返回pydantic对象或dict
            price_history = [h.dict() if hasattr(h, 'dict') else h for h in history]
            # 增加is_promotion字段（如有）
            for h in price_history:
                h['is_promotion'] = h.get('price_change', 0) != 0
        except Exception:
            pass
        return {
            "sku_id": sku_id,
            "current_prediction": pred,
            "price_history": price_history
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取SKU详情失败: {str(e)}")