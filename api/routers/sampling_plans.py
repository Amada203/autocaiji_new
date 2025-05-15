from fastapi import APIRouter, Depends, HTTPException, Query
from typing import List, Dict, Any, Optional
from datetime import date
from pydantic import BaseModel
from api.services.sampling_service import SamplingService
from api.routers.auth import get_current_active_user

router = APIRouter()
sampling_service = SamplingService()

class SamplingPlanRequest(BaseModel):
    prediction_date: Optional[date] = None
    target_capture_rate: float = 0.95
    use_stratified_sampling: bool = True
    max_samples: Optional[int] = None

class SamplingPlanResponse(BaseModel):
    date: date
    sku_ids: List[str]
    sampling_rate: float
    estimated_capture_rate: float
    estimated_cost_saving: float

class SamplingPlanStats(BaseModel):
    total_skus: int
    sampled_skus: int
    sampling_rate: float
    cost_saving: float
    strata_stats: Optional[List[Dict[str, Any]]] = None

@router.get("/today", response_model=SamplingPlanResponse)
async def get_today_plan(
    current_user = Depends(get_current_active_user)
):
    """获取当日采样计划"""
    try:
        today = date.today()
        plan = sampling_service.get_plan(today)
        if not plan:
            # 生成新计划
            plan = sampling_service.generate_plan(prediction_date=today)
        return plan
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取采样计划失败: {str(e)}")

@router.post("/generate", response_model=SamplingPlanResponse)
async def generate_sampling_plan(
    request: SamplingPlanRequest,
    current_user = Depends(get_current_active_user)
):
    """生成采样计划"""
    try:
        plan = sampling_service.generate_plan(
            prediction_date=request.prediction_date,
            target_capture_rate=request.target_capture_rate,
            use_stratified_sampling=request.use_stratified_sampling,
            max_samples=request.max_samples
        )
        return plan
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"生成采样计划失败: {str(e)}")

@router.get("/stats", response_model=SamplingPlanStats)
async def get_sampling_stats(
    plan_date: Optional[date] = None,
    current_user = Depends(get_current_active_user)
):
    """获取采样计划统计信息"""
    try:
        stats = sampling_service.get_plan_stats(plan_date)
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取统计信息失败: {str(e)}")