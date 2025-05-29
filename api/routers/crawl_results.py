from fastapi import APIRouter, Depends, HTTPException, Query
from typing import List, Dict, Any, Optional
from datetime import date, datetime
from pydantic import BaseModel
from api.services.crawl_service import CrawlService
from api.routers.auth import get_current_active_user
import pandas as pd

router = APIRouter()
crawl_service = CrawlService()

class CrawlResult(BaseModel):
    sku_id: str
    price: float
    timestamp: Optional[datetime] = None

class CrawlResultsRequest(BaseModel):
    results: List[CrawlResult]
    crawl_date: Optional[date] = None

class CrawlResultsResponse(BaseModel):
    status: str
    processed_count: int

@router.post("", response_model=CrawlResultsResponse)
async def submit_crawl_results(
    request: CrawlResultsRequest,
    current_user = Depends(get_current_active_user)
):
    """上报爬取结果"""
    try:
        # 处理爬取结果
        result = crawl_service.process_crawl_results(
            results=request.results,
            crawl_date=request.crawl_date
        )
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理爬取结果失败: {str(e)}")

@router.get("/stats", response_model=Dict[str, Any])
async def get_crawl_stats(
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    current_user = Depends(get_current_active_user)
):
    """
    获取爬取统计信息，支持YYYY-MM-DD或完整ISO时间字符串
    """
    try:
        start = pd.to_datetime(start_date).date() if start_date else None
        end = pd.to_datetime(end_date).date() if end_date else None
        stats = crawl_service.get_crawl_stats(start, end)
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取爬取统计失败: {str(e)}")