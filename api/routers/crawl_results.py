from fastapi import APIRouter, Depends, HTTPException
from typing import List, Dict, Any, Optional
from datetime import date, datetime
from pydantic import BaseModel
from api.services.crawl_service import CrawlService
from api.routers.auth import get_current_active_user

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
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    current_user = Depends(get_current_active_user)
):
    """获取爬取统计信息"""
    try:
        stats = crawl_service.get_crawl_stats(start_date, end_date)
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取爬取统计失败: {str(e)}")