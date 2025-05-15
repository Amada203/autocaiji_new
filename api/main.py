from fastapi import FastAPI, Request, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import os
import uvicorn

# 导入路由
from api.routers import (
    auth,
    predictions,
    sampling_plans,
    crawl_results
)

# 创建FastAPI应用
app = FastAPI(
    title="商品价格变动预测系统",
    description="提供商品价格变动预测和智能采样计划生成服务",
    version="1.0.0"
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 创建必要的目录
os.makedirs("static", exist_ok=True)
os.makedirs("static/css", exist_ok=True)
os.makedirs("static/js", exist_ok=True)
os.makedirs("templates", exist_ok=True)
os.makedirs("data", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# 挂载静态文件目录
app.mount("/static", StaticFiles(directory="static"), name="static")

# 配置模板
templates = Jinja2Templates(directory="templates")

# 注册路由
app.include_router(auth.router, prefix="/api/auth", tags=["认证"])
app.include_router(predictions.router, prefix="/api/predictions", tags=["预测"])
app.include_router(sampling_plans.router, prefix="/api/sampling-plans", tags=["采样计划"])
app.include_router(crawl_results.router, prefix="/api/crawl-results", tags=["爬取结果"])

# 前端页面路由
@app.get("/")
async def home(request: Request):
    """首页"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/predictions")
async def predictions_page(request: Request):
    """预测结果页面"""
    return templates.TemplateResponse("predictions.html", {"request": request})

@app.get("/sampling")
async def sampling_page(request: Request):
    """采样计划页面"""
    return templates.TemplateResponse("sampling.html", {"request": request})

@app.get("/stats")
async def stats_page(request: Request):
    """统计信息页面"""
    return templates.TemplateResponse("stats.html", {"request": request})

# API根路由
@app.get("/api")
async def api_root():
    return {"message": "价格变动预测系统API服务正常运行"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)