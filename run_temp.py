from fastapi import FastAPI
from api.routers import auth, predictions

app = FastAPI()

@app.get("/", response_model=dict)
async def root():
    """根路径返回欢迎信息"""
    return {
        "message": "欢迎使用FastAPI服务",
        "documentation": "http://localhost:8001/docs",
        "status": "running"
    }

app.include_router(auth.router, prefix='/api/auth')
app.include_router(predictions.router, prefix='/api/predictions')

if __name__ == "__main__":
    import uvicorn
    print("已注册路由:")
    for route in app.routes:
        print(f"{route.path} ({', '.join(route.methods)})")
    uvicorn.run(app, host="0.0.0.0", port=8002)