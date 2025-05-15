#!/usr/bin/env python3
"""
系统启动脚本，用于启动API服务和定时任务
"""

import os
import sys
import subprocess
import multiprocessing
import logging
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/startup.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def start_api_service():
    """启动API服务"""
    logger.info("启动API服务...")
    try:
        # 使用uvicorn启动FastAPI应用
        subprocess.run([
            "uvicorn",
            "api.main:app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--reload",
            "--log-level", "info"
        ], check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"API服务启动失败: {str(e)}")
    except KeyboardInterrupt:
        logger.info("API服务已停止")

def start_scheduler():
    """启动定时任务调度器"""
    logger.info("启动定时任务调度器...")
    try:
        # 使用apscheduler运行定时任务
        subprocess.run([
            "python",
            "tasks/prediction_cron.py"
        ], check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"定时任务调度器启动失败: {str(e)}")
    except KeyboardInterrupt:
        logger.info("定时任务调度器已停止")

def main():
    """主函数"""
    logger.info("开始启动系统...")
    
    # 创建必要的目录
    os.makedirs("logs", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    # 创建进程列表
    processes = []
    
    try:
        # 启动API服务进程
        api_process = multiprocessing.Process(target=start_api_service)
        api_process.start()
        processes.append(api_process)
        
        # 启动定时任务进程
        scheduler_process = multiprocessing.Process(target=start_scheduler)
        scheduler_process.start()
        processes.append(scheduler_process)
        
        # 等待所有进程完成
        for process in processes:
            process.join()
            
    except KeyboardInterrupt:
        logger.info("接收到中断信号，停止所有进程...")
        for process in processes:
            process.terminate()
        
    except Exception as e:
        logger.error(f"系统启动失败: {str(e)}")
        
    finally:
        logger.info("系统已停止")

if __name__ == "__main__":
    main()