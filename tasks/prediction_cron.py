#!/usr/bin/env python3
"""
定时任务脚本，用于定期运行数据管道，更新预测结果
"""

import os
import sys
import logging
from datetime import datetime
import time

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 导入数据管道
from src.data.data_pipeline import DataPipeline

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/prediction_cron.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def run_pipeline():
    """运行数据管道"""
    logger.info("开始运行数据管道")
    start_time = time.time()
    
    try:
        pipeline = DataPipeline()
        success = pipeline.run_full_pipeline()
        
        if success:
            logger.info("数据管道运行成功")
        else:
            logger.error("数据管道运行失败")
        
        end_time = time.time()
        logger.info(f"数据管道运行耗时: {end_time - start_time:.2f} 秒")
        
        return success
    
    except Exception as e:
        logger.exception(f"数据管道运行异常: {str(e)}")
        return False

def main():
    """主函数"""
    logger.info("开始执行定时任务")
    
    # 创建必要的目录
    os.makedirs("logs", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    # 运行数据管道
    success = run_pipeline()
    
    if success:
        logger.info("定时任务执行成功")
        sys.exit(0)
    else:
        logger.error("定时任务执行失败")
        sys.exit(1)

if __name__ == "__main__":
    main()