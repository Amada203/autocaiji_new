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
from src.data.data_fetcher import DataFetcher
from src.features.base_features import (
    standardize_date_column, add_collect_count_features, fill_missing_prices, add_price_change_features
)
from src.features.feature_engineering import FeatureEngineer
from src.models.fusion_model import PriceChangePredictor
from src.data.mysql_writer import MySQLWriter

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

logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)

def start_api_service():
    """启动API服务"""
    logger.info("启动API服务...")
    try:
        # 使用uvicorn启动FastAPI应用
        subprocess.run([
            "uvicorn",
            "api.main:app",
            "--host", "0.0.0.0",
            "--port", "8002",
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

def preprocess_pipeline(df):
    df = standardize_date_column(df)
    df = add_collect_count_features(df)  # 必须在填充前
    df = fill_missing_prices(df)
    df = add_price_change_features(df)
    return df

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
        
        # 1. 数据获取
        logger.info("拉取训练/验证/测试集数据...")
        fetcher = DataFetcher()
        datasets = fetcher.fetch_training_data(
            train_end="2024-12-31",
            val_end="2025-03-31",
            test_end="2025-04-30"
        )
        train_df, val_df, test_df = datasets['train'], datasets['val'], datasets['test']
        logger.info(f"训练集: {train_df.shape}, 验证集: {val_df.shape}, 测试集: {test_df.shape}")

        # 2. 数据预处理
        logger.info("预处理训练集...")
        train_df = preprocess_pipeline(train_df)
        logger.info("预处理验证集...")
        val_df = preprocess_pipeline(val_df)
        logger.info("预处理测试集...")
        test_df = preprocess_pipeline(test_df)

        # 3. 高阶特征工程
        logger.info("特征工程...")
        fe = FeatureEngineer()
        train_df = fe.transform(train_df)
        val_df = fe.transform(val_df)
        test_df = fe.transform(test_df)

        # 4. 模型训练与验证
        logger.info("训练融合模型...")
        model = PriceChangePredictor()
        model.fit(train_df, val_df)

        # 5. 评估与报告
        logger.info("评估模型效果...")
        metrics = model.evaluate(test_df)
        print(metrics)

        # 6. 未来1天预测
        logger.info("预测未来1天价格变动...")
        future_pred = model.predict(test_df)
        # 合并预测结果到test_df
        test_df['probability'] = future_pred['probability']
        test_df['predicted_change'] = future_pred['predicted_change']

        # 7. 写入MySQL
        logger.info("写入预测结果到MySQL...")
        mysql_writer = MySQLWriter()  # 自动读取配置
        mysql_writer.write_predictions(test_df)
        logger.info("流程完成！")
        
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