#!/usr/bin/env python3
"""
定时任务脚本，用于定期运行数据管道，更新预测结果
包含重试机制、详细的错误处理和状态监控
"""

import os
import sys
import logging
import time
import json
from datetime import datetime
from typing import Dict, Any
import mysql.connector
from tenacity import retry, stop_after_attempt, wait_exponential

# 添加项目根目录和src目录到Python路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

# 导入数据管道和配置
from src.data.data_pipeline import DataPipeline
from src.utils.data_normalizer import DataNormalizer

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

logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)

def load_database_config() -> Dict[str, Any]:
    """加载MySQL数据库配置"""
    try:
        with open("config/database.json", "r") as f:
            config = json.load(f)
            # 只返回MySQL配置部分
            return {
                'host': config['mysql']['host'],
                'port': config['mysql']['port'],
                'user': config['mysql']['user'],
                'password': config['mysql']['password'],
                'database': config['mysql']['database']
            }
    except Exception as e:
        logger.error(f"加载数据库配置失败: {str(e)}")
        raise

def check_database_connection(config: Dict[str, Any]) -> bool:
    """检查数据库连接状态"""
    try:
        conn = mysql.connector.connect(
            **config,
            auth_plugin='mysql_native_password'
        )
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        cursor.fetchall()
        cursor.close()
        conn.close()
        logger.info("数据库连接测试成功")
        return True
    except Exception as e:
        logger.error(f"数据库连接测试失败: {str(e)}")
        return False

def check_system_resources() -> bool:
    """检查系统资源状态"""
    try:
        # 检查磁盘空间
        disk_usage = os.statvfs('/')
        free_space_gb = (disk_usage.f_bavail * disk_usage.f_frsize) / (1024 * 1024 * 1024)
        if free_space_gb < 1:  # 如果剩余空间小于1GB
            logger.error(f"磁盘空间不足: {free_space_gb:.2f}GB")
            return False
            
        # 检查必要的目录和文件
        required_paths = [
            "logs",
            "data",
            "models",
            "config/database.json",
            "src/data/data_pipeline.py"
        ]
        for path in required_paths:
            if not os.path.exists(path):
                logger.error(f"必要的路径不存在: {path}")
                return False
                
        logger.info("系统资源检查通过")
        return True
    except Exception as e:
        logger.error(f"系统资源检查失败: {str(e)}")
        return False

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def run_pipeline_with_retry() -> bool:
    """运行数据管道（带重试机制）"""
    logger.info("开始运行数据管道")
    start_time = time.time()
    
    try:
        # 初始化数据管道
        pipeline = DataPipeline()
        
        # 记录开始状态
        logger.info("数据管道初始化成功，开始处理数据")
        
        # 运行完整的管道
        success = pipeline.run_full_pipeline()
        
        # 记录结果
        if success:
            end_time = time.time()
            duration = end_time - start_time
            logger.info(f"数据管道运行成功，耗时: {duration:.2f} 秒")
            
            # 记录处理的数据统计
            stats = pipeline.get_processing_stats()
            logger.info(f"处理统计: {json.dumps(stats, indent=2)}")
            
            return True
        else:
            logger.error("数据管道运行失败")
            return False
            
    except Exception as e:
        logger.exception(f"数据管道运行异常: {str(e)}")
        raise

def update_status_file(success: bool, message: str):
    """更新状态文件"""
    status = {
        "last_run": datetime.now().isoformat(),
        "success": success,
        "message": message
    }
    try:
        with open("logs/pipeline_status.json", "w") as f:
            json.dump(status, f, indent=2)
    except Exception as e:
        logger.error(f"更新状态文件失败: {str(e)}")

def main():
    """主函数"""
    logger.info("开始执行定时任务")
    
    try:
        # 创建必要的目录
        os.makedirs("logs", exist_ok=True)
        os.makedirs("data", exist_ok=True)
        os.makedirs("models", exist_ok=True)
        
        # 检查系统资源
        if not check_system_resources():
            logger.error("系统资源检查失败，终止执行")
            update_status_file(False, "系统资源检查失败")
            sys.exit(1)
            
        # 加载并检查数据库配置
        db_config = load_database_config()
        if not check_database_connection(db_config):
            logger.error("数据库连接检查失败，终止执行")
            update_status_file(False, "数据库连接失败")
            sys.exit(1)
            
        # 运行数据管道（带重试）
        success = run_pipeline_with_retry()
        
        if success:
            message = "定时任务执行成功"
            logger.info(message)
            update_status_file(True, message)
            sys.exit(0)
        else:
            message = "定时任务执行失败"
            logger.error(message)
            update_status_file(False, message)
            sys.exit(1)
            
    except Exception as e:
        message = f"定时任务执行异常: {str(e)}"
        logger.exception(message)
        update_status_file(False, message)
        sys.exit(1)

if __name__ == "__main__":
    main()