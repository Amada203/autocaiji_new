#!/usr/bin/env python3
"""
环境验证脚本
"""

import sys
import logging
import importlib
from typing import List, Dict, Tuple

# 必需依赖列表
REQUIRED_PACKAGES = [
    ('pandas', '>=1.3.0'),
    ('mysql-connector-python', '>=8.0.0'),
    ('impyla', '>=0.18.0'),
    ('scikit-learn', '>=1.0.0'),
    ('joblib', '>=1.0.0')
]

# 配置检查
CONFIG_CHECKS = [
    ('config/database.json', 'impala.host'),
    ('config/database.json', 'mysql.host')
]

def setup_logging():
    """配置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def check_packages(logger) -> bool:
    """检查Python包依赖"""
    logger.info("检查Python依赖包...")
    all_ok = True
    
    for pkg, version in REQUIRED_PACKAGES:
        try:
            mod = importlib.import_module(pkg.split('-')[0])
            actual_version = mod.__version__
            logger.info(f"✅ {pkg:20s} 已安装 (版本: {actual_version})")
        except ImportError:
            logger.error(f"❌ {pkg:20s} 未安装 (需要: {version})")
            all_ok = False
            
    return all_ok

def check_configs(logger) -> bool:
    """检查配置文件"""
    logger.info("检查配置文件...")
    all_ok = True
    
    try:
        import json
        import os
        
        for file, key in CONFIG_CHECKS:
            if not os.path.exists(file):
                logger.error(f"❌ 配置文件不存在: {file}")
                all_ok = False
                continue
                
            with open(file) as f:
                config = json.load(f)
                
            keys = key.split('.')
            val = config
            for k in keys:
                val = val.get(k, None)
                if val is None:
                    break
                    
            if val is None:
                logger.error(f"❌ 配置项缺失: {key} (文件: {file})")
                all_ok = False
            else:
                logger.info(f"✅ 配置项正常: {key} = {val}")
                
    except Exception as e:
        logger.error(f"配置检查失败: {str(e)}")
        all_ok = False
        
    return all_ok

def validate_data_pipeline(logger) -> bool:
    """验证数据管道"""
    logger.info("验证数据管道...")
    try:
        from src.data.data_pipeline import DataPipeline
        
        logger.info("初始化数据管道...")
        pipeline = DataPipeline()
        
        logger.info("运行测试管道...")
        success = pipeline.run_full_pipeline()
        
        if success:
            logger.info("✅ 数据管道验证成功")
        else:
            logger.error("❌ 数据管道运行失败")
            
        return success
        
    except Exception as e:
        logger.error(f"验证失败: {str(e)}")
        return False

def main():
    logger = setup_logging()
    logger.info("开始环境验证")
    
    # 执行检查
    pkg_ok = check_packages(logger)
    config_ok = check_configs(logger)
    
    if pkg_ok and config_ok:
        logger.info("基本检查通过，开始验证数据管道")
        pipeline_ok = validate_data_pipeline(logger)
    else:
        logger.error("请先解决依赖和配置问题")
        pipeline_ok = False
    
    # 最终报告
    logger.info("\n验证结果汇总:")
    logger.info(f"依赖包: {'通过' if pkg_ok else '失败'}")
    logger.info(f"配置文件: {'通过' if config_ok else '失败'}")
    logger.info(f"数据管道: {'通过' if pipeline_ok else '失败'}")
    
    sys.exit(0 if all([pkg_ok, config_ok, pipeline_ok]) else 1)

if __name__ == "__main__":
    main()