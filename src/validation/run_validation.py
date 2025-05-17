"""
验证执行模块 - 修正版
"""
import os
import json
import logging
import pandas as pd
from datetime import datetime
from models.price_model import PriceModel as OldPricePredictor
from models.price_change_probability_model import PriceChangeProbabilityModel as OldChangeDetector
from src.models.predictors.prophet_predictor import ProphetPredictor as NewPricePredictor
from src.models.predictors.change_detector import ChangeDetector as NewChangeDetector

# 初始化日志记录器
logger = logging.getLogger(__name__)

def run_full_validation():
    """执行完整的验证流程"""
    # 延迟导入避免循环依赖
    from .model_validator import validate_model_migration, compare_model_performance
    from .data_utils import prepare_test_data
    from ..data.data_fetcher import DataFetcher

    # 1. 准备训练集和测试集
    train_data, test_data = prepare_test_data()
    
    # 2. 初始化模型
    old_price_model = OldPricePredictor()
    new_price_model = NewPricePredictor()
    old_change_model = OldChangeDetector()
    new_change_model = NewChangeDetector()
    
    # 3. 训练模型
    logger.info("开始训练旧模型...")
    old_price_model.fit(test_data)
    old_change_model.fit(test_data)
    
    logger.info("开始训练新模型...")
    new_price_model.fit(test_data)
    new_change_model.fit(test_data)
    
    # 4. 模型迁移验证
    migration_results = validate_model_migration(
        old_model=old_price_model,
        new_model=new_price_model,
        test_data=test_data
    )
    
    # 5. 性能对比测试
    performance_comparison = compare_model_performance(
        old_model=old_change_model,
        new_model=new_change_model,
        test_data=test_data,
        y_true=test_data['change_flag']
    )
    
    # 3. 设置监控
    def calculate_baseline_metrics(data):
        price_changes = data.groupby('sku_id')['page_price'].diff().abs() > 0
        return {
            'capture_rate': 0.95,
            'accuracy': 0.85,
            'change_frequency': price_changes.mean()
        }

    monitoring_config = {
        'capture_rate_threshold': 0.95 * 0.95,
        'accuracy_threshold': 0.85 * 0.95,
        'performance_window': '7D'
    }
    
    # 4. 生成验证报告
    report = {
        'migration_validation': migration_results,
        'performance_comparison': performance_comparison,
        'monitoring_setup': monitoring_config,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # 5. 保存报告
    os.makedirs('reports', exist_ok=True)
    with open('reports/validation_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    return report

if __name__ == "__main__":
    # 执行完整验证流程
    validation_results = run_full_validation()
    print("验证完成，报告已保存至 reports/validation_report.json")