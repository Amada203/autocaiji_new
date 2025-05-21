#!/usr/bin/env python3
import sys
import os
import logging

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
from datetime import datetime, timedelta
from src.data.data_fetcher import DataFetcher
from src.data.data_processor import DataProcessor
from src.data.data_pipeline import DataPipeline

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/pipeline_test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def test_data_processor():
    """测试数据处理器"""
    logger.info("测试数据处理器...")
    
    # 创建测试数据
    test_data = pd.DataFrame({
        'sku_id': ['SKU001', 'SKU001', 'SKU002'],
        'dt': [datetime.now() - timedelta(days=2), 
               datetime.now() - timedelta(days=1),
               datetime.now()],
        'page_price': [100.0, 105.0, 200.0],
        'discount_price': [90.0, 95.0, 180.0]
    })
    
    # 测试正常数据
    processor = DataProcessor(test_data)
    result = processor.process()
    assert not result.empty, "处理器返回空结果"
    logger.info(f"处理器测试通过，结果形状: {result.shape}")

    # 测试空数据
    try:
        DataProcessor(pd.DataFrame()).process()
        assert False, "空数据测试失败"
    except ValueError:
        logger.info("空数据测试通过")

    # 测试缺失列
    try:
        bad_data = test_data.drop(columns=['page_price'])
        DataProcessor(bad_data).process()
        assert False, "缺失列测试失败"
    except ValueError:
        logger.info("缺失列测试通过")

def test_pipeline_integration():
    """测试完整管道集成"""
    logger.info("测试完整数据管道...")
    
    # 创建测试数据
    test_data = pd.DataFrame({
        'sku_id': ['SKU001', 'SKU001', 'SKU002'],
        'dt': [datetime.now() - timedelta(days=2), 
               datetime.now() - timedelta(days=1),
               datetime.now()],
        'page_price': [100.0, 105.0, 200.0],
        'discount_price': [90.0, 95.0, 180.0]
    })
    
    # 模拟数据获取
    class MockFetcher:
        def fetch_training_data(self):
            return test_data
    
    # 运行管道
    pipeline = DataPipeline(fetcher=MockFetcher())
    result = pipeline.run()
    
    assert not result.empty, "管道返回空结果"
    logger.info(f"管道测试通过，结果形状: {result.shape}")

def main():
    try:
        test_data_processor()
        test_pipeline_integration()
        logger.info("所有测试通过!")
    except Exception as e:
        logger.error(f"测试失败: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()