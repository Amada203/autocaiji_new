import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.data.data_fetcher import DataFetcher
from src.features.feature_engineering import FeatureEngineer
from src.models.fusion_model import PriceChangePredictor
from src.data.mysql_writer import MySQLWriter

def generate_test_data():
    """生成测试数据集"""
    dates = pd.date_range('2023-01-01', periods=30).tolist()
    skus = ['SKU001', 'SKU002']
    test_data = []
    
    for sku in skus:
        base_price = np.random.randint(50, 100)
        for i, date in enumerate(dates):
            # 模拟价格变动（约30%概率变动）
            if np.random.random() < 0.3 and i > 0:
                base_price += np.random.randint(-10, 10)
                
            test_data.append({
                'sku_id': sku,
                'date': date,
                'discount_price': max(10, base_price),
                'category': np.random.choice(['A', 'B']),
                'is_promotion': int(np.random.random() < 0.2)
            })
    
    return pd.DataFrame(test_data)

def test_integration():
    print("=== 开始端到端集成测试 ===")
    
    # 1. 准备测试数据
    print("生成测试数据...")
    test_df = generate_test_data()
    
    # 2. 模拟数据获取
    print("模拟数据获取...")
    impala_config = {'host': 'localhost', 'port': 21050}  # 测试配置
    mysql_config = {'host': 'localhost', 'user': 'test', 'password': 'test'}
    
    # 3. 特征工程
    print("执行特征工程...")
    fe = FeatureEngineer(cutoff_date='2023-01-30')
    features = fe.transform(test_df)
    print(f"生成特征数: {len(fe.get_feature_names())}")
    
    # 4. 模型训练
    print("训练模型...")
    model = PriceChangePredictor()
    model.fit(features[features['date'] <= '2023-01-20'],  # 训练集
              features[(features['date'] > '2023-01-20') & (features['date'] <= '2023-01-25')])  # 验证集
    
    # 5. 预测
    print("执行预测...")
    test_data = features[features['date'] > '2023-01-25']  # 测试集
    predictions = model.predict(test_data)
    print(f"预测结果示例: {predictions['predicted_change'][:5]}")
    
    # 6. 存储结果
    print("存储结果...")
    writer = MySQLWriter(mysql_config)
    # 注意: 实际测试时需要真实MySQL连接或mock
    
    print("=== 集成测试完成 ===")

if __name__ == '__main__':
    test_integration()