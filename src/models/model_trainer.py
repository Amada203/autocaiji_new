import pandas as pd
import numpy as np
import argparse
import joblib
import os
import sys
import logging
from datetime import datetime, timedelta
from .fusion_model import PriceChangePredictor
from ..data.data_fetcher import DataFetcher
from ..data.mysql_writer import MySQLWriter
from ..features.feature_engineering import FeatureEngineer

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/model_training.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, config):
        """
        初始化模型训练器
        
        Args:
            config (dict): 配置字典，包含:
                - impala_config: Impala连接配置
                - mysql_config: MySQL连接配置
                - model_params: 模型参数
                - train_end: 训练集结束日期 (YYYY-MM-DD)
                - val_end: 验证集结束日期 (YYYY-MM-DD)
                - test_end: 测试集结束日期 (YYYY-MM-DD)
        """
        self.config = config
        self.model = PriceChangePredictor(
            prophet_params=config.get('prophet_params', {}),
            lgbm_params=config.get('lgbm_params', {})
        )
        self.data_fetcher = DataFetcher(config['impala_config'])
        self.mysql_writer = MySQLWriter(config['mysql_config'])
        self.feature_engineer = FeatureEngineer()
        
    def train(self):
        """训练价格变动预测模型"""
        try:
            logger.info("开始训练价格变动预测模型")
            
            # 1. 获取数据
            logger.info("从Impala获取训练数据...")
            datasets = self.data_fetcher.fetch_training_data(
                train_end=self.config['train_end'],
                val_end=self.config['val_end'],
                test_end=self.config['test_end']
            )
            
            # 2. 特征工程
            logger.info("执行特征工程...")
            train_df = self.feature_engineer.transform(datasets['train'])
            val_df = self.feature_engineer.transform(datasets['val'])
            test_df = self.feature_engineer.transform(datasets['test'])
            
            # 3. 训练模型
            logger.info("训练融合模型...")
            self.model.fit(train_df, val_df)
            
            # 4. 评估模型
            logger.info("评估模型性能...")
            metrics = self.model.evaluate(test_df)
            
            # 5. 保存模型和结果
            logger.info("保存模型和评估结果...")
            self._save_model()
            self._save_results(metrics)
            
            logger.info("模型训练流程完成")
            return metrics
            
        except Exception as e:
            logger.error(f"模型训练流程失败: {str(e)}", exc_info=True)
            raise
    
    def _prepare_features(self, df):
        """准备特征和目标变量"""
        # 删除缺失值
        df = df.dropna(subset=[self.target])
        
        # 选择特征
        self.features = [
            'day_of_week', 'month', 'price_diff', 'is_promotion'
        ]
        
        # 准备特征矩阵和目标向量
        self.X = df[self.features]
        self.y = df[self.target]
        
        logger.info(f"特征: {self.features}")
        logger.info(f"样本数量: {len(df)}")
        logger.info(f"正样本比例: {self.y.mean():.2%}")
    
    def _evaluate_model(self, X_test, y_test):
        """评估模型性能"""
        # 预测
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]
        
        # 计算评估指标
        report = classification_report(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        
        logger.info(f"模型评估:\n{report}")
        logger.info(f"AUC: {auc:.4f}")
    
    def _save_model(self, scaler):
        """保存模型和相关组件"""
        # 创建模型目录
        os.makedirs('models', exist_ok=True)
        
        # 保存模型
        model_path = 'models/price_model.pkl'
        joblib.dump(self.model, model_path)
        
        # 保存缩放器
        scaler_path = 'models/scaler.pkl'
        joblib.dump(scaler, scaler_path)
        
        # 保存特征列表
        with open('models/features.txt', 'w') as f:
            f.write('\n'.join(self.features))
        
        logger.info(f"模型已保存: {model_path}")

if __name__ == "__main__":
    # 创建日志目录
    os.makedirs('logs', exist_ok=True)
    
    # 训练模型
    trainer = ModelTrainer()
    trainer.train()