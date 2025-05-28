import pandas as pd
import logging
import joblib
import os
from src.data.fetcher import fetch_training_data
from src.data.processor import preprocess_data
from src.features.feature_generator import generate_all_features
from src.models.fusion_model import PriceChangePredictor

# 日志配置，确保info级别能输出到终端
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger("model_trainer")

class ModelTrainer:
    def __init__(self, config):
        self.config = config
        self.model = PriceChangePredictor(
            prophet_params=config.get('prophet_params', {}),
            lgbm_params=config.get('lgbm_params', {})
        )

    def train(self):
        print(">>> Start training ...")
        try:
            logger.info("开始训练价格变动预测模型")
            # 1. 拉取数据
            datasets = fetch_training_data(
                train_end=self.config['train_end'],
                val_end=self.config['val_end'],
                test_end=self.config['test_end']
            )
            # 2. 数据预处理
            train_df = preprocess_data(datasets['train'])
            val_df = preprocess_data(datasets['val'])
            test_df = preprocess_data(datasets['test'])
            # 3. 特征工程
            train_feat = generate_all_features(train_df)
            val_feat = generate_all_features(val_df)
            test_feat = generate_all_features(test_df)
            # 4. 训练
            self.model.fit(train_feat, val_feat)
            # 5. 评估
            metrics = self.model.evaluate(test_feat)
            logger.info(f"模型评估结果: {metrics}")
            # 6. 保存模型
            self.save_model()
            logger.info("模型训练流程完成")
            return metrics
        except Exception as e:
            logger.error(f"模型训练流程失败: {str(e)}", exc_info=True)
            raise

    def save_model(self):
        os.makedirs('models', exist_ok=True)
        model_path = self.config.get('model_path', 'models/price_model.pkl')
        # 只保存底层sklearn模型
        joblib.dump(self.model.lgbm_model, model_path)
        logger.info(f"模型已保存: {model_path}")
        # 保存特征名列表
        features_path = model_path.replace('.pkl', '_features.json')
        import json
        with open(features_path, 'w') as f:
            json.dump(self.model.features, f, ensure_ascii=False)
        logger.info(f"特征名列表已保存: {features_path}")

if __name__ == "__main__":
    print(">>> Trainer main entry")
    config = {
        "train_end": "2024-12-31",
        "val_end": "2025-04-23",
        "test_end": "2025-04-30",
        "model_path": "models/price_model.pkl",
        "prophet_params": {},
        "lgbm_params": {}
        # 其它如数据库连接等参数
    }
    trainer = ModelTrainer(config)
    trainer.train()