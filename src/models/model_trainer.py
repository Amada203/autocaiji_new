import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import os
import sys
import logging

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
    def __init__(self, data_path='data/processed_data.csv'):
        """初始化模型训练器"""
        self.data_path = data_path
        self.model = None
        self.features = None
        self.target = 'significant_change'
        
    def train(self):
        """训练模型"""
        try:
            # 1. 加载数据
            logger.info(f"加载数据: {self.data_path}")
            df = pd.read_csv(self.data_path)
            
            # 2. 准备特征和目标
            self._prepare_features(df)
            
            # 3. 划分训练集和测试集
            X_train, X_test, y_train, y_test = train_test_split(
                self.X, self.y, test_size=0.2, random_state=42
            )
            
            # 4. 特征标准化
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # 5. 训练模型
            logger.info("开始训练模型")
            self.model = LGBMClassifier(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=7,
                random_state=42
            )
            self.model.fit(X_train_scaled, y_train)
            
            # 6. 评估模型
            self._evaluate_model(X_test_scaled, y_test)
            
            # 7. 保存模型
            self._save_model(scaler)
            
            return self.model
            
        except Exception as e:
            logger.error(f"模型训练失败: {str(e)}")
            return None
    
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