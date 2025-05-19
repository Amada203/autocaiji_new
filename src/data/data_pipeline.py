import os
import sys
import logging
import pandas as pd
import joblib
from datetime import datetime
from typing import Dict, Any
from .data_fetcher import DataFetcher
from .mysql_writer import MySQLWriter
from ..features.feature_engineering import FeatureEngineering

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/data_pipeline.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class DataPipeline:
    def __init__(self):
        """初始化数据管道"""
        os.makedirs("data", exist_ok=True)
        os.makedirs("models", exist_ok=True)
        os.makedirs("logs", exist_ok=True)
        
        self.data_dir = "data"
        self.models_dir = "models"
        self.processed_data_path = os.path.join(self.data_dir, "processed_data.csv")
        self.model_path = os.path.join(self.models_dir, "fusion_model.pkl")
        
        # 初始化监控
        from src.utils.monitoring import start_monitoring_server
        start_monitoring_server(port=8000)
        self.stats = {
            "processed_records": 0,
            "predictions_generated": 0,
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "success": False,
            "errors": []
        }

    def run_full_pipeline(self, max_retries=3):
        """运行完整数据管道，支持自动恢复"""
        logger.info("启动数据管道")
        attempt = 0
        
        while attempt < max_retries:
            try:
                # 检查恢复点
                recovery_point = self._get_recovery_point()
                
                # 1. 数据获取
                if recovery_point <= 1:
                    logger.info(f"阶段1: 数据获取 (尝试 {attempt+1}/{max_retries})")
                    raw_data = self._fetch_data()
                    if raw_data.empty:
                        raise ValueError("获取的数据为空")
                    self._update_recovery_point(2)

                # 2. 数据处理
                if recovery_point <= 2:
                    logger.info(f"阶段2: 数据处理 (尝试 {attempt+1}/{max_retries})")
                    processed_data = self._process_data(raw_data)
                    processed_data.to_csv(self.processed_data_path, index=False)
                    self._update_recovery_point(3)

                # 3. 模型训练
                if recovery_point <= 3:
                    logger.info(f"阶段3: 模型训练 (尝试 {attempt+1}/{max_retries})")
                    model = self._train_model(processed_data)
                    self._update_recovery_point(4)

                # 4. 生成预测
                if recovery_point <= 4:
                    logger.info(f"阶段4: 生成预测 (尝试 {attempt+1}/{max_retries})")
                    predictions = self._generate_predictions(processed_data, model)
                    self._update_recovery_point(5)

                # 5. 存储结果
                logger.info(f"阶段5: 存储结果 (尝试 {attempt+1}/{max_retries})")
                self._save_results(predictions, raw_data)
                
                # 成功完成则清除恢复点
                self._clear_recovery_point()
                logger.info("数据管道执行成功")
                self.stats["end_time"] = datetime.now().isoformat()
                self.stats["success"] = True
            
                # 发送成功通知
                from src.utils.notifier import send_pipeline_notification
                send_pipeline_notification(True, self.get_processing_stats())
            
                return True
                
            except Exception as e:
                logger.error(f"阶段执行失败: {str(e)}")
                attempt += 1
                if attempt >= max_retries:
                    logger.error(f"经过 {max_retries} 次尝试后仍失败")
                    self.stats["end_time"] = datetime.now().isoformat()
                    self.stats["success"] = False
            
                    # 发送失败通知
                    from src.utils.notifier import send_pipeline_notification
                    send_pipeline_notification(False, self.get_processing_stats())
            
                    return False
                logger.info(f"等待 {attempt*10} 秒后重试...")
                time.sleep(attempt * 10)

    def _fetch_data(self) -> pd.DataFrame:
        """获取原始数据"""
        try:
            fetcher = DataFetcher()
            return fetcher.fetch_training_data()
        except Exception as e:
            logger.error(f"数据获取失败: {str(e)}")
            return pd.DataFrame()

    def _process_data(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """处理原始数据"""
        processor = FeatureEngineering(raw_data)
        return processor.process()

    def _train_model(self, data: pd.DataFrame):
        """训练模型，支持模型缓存"""
        try:
            from src.models.fusion_model import ProphetLGBMFusion
            
            # 检查是否有可用的缓存模型
            if os.path.exists(self.model_path):
                model_mtime = os.path.getmtime(self.model_path)
                data_mtime = os.path.getmtime(self.processed_data_path)
                
                # 如果模型比数据新，则直接加载
                if model_mtime > data_mtime:
                    logger.info("加载缓存的模型...")
                    model = joblib.load(self.model_path)
                    return model
            
            logger.info("开始训练新模型...")
            model = ProphetLGBMFusion()
            model.fit(data)
            
            # 保存模型
            logger.info(f"保存模型到 {self.model_path}")
            joblib.dump(model, self.model_path)
            
            return model
            
        except Exception as e:
            error_msg = f"模型训练失败: {str(e)}"
            logger.error(error_msg)
            self.stats["errors"].append(error_msg)
            raise

    def _generate_predictions(self, data: pd.DataFrame, model):
        """生成预测"""
        try:
            logger.info("开始生成预测...")
            
            # 使用模型生成预测
            forecast = model.predict(data)
            
            # 创建预测结果DataFrame，确保列名与数据库表结构完全匹配
            predictions_df = pd.DataFrame({
                'sku': data['sku_id'].values,  # 使用'sku'作为列名
                'price': forecast['yhat'].values,  # 使用'price'作为列名
                'date': datetime.now(),  # 使用'date'作为列名
                'confidence': 0.9
            })
            
            self.stats["predictions_generated"] = len(predictions_df)
            logger.info(f"成功生成 {len(predictions_df)} 条预测")
            
            return predictions_df
            
        except Exception as e:
            error_msg = f"预测生成失败: {str(e)}"
            logger.error(error_msg)
            self.stats["errors"].append(error_msg)
            raise

    def _save_results(self, predictions: pd.DataFrame, raw_data: pd.DataFrame):
        """保存结果到MySQL"""
        try:
            writer = MySQLWriter()
            
            # 保存预测结果
            logger.info("保存预测结果到数据库...")
            if not writer.write_predictions(predictions):
                raise Exception("写入预测结果失败")
                
            # 保存历史数据
            logger.info("保存历史数据到数据库...")
            if not writer.write_history(raw_data):
                raise Exception("写入历史数据失败")
                
            self.stats["success"] = True
            self.stats["end_time"] = datetime.now().isoformat()
            
        except Exception as e:
            error_msg = f"结果保存失败: {str(e)}"
            logger.error(error_msg)
            self.stats["errors"].append(error_msg)
            raise

    def get_processing_stats(self) -> Dict[str, Any]:
        """获取处理统计信息"""
        # 添加内存使用信息
        process = psutil.Process(os.getpid())
        self.stats["memory_usage_mb"] = process.memory_info().rss / 1024 / 1024
        self.stats["cpu_usage"] = process.cpu_percent()
        
        # 记录到单独的性能日志
        perf_log = {
            "timestamp": datetime.now().isoformat(),
            "stats": self.stats
        }
        with open("logs/performance.log", "a") as f:
            f.write(json.dumps(perf_log) + "\n")
            
        # 更新监控指标
        from src.utils.monitoring import update_metrics
        update_metrics(self.stats)
            
        return self.stats

    def _get_recovery_point(self) -> int:
        """获取当前恢复点"""
        recovery_file = "logs/recovery_point.json"
        try:
            with open(recovery_file, "r") as f:
                data = json.load(f)
                return data.get("recovery_point", 1)
        except (FileNotFoundError, json.JSONDecodeError):
            return 1

    def _update_recovery_point(self, point: int):
        """更新恢复点"""
        recovery_file = "logs/recovery_point.json"
        data = {"recovery_point": point}
        with open(recovery_file, "w") as f:
            json.dump(data, f)

    def _clear_recovery_point(self):
        """清除恢复点"""
        recovery_file = "logs/recovery_point.json"
        if os.path.exists(recovery_file):
            os.remove(recovery_file)

if __name__ == "__main__":
    pipeline = DataPipeline()
    success = pipeline.run_full_pipeline()
    sys.exit(0 if success else 1)