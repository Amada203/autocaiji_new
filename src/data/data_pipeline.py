import os
import sys
import json
import time
import psutil
import socket
import logging
import pandas as pd
import joblib
from datetime import datetime
from typing import Dict, Any
from .data_fetcher import DataFetcher
from .mysql_writer import MySQLWriter
from ..features.feature_engineering import FeatureEngineer

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
        try:
            self.monitor_port = start_monitoring_server(start_port=8000)
            logger.info(f"监控服务器启动在端口 {self.monitor_port}")
        except Exception as e:
            logger.error(f"无法启动监控服务器: {str(e)}")
            self.monitor_port = None
            
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
        
        # 初始化变量
        raw_data = None
        processed_data = None
        model = None
        predictions = None
        
        while attempt < max_retries:
            try:
                # 检查恢复点
                recovery_point = self._get_recovery_point()
                
                # 如果从高级恢复点开始，确保必要的数据存在
                if recovery_point > 1 and raw_data is None:
                    logger.info("从较高恢复点启动，重新获取数据...")
                    recovery_point = 1  # 强制从数据获取开始
                
                # 1. 数据获取
                if recovery_point <= 1:
                    logger.info(f"阶段1: 数据获取 (尝试 {attempt+1}/{max_retries})")
                    raw_data = self._fetch_data()
                    if raw_data is None or raw_data.empty:
                        raise ValueError("获取的数据为空")
                    self._update_recovery_point(2)

                # 2. 数据处理
                if recovery_point <= 2:
                    if raw_data is None or raw_data.empty:
                        raise ValueError("缺少原始数据，无法进行处理")
                    logger.info(f"阶段2: 数据处理 (尝试 {attempt+1}/{max_retries})")
                    processed_data = self._process_data(raw_data)
                    if processed_data is None or processed_data.empty:
                        raise ValueError("数据处理结果为空")
                    processed_data.to_csv(self.processed_data_path, index=False)
                    self._update_recovery_point(3)

                # 3. 模型训练
                if recovery_point <= 3:
                    if processed_data is None or processed_data.empty:
                        raise ValueError("缺少处理后的数据，无法训练模型")
                    logger.info(f"阶段3: 模型训练 (尝试 {attempt+1}/{max_retries})")
                    model = self._train_model(processed_data)
                    if model is None:
                        raise ValueError("模型训练失败")
                    self._update_recovery_point(4)

                # 4. 生成预测
                if recovery_point <= 4:
                    if processed_data is None or model is None:
                        raise ValueError("缺少必要数据或模型，无法生成预测")
                    logger.info(f"阶段4: 生成预测 (尝试 {attempt+1}/{max_retries})")
                    predictions = self._generate_predictions(processed_data, model)
                    if predictions is None or predictions.empty:
                        raise ValueError("预测生成失败")
                    self._update_recovery_point(5)

                # 5. 存储结果
                if predictions is None or raw_data is None:
                    raise ValueError("缺少必要数据，无法保存结果")
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
        processor = None
        try:
            # 记录原始数据统计信息
            self._log_data_stats(raw_data, "原始数据")
            
            # 处理数据
            from src.data.data_processor import DataProcessor
            processor = DataProcessor(raw_data)
            processed_data = processor.process()
            
            # 验证处理结果
            if processed_data is None or processed_data.empty:
                raise ValueError("数据处理结果为空")
                
            # 记录处理后数据统计信息
            self._log_data_stats(processed_data, "处理后数据")
            
            return processed_data
            
        except Exception as e:
            error_msg = f"数据处理失败: {str(e)}"
            if processor and hasattr(processor, 'df'):
                error_msg += f", 部分处理数据形状: {processor.df.shape}"
            logger.error(error_msg)
            raise ValueError(error_msg)
            
    def _log_data_stats(self, df: pd.DataFrame, stage: str):
        """记录数据统计信息"""
        if df is None or df.empty:
            logger.warning(f"{stage}: 无数据")
            return
            
        stats = {
            "记录数": len(df),
            "列数": len(df.columns),
            "开始日期": df['dt'].min(),
            "结束日期": df['dt'].max(),
            "SKU数量": df['sku_id'].nunique(),
            "平均价格": df['page_price'].mean(),
            "价格标准差": df['page_price'].std()
        }
        
        logger.info(f"{stage}统计: {json.dumps(stats, indent=2, default=str)}")

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

    def _save_results(self, predictions: pd.DataFrame, raw_data: pd.DataFrame, rebuild_tables=False):
        """
        保存结果到MySQL
        
        Args:
            predictions: 预测结果DataFrame
            raw_data: 原始数据DataFrame
            rebuild_tables: 是否重建表，默认False（使用增量更新模式）
        """
        try:
            # 从配置文件加载数据库配置
            db_config = {}
            try:
                with open("config/database.json", "r") as f:
                    config = json.load(f)
                    db_config = {
                        'host': config['mysql']['host'],
                        'port': config['mysql']['port'],
                        'user': config['mysql']['user'],
                        'password': config['mysql']['password'],
                        'database': config['mysql']['database']
                    }
            except Exception as e:
                logger.warning(f"加载数据库配置失败，使用默认配置: {str(e)}")
            
            # 创建MySQL写入器
            writer = MySQLWriter(**db_config)
            
            # 保存预测结果（使用增量更新模式）
            logger.info(f"保存预测结果到数据库... (模式: {'重建表' if rebuild_tables else '增量更新'})")
            if not writer.write_predictions(predictions, rebuild_table=rebuild_tables):
                raise Exception("写入预测结果失败")
                
            # 保存历史数据（使用增量更新模式）
            logger.info(f"保存历史数据到数据库... (模式: {'重建表' if rebuild_tables else '增量更新'})")
            logger.info("测试模式：仅处理前50条记录")
            if not writer.write_history(raw_data, rebuild_table=rebuild_tables, limit=50):
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