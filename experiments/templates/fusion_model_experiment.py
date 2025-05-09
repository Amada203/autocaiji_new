"""
Prophet+LightGBM融合模型实验模板
"""
import os
import sys
import json
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union
import pickle

# 确保可以导入项目模块
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

from models.prophet_model import ProphetModel
from models.fusion_model import ProphetLGBMFusion
from src.utils.data_normalizer import DataNormalizer
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import seaborn as sns

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FusionModelExperiment:
    """
    Prophet+LightGBM融合模型实验类
    """
    
    def __init__(self, config_path):
        """
        初始化实验
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        self.config = self._load_config()
        
        # 创建输出目录
        self.output_dir = self.config.get('output_dir', 'experiments/runs/fusion_model_experiment')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 设置日志文件
        log_file = os.path.join(self.output_dir, 'experiment.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
        
        # 保存配置
        self._save_config()
        
        # 初始化模型和数据
        self.data = None
        self.train_data = None
        self.test_data = None
        self.normalizer = None
        self.prophet_model = None
        self.lgbm_model = None
        self.fusion_model = None
        
        logger.info(f"实验初始化完成，配置已保存到 {self.output_dir}")
        
    def _load_config(self):
        """加载配置文件"""
        with open(self.config_path, 'r') as f:
            return json.load(f)
            
    def _save_config(self):
        """保存配置到输出目录"""
        config_output = os.path.join(self.output_dir, 'config.json')
        with open(config_output, 'w') as f:
            json.dump(self.config, f, indent=2)
            
    def load_data(self):
        """加载并处理数据"""
        data_path = self.config.get('data_path')
        if not data_path or not os.path.exists(data_path):
            # 如果数据文件不存在，生成模拟数据
            logger.info(f"数据文件 {data_path} 不存在，生成模拟数据...")
            self._generate_sample_data()
        else:
            # 加载数据
            logger.info(f"从 {data_path} 加载数据...")
            self.data = pd.read_csv(data_path)
            
        # 确保列名正确
        required_cols = ['ds', 'y']
        if not all(col in self.data.columns for col in required_cols):
            raise ValueError(f"数据必须包含以下列: {required_cols}")
            
        # 确保日期格式正确
        self.data['ds'] = pd.to_datetime(self.data['ds'])
        
        # 标准化数据
        if self.config.get('experiment', {}).get('normalize_data', False):
            normalize_method = self.config.get('experiment', {}).get('normalize_method', 'z-score')
            logger.info(f"使用 {normalize_method} 方法标准化数据...")
            
            self.normalizer = DataNormalizer(method=normalize_method)
            original_data = self.data.copy()
            
            # 只标准化y列
            y_values = self.data[['y']].values
            self.data['y'] = self.normalizer.fit_transform(y_values).flatten()
            
            # 保存原始数据和标准化数据的对比图
            self._plot_normalization_comparison(original_data, self.data)
            
        # 划分训练集和测试集
        test_size = self.config.get('experiment', {}).get('test_size', 0.2)
        random_state = self.config.get('experiment', {}).get('random_state', 42)
        
        # 时间序列数据，我们使用最后一段时间作为测试集
        split_idx = int(len(self.data) * (1 - test_size))
        self.train_data = self.data.iloc[:split_idx].copy()
        self.test_data = self.data.iloc[split_idx:].copy()
        
        logger.info(f"数据加载完成。训练集: {len(self.train_data)}行, 测试集: {len(self.test_data)}行")
        
    def _generate_sample_data(self):
        """生成样本数据用于测试"""
        # 确保输出目录存在
        data_dir = os.path.dirname(self.config.get('data_path', 'data/processed/time_series_data.csv'))
        os.makedirs(data_dir, exist_ok=True)
        
        # 生成日期序列
        periods = 365 * 2  # 两年数据
        end_date = datetime.now()
        dates = pd.date_range(end=end_date, periods=periods)
        
        # 生成基本趋势
        t = np.arange(len(dates))
        trend = 100 + 0.1 * t
        
        # 添加季节性
        annual_seasonality = 20 * np.sin(2 * np.pi * t / 365)
        weekly_seasonality = 5 * np.sin(2 * np.pi * t / 7)
        
        # 添加噪声
        noise = np.random.normal(0, 5, size=len(dates))
        
        # 组合成最终价格
        y = trend + annual_seasonality + weekly_seasonality + noise
        
        # 创建数据框
        self.data = pd.DataFrame({
            'ds': dates,
            'y': y
        })
        
        # 保存到文件
        output_path = self.config.get('data_path', 'data/processed/time_series_data.csv')
        self.data.to_csv(output_path, index=False)
        logger.info(f"生成的样本数据已保存到 {output_path}")
        
    def _plot_normalization_comparison(self, original_data, normalized_data):
        """绘制标准化前后的数据对比图"""
        plt.figure(figsize=(15, 6))
        
        # 绘制原始数据
        plt.subplot(1, 2, 1)
        plt.plot(original_data['ds'], original_data['y'], 'b-')
        plt.title('原始数据')
        plt.xlabel('日期')
        plt.ylabel('值')
        plt.grid(True, alpha=0.3)
        
        # 绘制标准化后数据
        plt.subplot(1, 2, 2)
        plt.plot(normalized_data['ds'], normalized_data['y'], 'r-')
        plt.title('标准化后数据')
        plt.xlabel('日期')
        plt.ylabel('标准化值')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'normalization_comparison.png'), dpi=300)
        plt.close()
        
        # 分布对比
        plt.figure(figsize=(15, 6))
        
        plt.subplot(1, 2, 1)
        sns.histplot(original_data['y'], kde=True)
        plt.title('原始数据分布')
        plt.xlabel('值')
        
        plt.subplot(1, 2, 2)
        sns.histplot(normalized_data['y'], kde=True)
        plt.title('标准化后数据分布')
        plt.xlabel('标准化值')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'normalization_distribution.png'), dpi=300)
        plt.close()
        
    def run_prophet_model(self):
        """运行Prophet模型"""
        if self.train_data is None:
            raise ValueError("请先加载数据")
            
        logger.info("训练Prophet模型...")
        prophet_params = self.config.get('model_params', {}).get('prophet', {})
        self.prophet_model = ProphetModel(**prophet_params)
        self.prophet_model.fit(self.train_data)
        
        # 生成预测
        logger.info("生成Prophet预测...")
        future_periods = self.config.get('experiment', {}).get('forecast_periods', 30)
        future = self.prophet_model.model.make_future_dataframe(periods=future_periods)
        forecast = self.prophet_model.predict(future)
        
        # 评估模型
        metrics = self._evaluate_forecast(forecast, self.test_data, 'prophet')
        logger.info(f"Prophet模型评估指标: {metrics}")
        
        # 保存预测结果
        self._save_forecast(forecast, 'prophet')
        
        # 可视化
        if self.config.get('visualization', {}).get('save_plots', True):
            self._plot_prophet_results(forecast)
            
        return metrics
        
    def run_fusion_model(self):
        """运行融合模型"""
        if self.train_data is None:
            raise ValueError("请先加载数据")
            
        logger.info("训练Prophet+LightGBM融合模型...")
        prophet_params = self.config.get('model_params', {}).get('prophet', {})
        lgbm_params = self.config.get('model_params', {}).get('lgbm', {})
        
        self.fusion_model = ProphetLGBMFusion(
            prophet_params=prophet_params,
            lgbm_params=lgbm_params
        )
        self.fusion_model.fit(self.train_data)
        
        # 生成预测
        logger.info("生成融合模型预测...")
        future_periods = self.config.get('experiment', {}).get('forecast_periods', 30)
        future = pd.DataFrame({
            'ds': pd.date_range(
                start=self.train_data['ds'].max() + pd.Timedelta(days=1),
                periods=future_periods
            )
        })
        # 合并历史数据和未来数据
        future_all = pd.concat([self.data[['ds']], future]).drop_duplicates().sort_values('ds')
        forecast = self.fusion_model.predict(future_all)
        
        # 评估模型
        metrics = self._evaluate_forecast(forecast, self.test_data, 'fusion')
        logger.info(f"融合模型评估指标: {metrics}")
        
        # 保存预测结果
        self._save_forecast(forecast, 'fusion')
        
        # 可视化
        if self.config.get('visualization', {}).get('save_plots', True):
            self._plot_fusion_results(forecast)
            
        return metrics
        
    def _evaluate_forecast(self, forecast, test_data, model_name):
        """评估预测结果"""
        # 合并预测结果和测试数据
        merged = pd.merge(forecast, test_data, on='ds', how='inner')
        
        if len(merged) == 0:
            logger.warning("测试数据和预测结果没有重叠部分，无法评估")
            return {}
            
        # 如果数据已标准化，需要反标准化
        y_true = merged['y'].values
        y_pred = merged['yhat'].values
        
        if self.normalizer is not None:
            y_true = self.normalizer.inverse_transform(y_true.reshape(-1, 1)).flatten()
            y_pred = self.normalizer.inverse_transform(y_pred.reshape(-1, 1)).flatten()
            
        # 计算指标
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        metrics = {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2
        }
        
        # 保存指标到文件
        metrics_file = os.path.join(self.output_dir, f'{model_name}_metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
            
        return metrics
        
    def _save_forecast(self, forecast, model_name):
        """保存预测结果"""
        # 如果数据已标准化，需要反标准化
        if self.normalizer is not None:
            forecast_copy = forecast.copy()
            forecast_copy['yhat'] = self.normalizer.inverse_transform(
                forecast_copy['yhat'].values.reshape(-1, 1)
            ).flatten()
            
            if 'yhat_lower' in forecast_copy.columns:
                forecast_copy['yhat_lower'] = self.normalizer.inverse_transform(
                    forecast_copy['yhat_lower'].values.reshape(-1, 1)
                ).flatten()
                
            if 'yhat_upper' in forecast_copy.columns:
                forecast_copy['yhat_upper'] = self.normalizer.inverse_transform(
                    forecast_copy['yhat_upper'].values.reshape(-1, 1)
                ).flatten()
        else:
            forecast_copy = forecast
            
        # 保存到CSV
        forecast_file = os.path.join(self.output_dir, f'{model_name}_forecast.csv')
        forecast_copy.to_csv(forecast_file, index=False)
        logger.info(f"{model_name}预测结果已保存到 {forecast_file}")
        
    def _plot_prophet_results(self, forecast):
        """绘制Prophet模型结果"""
        # 绘制预测结果
        fig = self.prophet_model.model.plot(forecast)
        fig.savefig(os.path.join(self.output_dir, 'prophet_forecast.png'), dpi=300)
        plt.close(fig)
        
        # 绘制组件图
        if self.config.get('visualization', {}).get('plot_prophet_components', True):
            fig = self.prophet_model.model.plot_components(forecast)
            fig.savefig(os.path.join(self.output_dir, 'prophet_components.png'), dpi=300)
            plt.close(fig)
            
    def _plot_fusion_results(self, forecast):
        """绘制融合模型结果"""
        # 绘制预测结果对比
        plt.figure(figsize=(15, 8))
        
        # 合并原始数据和预测结果
        merged = pd.merge(self.data, forecast[['ds', 'yhat', 'yhat_original']], on='ds', how='left')
        
        # 如果数据已标准化，需要反标准化
        if self.normalizer is not None:
            merged['y_orig'] = self.normalizer.inverse_transform(merged['y'].values.reshape(-1, 1)).flatten()
            merged['yhat_orig'] = self.normalizer.inverse_transform(merged['yhat'].values.reshape(-1, 1)).flatten()
            merged['yhat_prophet_orig'] = self.normalizer.inverse_transform(merged['yhat_original'].values.reshape(-1, 1)).flatten()
        else:
            merged['y_orig'] = merged['y']
            merged['yhat_orig'] = merged['yhat']
            merged['yhat_prophet_orig'] = merged['yhat_original']
            
        # 绘制实际值和预测值
        plt.plot(merged['ds'], merged['y_orig'], 'b-', label='实际值')
        plt.plot(merged['ds'], merged['yhat_orig'], 'r-', label='融合模型预测')
        plt.plot(merged['ds'], merged['yhat_prophet_orig'], 'g--', label='Prophet预测')
        
        # 标记训练集和测试集分界线
        train_end = self.train_data['ds'].max()
        plt.axvline(x=train_end, color='gray', linestyle='--')
        plt.text(train_end, plt.ylim()[1] * 0.9, '训练集/测试集分界', rotation=90, va='top')
        
        plt.title('Prophet+LightGBM融合模型预测结果')
        plt.xlabel('日期')
        plt.ylabel('值')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'fusion_forecast.png'), dpi=300)
        plt.close()
        
        # 绘制残差分析
        if self.config.get('visualization', {}).get('plot_residuals', True):
            plt.figure(figsize=(15, 8))
            
            # 计算残差
            residuals = merged['yhat'] - merged['yhat_original']
            
            plt.subplot(2, 1, 1)
            plt.plot(merged['ds'], residuals, 'b-')
            plt.title('LightGBM残差预测')
            plt.xlabel('日期')
            plt.ylabel('残差')
            plt.grid(True, alpha=0.3)
            
            plt.subplot(2, 1, 2)
            sns.histplot(residuals.dropna(), kde=True)
            plt.title('残差分布')
            plt.xlabel('残差值')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'residuals_analysis.png'), dpi=300)
            plt.close()
            
        # 绘制特征重要性
        if self.config.get('visualization', {}).get('plot_feature_importance', True) and hasattr(self.fusion_model, 'lgbm_model'):
            plt.figure(figsize=(12, 8))
            
            lgbm_model = self.fusion_model.lgbm_model
            feature_importance = pd.DataFrame(
                {'feature': self.fusion_model.feature_names,
                 'importance': lgbm_model.feature_importances_}
            ).sort_values('importance', ascending=False)
            
            sns.barplot(x='importance', y='feature', data=feature_importance)
            plt.title('LightGBM特征重要性')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'feature_importance.png'), dpi=300)
            plt.close()
            
            # 保存特征重要性到CSV
            feature_importance.to_csv(os.path.join(self.output_dir, 'feature_importance.csv'), index=False)
            
    def run_comparison(self):
        """运行模型比较实验"""
        # 加载数据
        self.load_data()
        
        results = {}
        
        # 根据配置运行不同模型
        comparison_config = self.config.get('experiment', {}).get('comparison', {})
        
        if comparison_config.get('run_prophet_only', True):
            logger.info("运行Prophet单独模型...")
            prophet_metrics = self.run_prophet_model()
            results['prophet'] = prophet_metrics
            
        if comparison_config.get('run_fusion', True):
            logger.info("运行Prophet+LightGBM融合模型...")
            fusion_metrics = self.run_fusion_model()
            results['fusion'] = fusion_metrics
            
        # 比较结果并生成报告
        self._generate_comparison_report(results)
        
        logger.info("模型比较实验完成")
        return results
        
    def _generate_comparison_report(self, results):
        """生成模型比较报告"""
        if not results:
            logger.warning("没有模型结果可比较")
            return
            
        # 创建比较表格
        comparison_df = pd.DataFrame(results).T
        
        # 保存到CSV
        comparison_file = os.path.join(self.output_dir, 'model_comparison.csv')
        comparison_df.to_csv(comparison_file)
        
        # 生成可视化比较
        self._plot_model_comparison(comparison_df)
        
        # 生成文本报告
        report_file = os.path.join(self.output_dir, 'comparison_report.txt')
        with open(report_file, 'w') as f:
            f.write("模型比较报告\n")
            f.write("=" * 50 + "\n\n")
            
            # 写入表格
            f.write(comparison_df.to_string() + "\n\n")
            
            # 如果有融合模型，计算改进
            if 'prophet' in results and 'fusion' in results:
                improvement_rmse = (results['prophet']['rmse'] - results['fusion']['rmse']) / results['prophet']['rmse'] * 100
                improvement_mae = (results['prophet']['mae'] - results['fusion']['mae']) / results['prophet']['mae'] * 100
                
                f.write(f"融合模型相对于Prophet的改进:\n")
                f.write(f"RMSE: {improvement_rmse:.2f}%\n")
                f.write(f"MAE: {improvement_mae:.2f}%\n\n")
                
            f.write("模型参数:\n")
            f.write("Prophet: " + json.dumps(self.config.get('model_params', {}).get('prophet', {}), indent=2) + "\n")
            f.write("LightGBM: " + json.dumps(self.config.get('model_params', {}).get('lgbm', {}), indent=2) + "\n")
            
        logger.info(f"比较报告已保存到 {report_file}")
        
    def _plot_model_comparison(self, comparison_df):
        """绘制模型比较图"""
        metrics = ['rmse', 'mae']
        
        plt.figure(figsize=(12, 6))
        
        for i, metric in enumerate(metrics):
            plt.subplot(1, 2, i+1)
            sns.barplot(x=comparison_df.index, y=comparison_df[metric])
            plt.title(f'{metric.upper()} 比较')
            plt.ylabel(metric.upper())
            
            # 添加数值标签
            for j, v in enumerate(comparison_df[metric]):
                plt.text(j, v + 0.01, f"{v:.4f}", ha='center')
                
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'metrics_comparison.png'), dpi=300)
        plt.close()
        
    def save_models(self):
        """保存训练好的模型"""
        if self.prophet_model is not None:
            prophet_path = os.path.join(self.output_dir, 'prophet_model.pkl')
            self.prophet_model.save(prophet_path)
            logger.info(f"Prophet模型已保存到 {prophet_path}")
            
        if self.fusion_model is not None:
            fusion_path = os.path.join(self.output_dir, 'fusion_model.pkl')
            self.fusion_model.save(fusion_path)
            logger.info(f"融合模型已保存到 {fusion_path}")
            
    def run(self):
        """运行完整实验流程"""
        try:
            logger.info(f"开始运行Prophet+LightGBM融合模型实验...")
            
            # 运行模型比较
            results = self.run_comparison()
            
            # 保存模型
            self.save_models()
            
            logger.info(f"实验完成，结果已保存到 {self.output_dir}")
            return results
            
        except Exception as e:
            logger.exception(f"实验运行出错: {str(e)}")
            raise 