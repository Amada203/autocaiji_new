"""
价格变化采样策略实验
整合价格变化概率预测、SKU聚类和采样调度
"""
import os
import sys
import json
import argparse
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union

# 确保可以导入项目模块
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from models.price_model import PriceModel
from src.features.sku_clusterer import SKUClusterer
from src.scheduling.sampling_scheduler import SamplingScheduler
from src.utils.data_normalizer import DataNormalizer
from src.features.granger_causality import SKUCausalNetwork

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SamplingExperiment:
    """
    价格变化采样策略实验
    整合价格变化概率预测、SKU聚类和采样调度
    """
    
    def __init__(self, config_path):
        """
        初始化实验
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        
        # 加载配置
        with open(config_path, 'r') as f:
            self.config = json.load(f)
            
        # 创建输出目录
        self.output_dir = self.config.get('output_dir', 'experiments/runs/sampling_experiment')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 设置日志文件
        log_file = os.path.join(self.output_dir, 'experiment.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
        
        # 保存配置
        config_output = os.path.join(self.output_dir, 'config.json')
        with open(config_output, 'w') as f:
            json.dump(self.config, f, indent=2)
            
        # 初始化组件
        self.data = None
        self.train_data = None
        self.test_data = None
        self.probability_model = None
        self.clusterer = None
        self.scheduler = None
        self.causal_network = None
        
        logger.info(f"采样策略实验初始化完成，配置已保存到 {self.output_dir}")
        
    def load_data(self):
        """加载并处理数据"""
        data_path = self.config.get('data_path')
        if not data_path or not os.path.exists(data_path):
            raise ValueError(f"数据文件不存在: {data_path}")
            
        logger.info(f"从 {data_path} 加载数据...")
        self.data = pd.read_csv(data_path)
        
        # 确保必要的列存在
        required_cols = ['ds', 'y']
        if 'sku_id' in self.config.get('experiment', {}).get('required_columns', []):
            required_cols.append('sku_id')
            
        missing_cols = [col for col in required_cols if col not in self.data.columns]
        if missing_cols:
            raise ValueError(f"数据缺少必要的列: {missing_cols}")
            
        # 确保日期格式正确
        self.data['ds'] = pd.to_datetime(self.data['ds'])
        
        # 数据标准化（如果需要）
        if self.config.get('experiment', {}).get('normalize_data', False):
            normalize_method = self.config.get('experiment', {}).get('normalize_method', 'z-score')
            logger.info(f"使用 {normalize_method} 方法标准化数据...")
            
            normalizer = DataNormalizer(method=normalize_method)
            original_data = self.data.copy()
            
            # 只标准化y列
            y_values = self.data[['y']].values
            self.data['y'] = normalizer.fit_transform(y_values).flatten()
            
        # 划分训练集和测试集
        test_size = self.config.get('experiment', {}).get('test_size', 0.2)
        
        # 时间序列数据，使用最后一段时间作为测试集
        split_idx = int(len(self.data) * (1 - test_size))
        self.train_data = self.data.iloc[:split_idx].copy()
        self.test_data = self.data.iloc[split_idx:].copy()
        
        logger.info(f"数据加载完成。训练集: {len(self.train_data)}行, 测试集: {len(self.test_data)}行")
        
        # 计算价格变化标志（如果没有）
        change_threshold = self.config.get('experiment', {}).get('change_threshold', 0.01)
        if 'change_flag' not in self.data.columns:
            logger.info(f"计算价格变化标志，使用严格不等判定")
            
            # 按SKU分组计算（如果有sku_id列）
            if 'sku_id' in self.data.columns:
                self.data['prev_y'] = self.data.groupby('sku_id')['y'].shift(1)
            else:
                self.data['prev_y'] = self.data['y'].shift(1)
                
            # 计算变化比例（保留用于其他可能的分析）
            self.data['change_ratio'] = (self.data['y'] - self.data['prev_y']) / (self.data['prev_y'] + 1e-10)
            
            # 使用严格不等判定
            self.data['change_flag'] = (self.data['y'] != self.data['prev_y']).astype(int)
            
            # 更新训练集和测试集
            self.train_data = self.data.iloc[:split_idx].copy()
            self.test_data = self.data.iloc[split_idx:].copy()
            
        # 分析价格变化频率
        change_freq = self.data['change_flag'].mean()
        logger.info(f"数据中的价格变化频率: {change_freq:.4f}")
        
        return self.data
        
    def train_probability_model(self):
        """训练价格变化概率预测模型"""
        if self.train_data is None:
            self.load_data()
            
        logger.info("训练价格变化概率预测模型...")
        
        # 获取模型参数
        prophet_params = self.config.get('model_params', {}).get('prophet', {})
        lgbm_params = self.config.get('model_params', {}).get('lgbm', {})
        change_threshold = self.config.get('experiment', {}).get('change_threshold', 0.01)
        
        # 初始化并训练模型
        self.probability_model = PriceModel(
            prophet_params=prophet_params,
            lgbm_params=lgbm_params,
            change_threshold=change_threshold
        )
        
        self.probability_model.fit(self.train_data)
        
        # 评估模型
        metrics = self.probability_model.evaluate(self.test_data)
        logger.info(f"价格变化概率模型评估指标: {metrics}")
        
        # 保存评估结果
        metrics_path = os.path.join(self.output_dir, 'probability_model_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
            
        # 保存模型
        model_path = os.path.join(self.output_dir, 'probability_model.pkl')
        self.probability_model.save(model_path)
        
        logger.info(f"价格变化概率模型训练完成并保存到 {model_path}")
        
        # 绘制ROC曲线和PR曲线
        self._plot_probability_model_evaluation()
        
        return metrics
        
    def cluster_skus(self):
        """对SKU进行聚类"""
        if self.data is None:
            self.load_data()
            
        if 'sku_id' not in self.data.columns:
            logger.warning("数据中没有sku_id列，无法进行SKU聚类")
            return None
            
        logger.info("开始SKU聚类...")
        
        # 获取聚类参数
        cluster_params = self.config.get('clustering', {})
        method = cluster_params.get('method', 'hierarchical')
        n_clusters = cluster_params.get('n_clusters', None)
        max_clusters = cluster_params.get('max_clusters', 10)
        
        # 初始化并执行聚类
        self.clusterer = SKUClusterer(
            method=method,
            n_clusters=n_clusters,
            max_clusters=max_clusters
        )
        
        self.clusterer.fit(self.data)
        
        # 保存聚类结果
        cluster_dir = os.path.join(self.output_dir, 'sku_clusters')
        self.clusterer.save_results(cluster_dir)
        
        # 获取聚类映射
        cluster_labels = self.clusterer.cluster_labels
        
        logger.info(f"SKU聚类完成，共{len(set(cluster_labels.values()))}个簇")
        
        return cluster_labels
        
    def build_causal_network(self):
        """构建SKU因果关系网络"""
        if self.data is None:
            self.load_data()
            
        if 'sku_id' not in self.data.columns:
            logger.warning("数据中没有sku_id列，无法构建因果网络")
            return None
            
        logger.info("开始构建SKU因果关系网络...")
        
        # 获取因果网络参数
        max_lags = self.config.get('causal_network', {}).get('max_lags', 5)
        significance_level = self.config.get('causal_network', {}).get('significance_level', 0.05)
        
        # 初始化并拟合因果网络
        self.causal_network = SKUCausalNetwork(
            max_lags=max_lags,
            significance_level=significance_level
        )
        
        # 使用训练集构建网络
        self.causal_network.fit(self.train_data)
        
        # 获取哨兵SKU
        sentinel_skus = self.causal_network.get_sentinel_skus()
        correlation_network = self.causal_network.get_correlation_network()
        
        logger.info(f"已识别 {len(sentinel_skus)} 个哨兵SKU，影响 {len(correlation_network)} 个关系")
        
        # 可视化因果网络
        if self.config.get('visualization', {}).get('plot_causal_network', True):
            output_path = os.path.join(self.output_dir, 'sku_causal_network.png')
            self.causal_network.visualize_network(output_path)
            
        # 保存哨兵SKU和关联网络
        sentinels_path = os.path.join(self.output_dir, 'sentinel_skus.json')
        with open(sentinels_path, 'w') as f:
            json.dump(sentinel_skus, f, indent=2)
            
        network_path = os.path.join(self.output_dir, 'correlation_network.json')
        with open(network_path, 'w') as f:
            # 将键转换为字符串（JSON要求）
            str_network = {str(k): v for k, v in correlation_network.items()}
            json.dump(str_network, f, indent=2)
            
        logger.info(f"因果网络已构建完成，结果保存至 {self.output_dir}")
        
        return correlation_network
            
    def generate_sampling_schedule(self):
        """生成采样计划"""
        if self.probability_model is None:
            self.train_probability_model()
            
        logger.info("开始生成采样计划...")
        
        # 获取调度参数
        capture_rate_target = self.config.get('scheduling', {}).get('capture_rate_target', 0.95)
        max_samples_per_day = self.config.get('scheduling', {}).get('max_samples_per_day', None)
        forecast_days = self.config.get('scheduling', {}).get('forecast_days', 30)
        use_clustering = self.config.get('scheduling', {}).get('use_clustering', True)
        use_sentinel = self.config.get('scheduling', {}).get('use_sentinel', True)
        
        # 初始化调度器
        self.scheduler = SamplingScheduler(
            capture_rate_target=capture_rate_target,
            max_samples_per_day=max_samples_per_day
        )
        
        # 生成预测日期范围
        last_date = self.data['ds'].max()
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=forecast_days,
            freq='D'
        )
        
        # 创建未来日期DataFrame
        if 'sku_id' in self.data.columns:
            unique_skus = self.data['sku_id'].unique()
            future_df = pd.DataFrame({
                'ds': np.repeat(future_dates, len(unique_skus)),
                'sku_id': np.tile(unique_skus, len(future_dates))
            })
        else:
            future_df = pd.DataFrame({'ds': future_dates})
            
        # 预测价格变化概率
        probabilities = self.probability_model.predict_probability(future_df)
        
        # 获取SKU聚类（如果启用）
        sku_clusters = None
        if use_clustering and self.clusterer is not None and hasattr(self.clusterer, 'cluster_labels'):
            sku_clusters = self.clusterer.cluster_labels
            
        # 获取相关性网络（如果启用）
        correlation_network = None
        if use_sentinel and self.causal_network is not None:
            correlation_network = self.causal_network.get_correlation_network()
        
        # 生成采样计划
        if correlation_network and use_sentinel:
            # 使用哨兵SKU生成计划
            schedule = self.scheduler.generate_sentinel_plan(
                probabilities=probabilities,
                correlation_network=correlation_network,
                days=forecast_days
            )
        else:
            # 使用常规或基于聚类的计划
            schedule = self.scheduler.generate_schedule(
                probabilities=probabilities,
                sku_clusters=sku_clusters,
                days=forecast_days
            )
            
        # 评估采样计划
        capture_rate = self.scheduler.estimate_capture_rate(schedule)
        logger.info(f"采样计划生成完成，估计捕获率: {capture_rate:.4f}")
        
        # 保存采样计划
        schedule_path = os.path.join(self.output_dir, 'sampling_schedule.csv')
        self.scheduler.save_schedule(schedule, schedule_path)
        
        # 可视化采样计划
        if self.config.get('visualization', {}).get('plot_sampling_schedule', True):
            self._visualize_sampling_schedule(schedule, probabilities)
            
        return schedule
        
    def _plot_probability_model_evaluation(self):
        """绘制概率模型评估图"""
        if self.probability_model is None or self.test_data is None:
            return
            
        logger.info("绘制概率模型评估图...")
        
        # 获取测试集预测结果
        test_pred = self.probability_model.predict_probability(self.test_data)
        
        # 合并实际标签
        evaluation_df = pd.merge(
            test_pred,
            self.test_data[['ds', 'sku_id' if 'sku_id' in self.test_data.columns else None, 'change_flag']].dropna(),
            on=['ds', 'sku_id'] if 'sku_id' in self.test_data.columns else ['ds'],
            how='inner'
        )
        
        # 绘制实际变化率vs预测概率
        plt.figure(figsize=(12, 10))
        
        # 1. 概率分布
        plt.subplot(2, 2, 1)
        sns.histplot(evaluation_df, x='change_probability', hue='change_flag', bins=20, 
                   element='step', stat='probability')
        plt.title('变化概率分布')
        plt.xlabel('预测变化概率')
        plt.ylabel('频率')
        
        # 2. 预测概率的准确性
        plt.subplot(2, 2, 2)
        
        # 将概率分成10组
        evaluation_df['prob_bin'] = pd.cut(evaluation_df['change_probability'], bins=10)
        bin_stats = evaluation_df.groupby('prob_bin')['change_flag'].agg(['mean', 'count']).reset_index()
        bin_stats['bin_center'] = bin_stats['prob_bin'].apply(lambda x: x.mid)
        
        plt.scatter(bin_stats['bin_center'], bin_stats['mean'], s=bin_stats['count']/10, alpha=0.7)
        plt.plot([0, 1], [0, 1], 'r--')
        plt.title('预测概率校准')
        plt.xlabel('预测概率')
        plt.ylabel('实际变化率')
        plt.grid(True, alpha=0.3)
        
        # 3. 时间序列预测
        plt.subplot(2, 2, 3)
        
        # 按时间聚合
        time_series = evaluation_df.groupby('ds').agg({
            'change_flag': 'mean',
            'change_probability': 'mean'
        }).reset_index()
        
        plt.plot(time_series['ds'], time_series['change_flag'], 'b-', label='实际变化率')
        plt.plot(time_series['ds'], time_series['change_probability'], 'r-', label='预测概率')
        plt.title('时间序列预测')
        plt.xlabel('日期')
        plt.ylabel('变化率/概率')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 4. ROC曲线
        from sklearn.metrics import roc_curve, auc
        plt.subplot(2, 2, 4)
        
        fpr, tpr, _ = roc_curve(evaluation_df['change_flag'], evaluation_df['change_probability'])
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, 'b-', label=f'ROC曲线 (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.title('ROC曲线')
        plt.xlabel('假阳性率')
        plt.ylabel('真阳性率')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图表
        output_path = os.path.join(self.output_dir, 'probability_model_evaluation.png')
        plt.savefig(output_path, dpi=300)
        plt.close()
        
    def _visualize_sampling_schedule(self, schedule, probabilities):
        """可视化采样计划"""
        if schedule is None or probabilities is None:
            return
            
        logger.info("可视化采样计划...")
        
        # 合并概率数据
        schedule_viz = pd.merge(
            schedule,
            probabilities[['sku_id', 'ds', 'change_probability']],
            on=['sku_id', 'ds'],
            how='left'
        )
        
        # 计算每日采样统计
        daily_stats = schedule_viz.groupby('ds').agg({
            'sample_flag': 'sum',
            'change_probability': 'mean'
        }).reset_index()
        
        # 1. 每日采样计划概览
        plt.figure(figsize=(14, 10))
        
        plt.subplot(2, 1, 1)
        plt.bar(daily_stats['ds'], daily_stats['sample_flag'], label='每日采样数')
        plt.plot(daily_stats['ds'], daily_stats['change_probability'] * 100, 'r-', label='平均变化概率 (%)')
        plt.title('每日采样计划')
        plt.xlabel('日期')
        plt.ylabel('采样数/概率')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. 按原因分类的采样分布
        plt.subplot(2, 1, 2)
        reason_counts = schedule_viz[schedule_viz['sample_flag'] == 1]['sampling_reason'].value_counts()
        sns.barplot(x=reason_counts.index, y=reason_counts.values)
        plt.title('采样原因分布')
        plt.xlabel('采样原因')
        plt.ylabel('数量')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        # 保存图表
        output_path = os.path.join(self.output_dir, 'sampling_schedule_overview.png')
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        # 如果有SKU，绘制SKU级别的采样计划
        if 'sku_id' in schedule_viz.columns and len(schedule_viz['sku_id'].unique()) <= 20:
            plt.figure(figsize=(16, len(schedule_viz['sku_id'].unique()) * 0.5 + 2))
            
            # 创建热图数据
            pivot_data = schedule_viz.pivot_table(
                index='sku_id',
                columns='ds',
                values='sample_flag',
                aggfunc='max',
                fill_value=0
            )
            
            # 绘制采样计划热图
            sns.heatmap(pivot_data, cmap='Blues', cbar_kws={'label': '采样标志'})
            plt.title('SKU级别采样计划')
            plt.xlabel('日期')
            plt.ylabel('SKU ID')
            plt.tight_layout()
            
            # 保存图表
            output_path = os.path.join(self.output_dir, 'sku_sampling_heatmap.png')
            plt.savefig(output_path, dpi=300)
            plt.close()
            
    def run(self):
        """运行完整实验"""
        self.load_data()
        self.train_probability_model()
        
        if 'sku_id' in self.data.columns:
            self.cluster_skus()
            self.build_causal_network()
            
        self.generate_sampling_schedule()
        logger.info(f"采样策略实验完成，结果保存在 {self.output_dir}")
        
        return self.output_dir
            
def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='运行价格变化采样策略实验')
    parser.add_argument('--config', type=str, 
                       default='experiments/configs/sampling_experiment.json',
                       help='配置文件路径')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='输出目录，不指定则使用配置文件中的设置')
    
    args = parser.parse_args()
    
    # 检查配置文件是否存在
    if not os.path.exists(args.config):
        logger.error(f"配置文件 {args.config} 不存在")
        return 1
    
    # 初始化实验
    experiment = SamplingExperiment(args.config)
    
    # 如果指定了输出目录，则覆盖配置
    if args.output_dir:
        experiment.output_dir = args.output_dir
        os.makedirs(args.output_dir, exist_ok=True)
    
    # 运行实验
    try:
        experiment.run()
        logger.info("实验成功完成")
        return 0
    except Exception as e:
        logger.exception(f"实验运行出错: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 