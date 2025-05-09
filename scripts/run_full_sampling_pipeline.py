#!/usr/bin/env python
"""
商品价格智能采样策略完整测试流程
整合数据获取、SKU聚类和采样调度的全过程
"""
import os
import sys
import json
import logging
import argparse
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 确保可以导入项目模块
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.data.data_fetcher import DataFetcher
from src.features.sku_clusterer import SKUClusterer
from src.scheduling.sampling_scheduler import SamplingScheduler

# 配置日志
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/sampling_pipeline.log', mode='w')
    ]
)
logger = logging.getLogger(__name__)

def load_db_config():
    """加载数据库配置"""
    config_path = os.path.join(project_root, 'config', 'database.json')
    with open(config_path, 'r') as f:
        return json.load(f)

def prepare_data(args):
    """准备数据"""
    logger.info("=== 第一阶段: 数据准备 ===")
    
    # 检查是否已有数据文件
    if os.path.exists(args.data_path) and not args.force_data_update:
        logger.info(f"使用现有数据文件: {args.data_path}")
        return pd.read_csv(args.data_path)
    
    # 加载数据库配置
    db_config = load_db_config()
    
    # 创建数据获取器
    fetcher = DataFetcher(**db_config)
    
    try:
        # 生成数据集
        logger.info(f"从数据库获取数据: 开始日期={args.start_date}, 结束日期={args.end_date}")
        dataset = fetcher.generate_sampling_dataset(
            categories=args.categories,
            start_date=args.start_date,
            end_date=args.end_date,
            output_path=args.data_path
        )
        
        logger.info(f"数据准备完成，已保存到: {args.data_path}")
        return dataset
    finally:
        fetcher.close()

def perform_clustering(data, args):
    """执行SKU聚类"""
    logger.info("=== 第二阶段: SKU聚类 ===")
    
    # 创建聚类器
    clusterer = SKUClusterer(
        method='hierarchical',
        n_clusters=args.n_clusters
    )
    
    # 拟合聚类模型
    logger.info("开始SKU聚类...")
    clusterer.fit(data, sku_col='sku_id')
    
    # 获取聚类结果
    cluster_labels = clusterer.cluster_labels
    
    # 保存聚类结果
    output_dir = os.path.join(args.output_dir, 'clusters')
    clusterer.save_results(output_dir)
    
    # 获取推荐策略
    strategies = clusterer.recommend_sampling_strategies()
    logger.info(f"基于聚类结果推荐的策略:\n{strategies}")
    
    return cluster_labels, strategies

def generate_sampling_strategy(data, cluster_labels, cluster_strategies, args):
    """生成采样策略"""
    logger.info("=== 第三阶段: 采样策略生成 ===")
    
    # 划分训练集和测试集
    split_date = pd.to_datetime(args.split_date)
    train_mask = pd.to_datetime(data['ds']) < split_date
    train_data = data[train_mask].copy()
    test_data = data[~train_mask].copy()
    
    logger.info(f"数据划分: 训练集 {len(train_data)} 条记录, 测试集 {len(test_data)} 条记录")
    
    # 创建采样调度器
    scheduler = SamplingScheduler(
        capture_rate_target=args.capture_rate,
        max_samples_per_day=args.max_samples_per_day
    )
    scheduler.cluster_strategies = cluster_strategies
    
    # 使用训练集优化策略
    logger.info("使用训练集优化采样策略...")
    optimized_strategies = scheduler.optimize_cluster_strategies(
        train_data,
        cluster_labels,
        target_capture_rate=args.capture_rate
    )
    
    logger.info(f"优化后的策略:\n{optimized_strategies}")
    
    # 使用测试集评估策略
    logger.info("使用测试集评估优化后的策略...")
    evaluation_results = scheduler.evaluate_strategies(
        test_data,
        cluster_labels,
        output_dir=os.path.join(args.output_dir, 'evaluation')
    )
    
    # 记录评估结果
    capture_rate = evaluation_results['overall']['capture_rate']
    sampling_rate = evaluation_results['overall']['sampling_rate']
    efficiency = evaluation_results['overall']['efficiency']
    
    logger.info(f"策略评估结果: 捕获率={capture_rate:.4f}, 采样率={sampling_rate:.4f}, 效率={efficiency:.4f}")
    
    # 生成未来30天的采样计划
    logger.info("生成未来采样计划...")
    
    # 准备预测数据框架
    last_date = pd.to_datetime(data['ds']).max()
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=args.forecast_days,
        freq='D'
    )
    
    skus = data['sku_id'].unique()
    future_df = pd.DataFrame([
        {'ds': date, 'sku_id': sku, 'change_probability': 0.5}
        for date in future_dates
        for sku in skus
    ])
    
    # 生成采样计划
    sampling_schedule = scheduler.optimize_schedule(
        probabilities=future_df,
        sku_clusters=cluster_labels,
        days=args.forecast_days
    )
    
    # 保存采样计划
    schedule_path = os.path.join(args.output_dir, 'sampling_schedule.csv')
    scheduler.save_schedule(sampling_schedule, schedule_path)
    
    logger.info(f"采样计划已保存到: {schedule_path}")
    
    # 可视化采样计划
    visualize_sampling_schedule(sampling_schedule, future_df, args.output_dir)
    
    return scheduler, sampling_schedule

def visualize_sampling_schedule(schedule, probabilities, output_dir):
    """可视化采样计划"""
    logger.info("可视化采样计划...")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 每日采样数量图
    plt.figure(figsize=(12, 6))
    
    daily_samples = schedule.groupby('ds')['sample_flag'].sum().reset_index()
    plt.bar(daily_samples['ds'], daily_samples['sample_flag'])
    
    plt.title('每日采样数量')
    plt.xlabel('日期')
    plt.ylabel('采样数量')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, 'daily_samples.png'), dpi=300)
    plt.close()
    
    # 2. 采样原因分布
    plt.figure(figsize=(10, 6))
    
    reason_counts = schedule[schedule['sample_flag'] == 1]['sampling_reason'].value_counts()
    reason_counts.plot(kind='pie', autopct='%1.1f%%')
    
    plt.title('采样原因分布')
    plt.axis('equal')
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, 'sampling_reasons.png'), dpi=300)
    plt.close()
    
    # 3. 如果SKU数量适合，创建热图
    if len(schedule['sku_id'].unique()) <= 30:
        plt.figure(figsize=(16, 10))
        
        # 创建热图数据
        pivot_data = schedule.pivot_table(
            index='sku_id',
            columns='ds',
            values='sample_flag',
            aggfunc='max',
            fill_value=0
        )
        
        # 绘制采样计划热图
        sns.heatmap(pivot_data, cmap='Blues', cbar_kws={'label': '采样标志'})
        plt.title('SKU采样计划热图')
        plt.xlabel('日期')
        plt.ylabel('SKU ID')
        plt.tight_layout()
        
        plt.savefig(os.path.join(output_dir, 'sampling_heatmap.png'), dpi=300)
        plt.close()

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='运行商品价格智能采样策略完整测试流程')
    
    # 数据参数
    parser.add_argument('--data_path', type=str, default='data/processed/time_series_data.csv',
                       help='数据文件路径')
    parser.add_argument('--categories', type=str, nargs='+',
                       help='要包含的类别列表，不指定则使用所有类别')
    parser.add_argument('--start_date', type=str, default='2022-01-01',
                       help='数据开始日期 (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, default='2024-01-01',
                       help='数据结束日期 (YYYY-MM-DD)')
    parser.add_argument('--split_date', type=str, default='2023-10-01',
                       help='训练集和测试集分割日期 (YYYY-MM-DD)')
    parser.add_argument('--force_data_update', action='store_true',
                       help='强制更新数据，即使已存在')
    
    # 聚类参数
    parser.add_argument('--n_clusters', type=int, default=None,
                       help='聚类数量，None时自动确定')
    
    # 采样策略参数
    parser.add_argument('--capture_rate', type=float, default=0.95,
                       help='目标捕获率')
    parser.add_argument('--max_samples_per_day', type=int, default=None,
                       help='每日最大采样次数')
    parser.add_argument('--forecast_days', type=int, default=30,
                       help='预测天数')
    
    # 输出参数
    parser.add_argument('--output_dir', type=str, default='results/sampling_pipeline',
                       help='输出目录')
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # 1. 准备数据
        data = prepare_data(args)
        
        # 2. 执行SKU聚类
        cluster_labels, cluster_strategies = perform_clustering(data, args)
        
        # 3. 生成采样策略
        scheduler, sampling_schedule = generate_sampling_strategy(
            data, cluster_labels, cluster_strategies, args
        )
        
        logger.info("采样策略流程全部完成!")
        
    except Exception as e:
        logger.exception(f"流程执行出错: {str(e)}")
        return 1
        
    return 0

if __name__ == "__main__":
    sys.exit(main()) 