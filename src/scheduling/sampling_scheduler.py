"""
采样调度器
根据价格变化概率生成最优采样计划，同时考虑采样成本和捕获率
"""
import pandas as pd
import numpy as np
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union

logger = logging.getLogger(__name__)

class SamplingScheduler:
    """
    采样调度器
    基于价格变化概率生成采样计划，平衡采样成本和捕获率
    """
    
    def __init__(self, capture_rate_target=0.95, max_samples_per_day=None):
        """
        初始化采样调度器
        
        Args:
            capture_rate_target: 目标捕获率，默认0.95（即95%）
            max_samples_per_day: 每日最大采样次数，None表示不限制
        """
        self.capture_rate_target = capture_rate_target
        self.max_samples_per_day = max_samples_per_day
        self.strategy_map = None
        self.cluster_strategies = None
        
    def generate_schedule(self, 
                         probabilities: pd.DataFrame, 
                         sku_clusters: Optional[Dict] = None,
                         start_date: Optional[datetime] = None,
                         days: int = 7) -> pd.DataFrame:
        """
        生成采样计划
        
        Args:
            probabilities: 含'ds', 'sku_id', 'change_probability'列的DataFrame
            sku_clusters: SKU到簇的映射，格式: {sku_id: cluster_label}
            start_date: 计划开始日期，None表示从probabilities中的最早日期开始
            days: 计划天数
            
        Returns:
            采样计划DataFrame，包含'sku_id', 'ds', 'sample_flag'列
        """
        # 验证输入
        required_cols = ['ds', 'change_probability']
        if 'sku_id' not in probabilities.columns:
            logger.info("未提供SKU ID，假设为单一SKU")
            probabilities = probabilities.copy()
            probabilities['sku_id'] = 'SKU001'  # 默认SKU ID
            
        if not all(col in probabilities.columns for col in required_cols):
            raise ValueError(f"概率数据必须包含列: {required_cols}")
            
        # 确保日期格式正确
        probabilities = probabilities.copy()
        probabilities['ds'] = pd.to_datetime(probabilities['ds'])
        
        # 确定计划起始日期
        if start_date is None:
            start_date = probabilities['ds'].min()
        else:
            start_date = pd.to_datetime(start_date)
            
        # 确定计划结束日期
        end_date = start_date + timedelta(days=days-1)
        
        # 筛选日期范围内的数据
        mask = (probabilities['ds'] >= start_date) & (probabilities['ds'] <= end_date)
        probabilities = probabilities[mask].copy()
        
        if len(probabilities) == 0:
            raise ValueError(f"在指定日期范围内没有概率数据: {start_date} 到 {end_date}")
            
        # 如果提供了簇信息，按簇生成策略
        if sku_clusters:
            return self._generate_cluster_based_schedule(probabilities, sku_clusters)
        else:
            return self._generate_direct_schedule(probabilities)
    
    def _generate_direct_schedule(self, probabilities: pd.DataFrame) -> pd.DataFrame:
        """
        直接基于概率生成采样计划（不使用簇）
        
        Args:
            probabilities: 含概率的DataFrame
            
        Returns:
            采样计划DataFrame
        """
        logger.info("生成直接基于概率的采样计划")
        
        # 对每个SKU单独处理
        all_schedules = []
        
        for sku_id, sku_probs in probabilities.groupby('sku_id'):
            # 创建日期范围内的所有日期记录
            date_range = pd.date_range(
                start=sku_probs['ds'].min(),
                end=sku_probs['ds'].max(),
                freq='D'
            )
            
            # 确保每个日期都有记录
            sku_schedule = pd.DataFrame({'ds': date_range})
            sku_schedule['sku_id'] = sku_id
            
            # 合并概率信息
            sku_schedule = pd.merge(
                sku_schedule,
                sku_probs[['ds', 'change_probability']],
                on='ds',
                how='left'
            )
            
            # 填充缺失概率（使用临近日期均值）
            sku_schedule['change_probability'] = sku_schedule['change_probability'].fillna(
                method='ffill'
            ).fillna(method='bfill').fillna(0.5)  # 默认0.5
            
            # 基于概率生成采样标志
            sku_schedule = self._apply_adaptive_sampling(sku_schedule)
            
            all_schedules.append(sku_schedule)
            
        # 合并所有SKU的计划
        if all_schedules:
            schedule = pd.concat(all_schedules, ignore_index=True)
            return schedule[['sku_id', 'ds', 'sample_flag', 'sampling_reason']]
        else:
            return pd.DataFrame(columns=['sku_id', 'ds', 'sample_flag', 'sampling_reason'])
    
    def _generate_cluster_based_schedule(self, probabilities: pd.DataFrame, 
                                        sku_clusters: Dict) -> pd.DataFrame:
        """
        基于簇信息生成采样计划
        
        Args:
            probabilities: 含概率的DataFrame
            sku_clusters: SKU到簇的映射
            
        Returns:
            采样计划DataFrame
        """
        logger.info("生成基于簇的采样计划")
        
        # 将簇信息添加到概率数据中
        probabilities['cluster'] = probabilities['sku_id'].map(
            lambda x: sku_clusters.get(x, -1)
        )
        
        # 计算每个簇的平均变化概率
        cluster_probs = probabilities.groupby(['cluster', 'ds'])['change_probability'].mean().reset_index()
        
        # 为每个簇生成采样策略
        if self.cluster_strategies is None:
            self._generate_cluster_strategies(cluster_probs)
            
        # 对每个SKU应用其所属簇的策略
        all_schedules = []
        
        for sku_id, sku_probs in probabilities.groupby('sku_id'):
            # 获取SKU所属的簇
            cluster = sku_clusters.get(sku_id, -1)
            
            # 创建日期范围内的所有日期记录
            date_range = pd.date_range(
                start=sku_probs['ds'].min(),
                end=sku_probs['ds'].max(),
                freq='D'
            )
            
            # 确保每个日期都有记录
            sku_schedule = pd.DataFrame({'ds': date_range})
            sku_schedule['sku_id'] = sku_id
            sku_schedule['cluster'] = cluster
            
            # 合并概率信息
            sku_schedule = pd.merge(
                sku_schedule,
                sku_probs[['ds', 'change_probability']],
                on='ds',
                how='left'
            )
            
            # 填充缺失概率
            sku_schedule['change_probability'] = sku_schedule['change_probability'].fillna(
                method='ffill'
            ).fillna(method='bfill').fillna(0.5)  # 默认0.5
            
            # 应用簇策略
            cluster_strategy = self.cluster_strategies.get(cluster, 'adaptive')
            sku_schedule = self._apply_strategy(sku_schedule, cluster_strategy)
            
            all_schedules.append(sku_schedule)
            
        # 合并所有SKU的计划
        if all_schedules:
            schedule = pd.concat(all_schedules, ignore_index=True)
            
            # 限制每日采样总数（如果需要）
            if self.max_samples_per_day is not None:
                schedule = self._limit_daily_samples(schedule)
                
            return schedule[['sku_id', 'ds', 'sample_flag', 'sampling_reason']]
        else:
            return pd.DataFrame(columns=['sku_id', 'ds', 'sample_flag', 'sampling_reason'])
    
    def _generate_cluster_strategies(self, cluster_probs: pd.DataFrame):
        """
        为每个簇生成采样策略
        
        Args:
            cluster_probs: 簇级别的概率数据
        """
        # 计算每个簇的平均变化概率
        cluster_avg_probs = cluster_probs.groupby('cluster')['change_probability'].mean()
        
        # 根据平均概率分配策略
        self.cluster_strategies = {}
        
        for cluster, avg_prob in cluster_avg_probs.items():
            if avg_prob > 0.3:
                # 高变化概率簇：每日采样
                self.cluster_strategies[cluster] = 'daily'
            elif avg_prob > 0.1:
                # 中变化概率簇：根据星期几采样
                self.cluster_strategies[cluster] = 'weekday_based'
            else:
                # 低变化概率簇：自适应采样
                self.cluster_strategies[cluster] = 'adaptive'
                
        logger.info(f"簇策略生成完成: {self.cluster_strategies}")
    
    def _apply_strategy(self, schedule: pd.DataFrame, strategy: str) -> pd.DataFrame:
        """
        应用采样策略
        
        Args:
            schedule: 采样计划
            strategy: 策略名称
            
        Returns:
            应用策略后的计划
        """
        schedule = schedule.copy()
        
        if strategy == 'daily':
            # 每天都采样
            schedule['sample_flag'] = 1
            schedule['sampling_reason'] = 'daily_strategy'
            
        elif strategy == 'weekday_based':
            # 工作日每天，周末每两天
            schedule['weekday'] = schedule['ds'].dt.dayofweek
            schedule['sample_flag'] = 1
            
            # 周末（5=周六，6=周日）减少采样
            weekend_mask = schedule['weekday'] >= 5
            schedule.loc[weekend_mask, 'sample_flag'] = schedule.loc[weekend_mask].index % 2
            
            schedule['sampling_reason'] = 'weekday_strategy'
            schedule.drop(columns=['weekday'], inplace=True)
            
        elif strategy == 'weekend_focus':
            # 周末每天采样，工作日每两天采样
            schedule['weekday'] = schedule['ds'].dt.dayofweek
            
            # 工作日（0-4）隔天采样
            weekday_mask = schedule['weekday'] < 5
            schedule.loc[weekday_mask, 'sample_flag'] = 0
            alternating_days = schedule[weekday_mask].index[::2]  # 每隔一天
            schedule.loc[alternating_days, 'sample_flag'] = 1
            
            # 周末（5-6）每天采样
            weekend_mask = schedule['weekday'] >= 5
            schedule.loc[weekend_mask, 'sample_flag'] = 1
            
            schedule['sampling_reason'] = 'weekend_focus_strategy'
            schedule.drop(columns=['weekday'], inplace=True)
            
        elif strategy == 'interval_based':
            # 基于历史变化间隔的采样
            # 可以根据簇的平均变化间隔设置不同的采样间隔
            avg_interval = 7  # 默认间隔7天
            if hasattr(self, 'strategy_details') and 'interval_based' in self.strategy_details:
                details = self.strategy_details['interval_based']
                if 'interval' in details:
                    avg_interval = details['interval']
                    
            # 初始化为0
            schedule['sample_flag'] = 0
            
            # 从开始日期按间隔采样
            start_idx = schedule.index[0]
            sample_indices = range(start_idx, schedule.index[-1] + 1, avg_interval)
            schedule.loc[sample_indices, 'sample_flag'] = 1
            
            schedule['sampling_reason'] = f'interval_{avg_interval}d_strategy'
            
        elif strategy == 'sparse':
            # 稀疏采样 - 每周两次
            schedule['day_of_week'] = schedule['ds'].dt.dayofweek
            schedule['sample_flag'] = 0
            
            # 每周一和周四采样
            monday_mask = schedule['day_of_week'] == 0  # 周一
            thursday_mask = schedule['day_of_week'] == 3  # 周四
            
            schedule.loc[monday_mask | thursday_mask, 'sample_flag'] = 1
            schedule['sampling_reason'] = 'sparse_strategy'
            schedule.drop(columns=['day_of_week'], inplace=True)
            
        elif strategy == 'adaptive':
            # 根据概率自适应采样
            schedule = self._apply_adaptive_sampling(schedule)
            
        else:
            # 默认：使用自适应采样
            schedule = self._apply_adaptive_sampling(schedule)
            
        return schedule
    
    def _apply_adaptive_sampling(self, schedule: pd.DataFrame) -> pd.DataFrame:
        """
        应用自适应采样策略（基于概率阈值）
        
        Args:
            schedule: 采样计划
            
        Returns:
            应用自适应策略后的计划
        """
        schedule = schedule.copy()
        
        # 定义概率阈值
        high_threshold = 0.5   # 高概率阈值
        medium_threshold = 0.2  # 中等概率阈值
        
        # 初始化采样标志
        schedule['sample_flag'] = 0
        schedule['sampling_reason'] = 'skipped'
        
        # 对高概率变化的日期，必须采样
        high_prob_mask = schedule['change_probability'] >= high_threshold
        schedule.loc[high_prob_mask, 'sample_flag'] = 1
        schedule.loc[high_prob_mask, 'sampling_reason'] = 'high_probability'
        
        # 对中等概率的，每两天采样一次
        medium_prob_mask = (schedule['change_probability'] >= medium_threshold) & \
                          (schedule['change_probability'] < high_threshold)
        medium_indices = schedule[medium_prob_mask].index
        
        # 隔天采样
        if len(medium_indices) > 0:
            alt_day_indices = medium_indices[::2]  # 每隔一个索引
            schedule.loc[alt_day_indices, 'sample_flag'] = 1
            schedule.loc[alt_day_indices, 'sampling_reason'] = 'medium_probability'
        
        # 对低概率的，每三天采样一次
        low_prob_mask = schedule['change_probability'] < medium_threshold
        low_indices = schedule[low_prob_mask].index
        
        # 每三天采样
        if len(low_indices) > 0:
            third_day_indices = low_indices[::3]  # 每隔两个索引
            schedule.loc[third_day_indices, 'sample_flag'] = 1
            schedule.loc[third_day_indices, 'sampling_reason'] = 'low_probability'
            
        # 确保第一天和最后一天都采样
        if len(schedule) > 0:
            schedule.iloc[0, schedule.columns.get_loc('sample_flag')] = 1
            schedule.iloc[0, schedule.columns.get_loc('sampling_reason')] = 'boundary_day'
            schedule.iloc[-1, schedule.columns.get_loc('sample_flag')] = 1
            schedule.iloc[-1, schedule.columns.get_loc('sampling_reason')] = 'boundary_day'
            
        return schedule
    
    def _limit_daily_samples(self, schedule: pd.DataFrame) -> pd.DataFrame:
        """
        限制每日采样总数
        
        Args:
            schedule: 采样计划
            
        Returns:
            限制采样数后的计划
        """
        schedule = schedule.copy()
        
        # 计算每天的采样总数
        daily_counts = schedule.groupby('ds')['sample_flag'].sum().reset_index()
        over_limit_days = daily_counts[daily_counts['sample_flag'] > self.max_samples_per_day]['ds']
        
        # 对超限的天数进行处理
        for day in over_limit_days:
            day_mask = schedule['ds'] == day
            day_samples = schedule[day_mask].copy()
            
            # 按优先级排序：高概率 > 中概率 > 低概率
            priority_order = {
                'high_probability': 0,
                'medium_probability': 1, 
                'low_probability': 2,
                'boundary_day': 0,  # 边界日与高优先级相同
                'daily_strategy': 0,
                'weekday_strategy': 1,
                'skipped': 3
            }
            
            # 添加优先级列
            day_samples['priority'] = day_samples['sampling_reason'].map(priority_order)
            
            # 按优先级排序，然后按概率排序
            day_samples = day_samples.sort_values(
                ['priority', 'change_probability'],
                ascending=[True, False]
            )
            
            # 保留前max_samples_per_day个
            keep_indices = day_samples.index[:self.max_samples_per_day]
            drop_indices = day_samples.index[self.max_samples_per_day:]
            
            # 更新采样标志
            schedule.loc[drop_indices, 'sample_flag'] = 0
            schedule.loc[drop_indices, 'sampling_reason'] = 'daily_limit_exceeded'
            
        return schedule
    
    def estimate_capture_rate(self, 
                             schedule: pd.DataFrame, 
                             true_changes: Optional[pd.DataFrame] = None) -> float:
        """
        估计采样计划的捕获率
        
        Args:
            schedule: 采样计划
            true_changes: 实际的价格变化数据，None时使用概率模型估计
            
        Returns:
            估计的捕获率（0-1之间）
        """
        if 'change_probability' not in schedule.columns:
            raise ValueError("采样计划必须包含'change_probability'列以估计捕获率")
            
        if true_changes is not None:
            # 使用实际变化数据计算捕获率
            # 合并实际变化和采样计划
            merged = pd.merge(
                true_changes[['sku_id', 'ds', 'change_flag']],
                schedule[['sku_id', 'ds', 'sample_flag']],
                on=['sku_id', 'ds'],
                how='left'
            )
            
            # 计算捕获的变化数量和总变化数量
            total_changes = (merged['change_flag'] == 1).sum()
            captured_changes = ((merged['change_flag'] == 1) & (merged['sample_flag'] == 1)).sum()
            
            if total_changes == 0:
                return 1.0  # 没有变化，视为100%捕获
                
            return captured_changes / total_changes
        else:
            # 使用概率模型估计捕获率
            # 每个SKU,日期组合的预期捕获率 = 采样标志 * 变化概率
            expected_captures = schedule['sample_flag'] * schedule['change_probability']
            
            # 总体预期捕获率 = 预期捕获的变化数 / 预期的总变化数
            total_expected_changes = schedule['change_probability'].sum()
            
            if total_expected_changes == 0:
                return 1.0  # 没有预期变化，视为100%捕获
                
            return expected_captures.sum() / total_expected_changes
    
    def optimize_schedule(self, 
                         probabilities: pd.DataFrame, 
                         sku_clusters: Optional[Dict] = None,
                         start_date: Optional[datetime] = None,
                         days: int = 7) -> pd.DataFrame:
        """
        优化采样计划以达到目标捕获率
        
        Args:
            probabilities: 含概率的DataFrame
            sku_clusters: SKU到簇的映射
            start_date: 计划开始日期
            days: 计划天数
            
        Returns:
            优化后的采样计划
        """
        logger.info(f"开始优化采样计划，目标捕获率: {self.capture_rate_target}")
        
        # 生成初始计划
        schedule = self.generate_schedule(
            probabilities, sku_clusters, start_date, days
        )
        
        # 确保schedule包含change_probability列
        if 'change_probability' not in schedule.columns:
            schedule = pd.merge(
                schedule,
                probabilities[['sku_id', 'ds', 'change_probability']],
                on=['sku_id', 'ds'],
                how='left'
            )
        
        # 计算初始捕获率
        initial_capture_rate = self.estimate_capture_rate(schedule)
        logger.info(f"初始捕获率: {initial_capture_rate:.4f}, 目标: {self.capture_rate_target}")
        
        # 如果初始捕获率已经达到目标，直接返回
        if initial_capture_rate >= self.capture_rate_target:
            logger.info("初始计划已达到目标捕获率")
            return schedule[['sku_id', 'ds', 'sample_flag', 'sampling_reason']]
            
        # 增加采样直到达到目标捕获率
        while initial_capture_rate < self.capture_rate_target:
            # 找出潜在价格变化但未采样的条目，按概率排序
            potential_additions = schedule[
                (schedule['sample_flag'] == 0) & 
                (schedule['change_probability'] > 0)
            ].sort_values('change_probability', ascending=False)
            
            if len(potential_additions) == 0:
                logger.warning("无法达到目标捕获率，已采样所有可能的日期")
                break
                
            # 添加最高概率的未采样条目
            add_idx = potential_additions.index[0]
            schedule.loc[add_idx, 'sample_flag'] = 1
            schedule.loc[add_idx, 'sampling_reason'] = 'optimization'
            
            # 重新计算捕获率
            new_capture_rate = self.estimate_capture_rate(schedule)
            logger.debug(f"添加采样后捕获率: {new_capture_rate:.4f}")
            
            # 如果已达到目标，结束循环
            if new_capture_rate >= self.capture_rate_target:
                initial_capture_rate = new_capture_rate
                break
                
            # 更新捕获率
            initial_capture_rate = new_capture_rate
            
        # 如果设置了每日最大采样数，确保不超限
        if self.max_samples_per_day is not None:
            schedule = self._limit_daily_samples(schedule)
            # 重新计算捕获率
            final_capture_rate = self.estimate_capture_rate(schedule)
            logger.info(f"应用每日采样限制后的捕获率: {final_capture_rate:.4f}")
            
        logger.info(f"采样计划优化完成，最终捕获率: {initial_capture_rate:.4f}")
        
        # 统计采样次数和采样比例
        total_days = len(schedule) / schedule['sku_id'].nunique()
        total_samples = schedule['sample_flag'].sum()
        sampling_rate = total_samples / len(schedule)
        
        logger.info(f"采样计划统计: 总SKU数={schedule['sku_id'].nunique()}, " 
                   f"总天数={total_days}, 总采样次数={total_samples}, " 
                   f"采样比例={sampling_rate:.4f}")
        
        return schedule[['sku_id', 'ds', 'sample_flag', 'sampling_reason']]
    
    def save_schedule(self, schedule: pd.DataFrame, output_path: str):
        """
        保存采样计划到文件
        
        Args:
            schedule: 采样计划DataFrame
            output_path: 输出文件路径
        """
        # 创建输出目录
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 保存为CSV
        schedule.to_csv(output_path, index=False)
        logger.info(f"采样计划已保存到 {output_path}")

    def generate_sentinel_plan(self, 
                             probabilities: pd.DataFrame,
                             correlation_network: Dict[str, List[str]],
                             start_date: Optional[datetime] = None,
                             days: int = 7) -> pd.DataFrame:
        """
        生成哨兵SKU采样计划
        
        Args:
            probabilities: 含概率的DataFrame
            correlation_network: SKU相关性网络，格式: {sentinel_sku: [dependent_sku1, ...]}
            start_date: 计划开始日期
            days: 计划天数
            
        Returns:
            包含哨兵机制的采样计划
        """
        logger.info("生成哨兵SKU采样计划")
        
        # 生成基础采样计划
        base_schedule = self.generate_schedule(
            probabilities, None, start_date, days
        )
        
        # 确定哨兵SKU列表
        sentinel_skus = list(correlation_network.keys())
        
        # 确保所有哨兵SKU都有高频率采样
        enhanced_schedule = base_schedule.copy()
        
        # 对哨兵SKU应用每日采样策略
        for sku in sentinel_skus:
            sku_mask = enhanced_schedule['sku_id'] == sku
            enhanced_schedule.loc[sku_mask, 'sample_flag'] = 1
            enhanced_schedule.loc[sku_mask, 'sampling_reason'] = 'sentinel_sku'
            
        # 添加触发逻辑的标记列
        enhanced_schedule['triggers'] = ''
        for sentinel, dependents in correlation_network.items():
            # 查找哨兵SKU的条目
            sentinel_entries = enhanced_schedule[enhanced_schedule['sku_id'] == sentinel]
            
            # 记录该哨兵触发的依赖SKU列表
            sentinel_mask = enhanced_schedule['sku_id'] == sentinel
            dependent_str = ','.join(dependents)
            enhanced_schedule.loc[sentinel_mask, 'triggers'] = dependent_str
            
        # 如果设置了每日最大采样数，确保不超限
        if self.max_samples_per_day is not None:
            enhanced_schedule = self._limit_daily_samples(enhanced_schedule)
            
        return enhanced_schedule

    def evaluate_strategies(self, validation_data, sku_clusters, output_dir=None):
        """
        评估不同采样策略的效果，并生成评估报告
        
        Args:
            validation_data: 包含实际价格变化的验证数据
            sku_clusters: SKU到簇的映射
            output_dir: 输出评估报告的目录（可选）
            
        Returns:
            dict: 包含评估结果的字典
        """
        if 'change_flag' not in validation_data.columns:
            raise ValueError("验证数据必须包含'change_flag'列以进行评估")
            
        # 确保日期格式正确
        validation_data = validation_data.copy()
        validation_data['ds'] = pd.to_datetime(validation_data['ds'])
        
        # 准备用于评估的数据
        evaluation_df = validation_data[['sku_id', 'ds', 'change_flag']].copy()
        evaluation_df['change_probability'] = 0.5  # 为采样计划生成提供默认概率
        
        # 生成采样计划
        schedule = self.generate_schedule(
            probabilities=evaluation_df,
            sku_clusters=sku_clusters,
            days=len(evaluation_df['ds'].unique())
        )
        
        # 合并采样计划与验证数据
        merged = pd.merge(
            evaluation_df[['sku_id', 'ds', 'change_flag']],
            schedule[['sku_id', 'ds', 'sample_flag', 'sampling_reason']],
            on=['sku_id', 'ds'],
            how='left'
        )
        
        # 填充缺失的采样标志为0
        merged['sample_flag'] = merged['sample_flag'].fillna(0)
        
        # 计算总体指标
        total_samples = merged['sample_flag'].sum()
        total_records = len(merged)
        total_changes = merged['change_flag'].sum()
        captured_changes = ((merged['change_flag'] == 1) & (merged['sample_flag'] == 1)).sum()
        
        # 计算捕获率和效率
        capture_rate = captured_changes / total_changes if total_changes > 0 else 1.0
        sampling_rate = total_samples / total_records
        efficiency = captured_changes / total_samples if total_samples > 0 else 0
        
        overall_metrics = {
            'total_records': total_records,
            'total_samples': total_samples,
            'total_changes': total_changes,
            'captured_changes': captured_changes,
            'capture_rate': capture_rate,
            'sampling_rate': sampling_rate,
            'efficiency': efficiency
        }
        
        # 按簇计算评估指标
        cluster_metrics = {}
        for cluster, skus in self._group_skus_by_cluster(sku_clusters).items():
            # 筛选该簇的数据
            cluster_data = merged[merged['sku_id'].isin(skus)]
            
            if len(cluster_data) == 0:
                continue
                
            # 计算该簇的指标
            c_total_samples = cluster_data['sample_flag'].sum()
            c_total_records = len(cluster_data)
            c_total_changes = cluster_data['change_flag'].sum()
            c_captured_changes = ((cluster_data['change_flag'] == 1) & 
                                 (cluster_data['sample_flag'] == 1)).sum()
                
            # 计算该簇的捕获率和效率
            c_capture_rate = c_captured_changes / c_total_changes if c_total_changes > 0 else 1.0
            c_sampling_rate = c_total_samples / c_total_records
            c_efficiency = c_captured_changes / c_total_samples if c_total_samples > 0 else 0
            
            # 记录策略信息
            strategy = self.cluster_strategies.get(cluster, 'adaptive')
            
            cluster_metrics[cluster] = {
                'strategy': strategy,
                'sku_count': len(skus),
                'total_records': c_total_records,
                'total_samples': c_total_samples,
                'total_changes': c_total_changes,
                'captured_changes': c_captured_changes,
                'capture_rate': c_capture_rate,
                'sampling_rate': c_sampling_rate,
                'efficiency': c_efficiency
            }
            
        # 按采样原因计算评估指标
        reason_metrics = {}
        for reason in merged['sampling_reason'].dropna().unique():
            # 筛选该采样原因的数据
            reason_data = merged[merged['sampling_reason'] == reason]
            
            if len(reason_data) == 0:
                continue
                
            # 计算该采样原因的指标
            r_total_samples = reason_data['sample_flag'].sum()
            r_total_records = len(reason_data)
            r_total_changes = reason_data['change_flag'].sum()
            r_captured_changes = ((reason_data['change_flag'] == 1) & 
                                 (reason_data['sample_flag'] == 1)).sum()
                
            # 计算该采样原因的捕获率和效率
            r_capture_rate = r_captured_changes / r_total_changes if r_total_changes > 0 else 1.0
            r_sampling_rate = r_total_samples / r_total_records
            r_efficiency = r_captured_changes / r_total_samples if r_total_samples > 0 else 0
            
            reason_metrics[reason] = {
                'total_records': r_total_records,
                'total_samples': r_total_samples,
                'total_changes': r_total_changes,
                'captured_changes': r_captured_changes,
                'capture_rate': r_capture_rate,
                'sampling_rate': r_sampling_rate,
                'efficiency': r_efficiency
            }
            
        # 汇总评估结果
        evaluation_results = {
            'overall': overall_metrics,
            'clusters': cluster_metrics,
            'sampling_reasons': reason_metrics
        }
        
        # 输出评估报告（如果指定了输出目录）
        if output_dir:
            self._save_evaluation_report(evaluation_results, output_dir)
            
        return evaluation_results
    
    def optimize_cluster_strategies(self, validation_data, sku_clusters, target_capture_rate=0.95):
        """
        根据验证数据优化簇的采样策略
        
        Args:
            validation_data: 包含实际价格变化的验证数据
            sku_clusters: SKU到簇的映射
            target_capture_rate: 目标捕获率
            
        Returns:
            dict: 优化后的簇策略映射
        """
        if not hasattr(self, 'cluster_strategies'):
            raise ValueError("请先初始化簇策略")
            
        # 评估当前策略
        evaluation = self.evaluate_strategies(validation_data, sku_clusters)
        cluster_metrics = evaluation['clusters']
        
        # 策略强度排序（由弱到强）
        strategy_strengths = {
            'sparse': 1,
            'interval_based': 2,
            'weekday_based': 3,
            'weekend_focus': 4,
            'daily': 5
        }
        
        # 预定义的策略选项（按强度排序）
        strategies = ['sparse', 'interval_based', 'weekday_based', 'weekend_focus', 'daily']
        
        # 优化每个簇的策略
        optimized_strategies = self.cluster_strategies.copy()
        
        for cluster, metrics in cluster_metrics.items():
            current_strategy = metrics['strategy']
            current_capture_rate = metrics['capture_rate']
            
            # 如果捕获率低于目标，提高策略强度
            if current_capture_rate < target_capture_rate:
                # 找到当前策略的强度
                current_strength = strategy_strengths.get(current_strategy, 0)
                
                # 选择更强的策略
                for s in strategies:
                    if strategy_strengths[s] > current_strength:
                        optimized_strategies[cluster] = s
                        break
                        
            # 如果捕获率远高于目标，且采样率很高，可以降低策略强度
            elif current_capture_rate > target_capture_rate + 0.1 and metrics['sampling_rate'] > 0.7:
                # 找到当前策略的强度
                current_strength = strategy_strengths.get(current_strategy, 0)
                
                # 选择更弱但足够的策略
                for s in reversed(strategies):
                    if strategy_strengths[s] < current_strength:
                        # 模拟应用该策略
                        self.cluster_strategies[cluster] = s
                        test_eval = self.evaluate_strategies(validation_data, sku_clusters)
                        test_capture_rate = test_eval['clusters'].get(cluster, {}).get('capture_rate', 0)
                        
                        # 如果仍能达到目标，则使用该策略
                        if test_capture_rate >= target_capture_rate:
                            optimized_strategies[cluster] = s
                            break
                            
                # 恢复原策略（用于下一次测试）
                self.cluster_strategies[cluster] = current_strategy
                
        # 更新策略
        self.cluster_strategies = optimized_strategies
        
        return optimized_strategies
    
    def _group_skus_by_cluster(self, sku_clusters):
        """
        将SKU按簇分组
        
        Args:
            sku_clusters: SKU到簇的映射 {sku_id: cluster}
            
        Returns:
            dict: 簇到SKU列表的映射 {cluster: [sku_id, ...]}
        """
        clusters = {}
        for sku, cluster in sku_clusters.items():
            if cluster not in clusters:
                clusters[cluster] = []
            clusters[cluster].append(sku)
            
        return clusters
    
    def _save_evaluation_report(self, evaluation_results, output_dir):
        """
        保存评估结果报告
        
        Args:
            evaluation_results: 评估结果字典
            output_dir: 输出目录
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存总体指标
        overall_df = pd.DataFrame([evaluation_results['overall']])
        overall_path = os.path.join(output_dir, 'overall_metrics.csv')
        overall_df.to_csv(overall_path, index=False)
        
        # 保存簇指标
        cluster_df = pd.DataFrame.from_dict(evaluation_results['clusters'], orient='index')
        cluster_df.index.name = 'cluster'
        cluster_df = cluster_df.reset_index()
        cluster_path = os.path.join(output_dir, 'cluster_metrics.csv')
        cluster_df.to_csv(cluster_path, index=False)
        
        # 保存采样原因指标
        reason_df = pd.DataFrame.from_dict(evaluation_results['sampling_reasons'], orient='index')
        reason_df.index.name = 'sampling_reason'
        reason_df = reason_df.reset_index()
        reason_path = os.path.join(output_dir, 'reason_metrics.csv')
        reason_df.to_csv(reason_path, index=False)
        
        # 生成可视化
        self._visualize_evaluation(evaluation_results, output_dir)
        
        logger.info(f"评估报告已保存到: {output_dir}")
    
    def _visualize_evaluation(self, evaluation_results, output_dir):
        """
        可视化评估结果
        
        Args:
            evaluation_results: 评估结果字典
            output_dir: 输出目录
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # 设置可视化风格
        sns.set_style("whitegrid")
        
        # 1. 簇捕获率对比图
        plt.figure(figsize=(12, 8))
        
        cluster_metrics = evaluation_results['clusters']
        clusters = list(cluster_metrics.keys())
        capture_rates = [m['capture_rate'] for m in cluster_metrics.values()]
        sampling_rates = [m['sampling_rate'] for m in cluster_metrics.values()]
        strategies = [m['strategy'] for m in cluster_metrics.values()]
        
        # 创建颜色映射
        strategy_colors = {
            'daily': 'red',
            'weekend_focus': 'orange',
            'weekday_based': 'green',
            'interval_based': 'blue',
            'sparse': 'purple',
            'adaptive': 'gray'
        }
        
        colors = [strategy_colors.get(s, 'gray') for s in strategies]
        
        # 捕获率条形图
        plt.subplot(2, 1, 1)
        bars = plt.bar(clusters, capture_rates, color=colors)
        plt.axhline(evaluation_results['overall']['capture_rate'], 
                   color='black', linestyle='--', label='平均捕获率')
        plt.axhline(0.95, color='red', linestyle='--', label='目标捕获率 (95%)')
        
        # 添加策略标签
        for i, bar in enumerate(bars):
            plt.text(bar.get_x() + bar.get_width()/2., 
                    bar.get_height() + 0.02, 
                    strategies[i], 
                    ha='center', va='bottom', rotation=0)
            
        plt.title('各簇捕获率与采用策略')
        plt.xlabel('簇标签')
        plt.ylabel('捕获率')
        plt.ylim(0, 1.1)
        plt.legend()
        
        # 采样率条形图
        plt.subplot(2, 1, 2)
        plt.bar(clusters, sampling_rates, color=colors)
        plt.axhline(evaluation_results['overall']['sampling_rate'], 
                   color='black', linestyle='--', label='平均采样率')
        
        plt.title('各簇采样率')
        plt.xlabel('簇标签')
        plt.ylabel('采样率')
        plt.ylim(0, 1.1)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'cluster_performance.png'), dpi=300)
        plt.close()
        
        # 2. 采样原因效率图
        plt.figure(figsize=(14, 8))
        
        reason_metrics = evaluation_results['sampling_reasons']
        reasons = list(reason_metrics.keys())
        reason_capture_rates = [m['capture_rate'] for m in reason_metrics.values()]
        reason_efficiencies = [m['efficiency'] for m in reason_metrics.values()]
        
        # 效率与捕获率对比
        plt.subplot(1, 2, 1)
        x_pos = range(len(reasons))
        width = 0.35
        
        plt.bar(x_pos, reason_capture_rates, width, label='捕获率')
        plt.bar([p + width for p in x_pos], reason_efficiencies, width, label='效率')
        
        plt.xlabel('采样原因')
        plt.ylabel('比率')
        plt.title('不同采样原因的捕获率与效率')
        plt.xticks([p + width/2 for p in x_pos], reasons, rotation=45, ha='right')
        plt.ylim(0, 1.1)
        plt.legend()
        
        # 采样数量分布
        plt.subplot(1, 2, 2)
        samples = [m['total_samples'] for m in reason_metrics.values()]
        plt.pie(samples, labels=reasons, autopct='%1.1f%%', startangle=90)
        plt.axis('equal')
        plt.title('采样数量分布')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'sampling_reason_analysis.png'), dpi=300)
        plt.close() 