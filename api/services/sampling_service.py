from typing import List, Dict, Any, Optional
from datetime import date, datetime
import pandas as pd
import numpy as np
import os
import json
import random
from api.services.prediction_service import PredictionService

class SamplingService:
    def __init__(self):
        """初始化采样服务"""
        # 数据存储路径
        self.data_dir = os.environ.get("DATA_DIR", "data")
        os.makedirs(self.data_dir, exist_ok=True)
        
        self.plans_path = os.path.join(self.data_dir, "sampling_plans.json")
        
        # 初始化存储
        if not os.path.exists(self.plans_path):
            with open(self.plans_path, 'w') as f:
                json.dump([], f)
        
        self.prediction_service = PredictionService()
    
    def generate_plan(
        self, 
        prediction_date=None,
        target_capture_rate=0.95,
        use_stratified_sampling=True,
        max_samples=None
    ):
        """生成采样计划
        
        Args:
            prediction_date: 预测日期
            target_capture_rate: 目标捕捉率
            use_stratified_sampling: 是否使用分层采样
            max_samples: 最大采样数量
            
        Returns:
            dict: 采样计划
        """
        if prediction_date is None:
            prediction_date = date.today()
            
        # 用真实数据
        sku_probs = self._get_all_skus_and_probs()
        all_skus = list(sku_probs.keys())
        
        # 采样逻辑
        threshold = 0.5  # 可根据业务调整
        selected_skus = [sku for sku, prob in sku_probs.items() if prob >= threshold]
        
        # 限制采样数量
        if max_samples and len(selected_skus) > max_samples:
            sorted_pairs = sorted(
                [(sku, sku_probs[sku]) for sku in all_skus],
                key=lambda x: x[1],
                reverse=True
            )
            selected_skus = [pair[0] for pair in sorted_pairs[:max_samples]]
        
        # 计算采样率和成本节省
        sampling_rate = len(selected_skus) / len(all_skus) if all_skus else 0
        cost_saving = 1 - sampling_rate if all_skus else 0
        
        # 构建计划
        plan = {
            'date': prediction_date,
            'sku_ids': selected_skus,
            'sampling_rate': sampling_rate,
            'estimated_capture_rate': target_capture_rate,
            'estimated_cost_saving': cost_saving
        }
        
        # 保存计划
        self._save_plan(plan)
        
        return plan
    
    def get_plan(self, plan_date=None):
        """获取指定日期的采样计划
        
        Args:
            plan_date: 计划日期
            
        Returns:
            dict: 采样计划，如果不存在则返回None
        """
        if plan_date is None:
            plan_date = date.today()
            
        # 从存储中获取计划
        plans = self._load_plans()
        
        # 查找匹配的计划
        for plan in plans:
            if isinstance(plan['date'], str):
                plan_date_str = plan_date.isoformat()
                if plan['date'] == plan_date_str:
                    # 转换日期字符串为日期对象
                    plan['date'] = date.fromisoformat(plan['date'])
                    return plan
            else:
                if plan['date'] == plan_date:
                    return plan
        
        return None
    
    def get_plan_stats(self, plan_date=None):
        """获取采样计划统计信息
        
        Args:
            plan_date: 计划日期
            
        Returns:
            dict: 统计信息
        """
        if plan_date is None:
            plan_date = date.today()
            
        # 获取计划
        plan = self.get_plan(plan_date)
        if not plan:
            raise ValueError(f"未找到日期为 {plan_date} 的采样计划")
        
        # 计算统计信息
        all_skus = self._get_all_skus()
        sampled_skus = plan['sku_ids']
        
        stats = {
            'total_skus': len(all_skus),
            'sampled_skus': len(sampled_skus),
            'sampling_rate': len(sampled_skus) / len(all_skus),
            'cost_saving': 1 - (len(sampled_skus) / len(all_skus))
        }
        
        # 模拟分层统计
        strata_stats = []
        for i in range(5):  # 假设有5个层
            total = len(all_skus) // 5
            sampled = len([sku for sku in sampled_skus if hash(sku) % 5 == i])
            strata_stats.append({
                'cluster_id': i,
                'total': total,
                'sampled': sampled,
                'sampling_rate': sampled / total if total > 0 else 0
            })
        
        stats['strata_stats'] = strata_stats
        
        return stats
    
    def _get_all_skus_and_probs(self):
        """从真实预测结果获取所有SKU及其预测概率"""
        predictions = self.prediction_service.get_predictions(limit=10000)
        sku_probs = {}
        for p in predictions:
            sku = p.get('sku_id') or p.get('sku')
            prob = p.get('probability') or p.get('predicted_prob') or 0
            if sku:
                sku_probs[sku] = prob
        return sku_probs

    def _get_all_skus(self):
        """获取所有SKU（真实数据）"""
        sku_probs = self._get_all_skus_and_probs()
        return list(sku_probs.keys())
    
    def _save_plan(self, plan):
        """保存采样计划
        
        Args:
            plan: 采样计划字典
        """
        # 加载现有计划
        plans = self._load_plans()
        
        # 转换日期为字符串
        if isinstance(plan['date'], date):
            plan_copy = plan.copy()
            plan_copy['date'] = plan['date'].isoformat()
        else:
            plan_copy = plan
        
        # 检查是否已存在同日期的计划
        for i, p in enumerate(plans):
            if p['date'] == plan_copy['date']:
                plans[i] = plan_copy
                break
        else:
            # 不存在则添加新计划
            plans.append(plan_copy)
        
        # 保存到文件
        with open(self.plans_path, 'w') as f:
            json.dump(plans, f, indent=2)
    
    def _load_plans(self):
        """加载采样计划
        
        Returns:
            list: 采样计划列表
        """
        try:
            with open(self.plans_path, 'r') as f:
                return json.load(f)
        except:
            return []