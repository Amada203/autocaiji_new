"""
模型评估指标计算模块
"""
import numpy as np
import pandas as pd

def calculate_capture_rate(predictions, actual):
    """计算价格变动捕获率
    
    Args:
        predictions: 模型预测的变动概率
        actual: 实际数据DataFrame
        
    Returns:
        float: 捕获率 [0,1]
    """
    # 实际价格变动
    actual_changes = actual.groupby('sku_id')['page_price'].diff().abs() > 0
    actual_changes = actual_changes.fillna(False)
    
    # 预测的价格变动
    pred_changes = predictions > 0.5  # 假设predictions是变动概率
    
    # 计算捕获率
    true_positives = (actual_changes & pred_changes).sum()
    total_changes = actual_changes.sum()
    
    return true_positives / total_changes if total_changes > 0 else 0

def calculate_accuracy(predictions, actual):
    """计算预测准确率
    
    Args:
        predictions: 模型预测的变动概率
        actual: 实际数据DataFrame
        
    Returns:
        float: 准确率 [0,1]
    """
    # 实际价格变动
    actual_changes = actual.groupby('sku_id')['page_price'].diff().abs() > 0
    actual_changes = actual_changes.fillna(False)
    
    # 预测的价格变动
    pred_changes = predictions > 0.5  # 假设predictions是变动概率
    
    # 计算准确率
    correct = (actual_changes == pred_changes).sum()
    total = len(actual_changes)
    
    return correct / total

def calculate_fpr(predictions, actual):
    """计算误报率
    
    Args:
        predictions: 模型预测的变动概率
        actual: 实际数据DataFrame
        
    Returns:
        float: 误报率 [0,1]
    """
    # 实际价格变动
    actual_changes = actual.groupby('sku_id')['page_price'].diff().abs() > 0
    actual_changes = actual_changes.fillna(False)
    
    # 预测的价格变动
    pred_changes = predictions > 0.5  # 假设predictions是变动概率
    
    # 计算误报率
    false_positives = ((~actual_changes) & pred_changes).sum()
    true_negatives = ((~actual_changes) & (~pred_changes)).sum()
    
    return false_positives / (false_positives + true_negatives) if (false_positives + true_negatives) > 0 else 0

def compare_metrics(old_metrics, new_metrics):
    """比较新旧模型指标
    
    Args:
        old_metrics: dict 旧模型指标
        new_metrics: dict 新模型指标
        
    Returns:
        dict: 指标对比结果
    """
    comparison = {}
    for metric in old_metrics:
        if metric in new_metrics:
            improvement = (new_metrics[metric] - old_metrics[metric]) / old_metrics[metric]
            comparison[metric] = {
                'old': old_metrics[metric],
                'new': new_metrics[metric],
                'improvement': improvement,
                'significant': abs(improvement) > 0.05  # 5%以上视为显著
            }
    return comparison