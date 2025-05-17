# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
from src.validation.metrics import calculate_capture_rate, calculate_accuracy, calculate_fpr, compare_metrics
from src.validation.data_utils import prepare_test_data, split_for_validation

# 导入旧版模型（从根目录的models文件夹）
try:
    from models.price_model import PriceModel as OldPricePredictor
    from models.price_change_probability_model import PriceChangeProbabilityModel as OldChangeDetector
    print("✅ 旧模型导入成功")
except ImportError as e:
    print(f"❌ 旧模型导入失败: {e}")
    raise

# 导入新版模型（从src/models/predictors）
try:
    from src.models.predictors.prophet_predictor import ProphetPredictor as NewPricePredictor
    from src.models.predictors.change_detector import ChangeDetector as NewChangeDetector
    print("✅ 新模型导入成功") 
except ImportError as e:
    print(f"❌ 新模型导入失败: {e}")
    raise

def validate_model_migration(old_model, new_model, test_data):
    """
    验证模型迁移是否正确
    
    Args:
        old_model: 旧模型实例
        new_model: 新模型实例
        test_data: 测试数据
        
    Returns:
        dict: 包含验证结果的字典
    """
    # 1. 确保新旧模型都能预测
    try:
        old_preds = old_model.predict(test_data)
        new_preds = new_model.predict(test_data)
    except Exception as e:
        return {
            'status': 'failed',
            'message': f'预测失败: {str(e)}'
        }
    
    # 2. 比较预测结果
    if len(old_preds) != len(new_preds):
        return {
            'status': 'failed',
            'message': f'预测结果长度不匹配: 旧模型{len(old_preds)} vs 新模型{len(new_preds)}'
        }
    
    # 3. 计算一致性
    agreement = np.mean(old_preds == new_preds)
    
    return {
        'status': 'success',
        'agreement': agreement,
        'old_predictions': old_preds,
        'new_predictions': new_preds
    }

def compare_model_performance(old_model, new_model, test_data, y_true):
    """
    比较新旧模型的性能
    
    Args:
        old_model: 旧模型实例
        new_model: 新模型实例
        test_data: 测试数据
        y_true: 真实标签
        
    Returns:
        dict: 包含性能比较结果的字典
    """
    results = {}
    
    # 旧模型性能
    old_metrics = {
        'capture_rate': calculate_capture_rate(y_true, old_model.predict(test_data)),
        'accuracy': calculate_accuracy(y_true, old_model.predict(test_data)),
        'fpr': calculate_fpr(y_true, old_model.predict(test_data))
    }
    
    # 新模型性能
    new_metrics = {
        'capture_rate': calculate_capture_rate(y_true, new_model.predict(test_data)),
        'accuracy': calculate_accuracy(y_true, new_model.predict(test_data)),
        'fpr': calculate_fpr(y_true, new_model.predict(test_data))
    }
    
    # 比较结果
    results['old_model'] = old_metrics
    results['new_model'] = new_metrics
    results['comparison'] = compare_metrics(old_metrics, new_metrics)
    
    return results