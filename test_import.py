#!/usr/bin/env python3
"""
测试脚本，验证模块导入
"""
import sys
import os

# 添加项目根目录和src目录到Python路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

try:
    from src.models.fusion_model import ProphetLGBMFusion
    print("成功导入ProphetLGBMFusion类")
    print(f"类路径: {ProphetLGBMFusion.__module__}")
except ImportError as e:
    print(f"导入失败: {str(e)}")
    print("当前Python路径:")
    for p in sys.path:
        print(f" - {p}")