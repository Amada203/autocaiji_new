"""
安全导入工具
"""
import sys
from pathlib import Path
import importlib

def safe_import(module_path, class_name):
    """
    安全导入类
    :param module_path: 模块路径，如'models.price_model'
    :param class_name: 类名，如'PriceModel'
    :return: 导入的类
    """
    try:
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    except Exception as e:
        print(f"导入失败: {module_path}.{class_name}")
        print(f"系统路径: {sys.path}")
        raise ImportError(f"无法导入 {module_path}.{class_name}: {str(e)}")