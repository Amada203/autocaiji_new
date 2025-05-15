#!/usr/bin/env python3
"""
项目目录初始化脚本
"""

import os
from pathlib import Path

# 需要创建的目录列表
REQUIRED_DIRS = [
    'logs',             # 日志文件
    'data/raw',         # 原始数据
    'data/processed',   # 处理后的数据
    'models',           # 训练好的模型
    'static/js',        # 前端JavaScript
    'static/css',       # 前端样式
    'templates'         # HTML模板
]

# 需要创建的初始化文件
INIT_FILES = {
    'logs/.gitkeep': '',
    'data/raw/.gitkeep': '',
    'data/processed/.gitkeep': ''
}

def initialize_project():
    """初始化项目目录结构"""
    print("初始化项目目录结构...")
    
    # 创建所有必要目录
    for dir_path in REQUIRED_DIRS:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"创建目录: {dir_path}")
    
    # 创建占位文件
    for file_path, content in INIT_FILES.items():
        with open(file_path, 'w') as f:
            f.write(content)
        print(f"创建文件: {file_path}")
    
    print("项目初始化完成！")

if __name__ == "__main__":
    initialize_project()