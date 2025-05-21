#!/usr/bin/env python3
"""
测试Impala查询性能脚本
"""

from impala.dbapi import connect
import sys
import os
import time
import json
from datetime import datetime

def test_impala_connection():
    """测试Impala基本连接"""
    print("开始测试Impala连接...")
    
    # 从JSON配置文件加载
    config_file = 'config/database.json'
    try:
        with open(config_file) as f:
            config = json.load(f)
        impala_config = {
            'host': config['impala']['host'],
            'port': int(config['impala']['port']),
            'user': config['impala']['user'],
            'password': config['impala']['password'],
            'database': config['impala']['database'],
            'auth_mechanism': 'PLAIN'
        }
        print(f"从JSON配置文件加载Impala配置: {config_file}")
    except Exception as e:
        print(f"配置加载失败: {str(e)}")
        return False
    
    # 打印连接信息（隐藏密码）
    safe_config = impala_config.copy()
    safe_config['password'] = '********'
    print(f"连接参数: {safe_config}")
    
    try:
        # 测试连接
        start_time = time.time()
        conn = connect(**impala_config)
        cursor = conn.cursor()
        
        # 检查数据库
        cursor.execute("SHOW DATABASES")
        databases = [db[0] for db in cursor.fetchall()]
        
        if impala_config['database'] in databases:
            print(f"✅ 数据库 '{impala_config['database']}' 存在")
        else:
            print(f"❌ 数据库 '{impala_config['database']}' 不存在")
            return False
        
        # 检查表
        cursor.execute(f"USE {impala_config['database']}")
        cursor.execute("SHOW TABLES")
        tables = [table[0] for table in cursor.fetchall()]
        
        cursor.close()
        conn.close()
        elapsed = time.time() - start_time
        print(f"✅ Impala连接测试成功，耗时: {elapsed:.2f}秒")
        return True
        
    except Exception as e:
        print(f"❌ Impala连接错误: {str(e)}")
        return False

if __name__ == "__main__":
    print(f"\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Impala测试开始")
    
    success = test_impala_connection()
    if not success:
        sys.exit(1)
    print(f"\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Impala测试结束")
    sys.exit(0 if success else 1)