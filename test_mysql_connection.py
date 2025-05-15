#!/usr/bin/env python3
"""
测试MySQL连接脚本
"""

import mysql.connector
import sys
import os
from configparser import ConfigParser

def test_mysql_connection():
    """测试MySQL连接"""
    print("开始测试MySQL连接...")
    
    # 尝试从配置文件加载
    config_file = 'config/database.ini'
    if os.path.exists(config_file):
        try:
            config = ConfigParser()
            config.read(config_file)
            mysql_config = {
                'host': config['mysql']['host'],
                'port': int(config['mysql']['port']),
                'user': config['mysql']['user'],
                'password': config['mysql']['password'],
                'database': config['mysql']['database']
            }
            print(f"从配置文件加载MySQL配置: {config_file}")
        except Exception as e:
            print(f"从配置文件加载失败: {str(e)}")
            mysql_config = None
    else:
        print(f"配置文件不存在: {config_file}")
        mysql_config = None
    
    # 使用硬编码配置作为备份
    if not mysql_config:
        mysql_config = {
            'host': 'localhost',
            'port': 13306,
            'user': 'root',
            'password': 'mypassword123',
            'database': 'price_prediction'
        }
        print("使用硬编码MySQL配置")
    
    # 打印连接信息（隐藏密码）
    safe_config = mysql_config.copy()
    safe_config['password'] = '********'
    print(f"连接参数: {safe_config}")
    
    # 尝试连接
    try:
        # 先尝试不指定数据库连接
        base_config = mysql_config.copy()
        if 'database' in base_config:
            del base_config['database']
        
        print("尝试连接到MySQL服务器（不指定数据库）...")
        conn = mysql.connector.connect(
            **base_config,
            auth_plugin='mysql_native_password',
            connection_timeout=10
        )
        print("✅ 成功连接到MySQL服务器！")
        
        # 检查数据库是否存在
        cursor = conn.cursor()
        cursor.execute("SHOW DATABASES")
        databases = [db[0] for db in cursor.fetchall()]
        
        if mysql_config['database'] in databases:
            print(f"✅ 数据库 '{mysql_config['database']}' 已存在")
        else:
            print(f"❌ 数据库 '{mysql_config['database']}' 不存在")
            print(f"创建数据库 '{mysql_config['database']}'...")
            cursor.execute(f"CREATE DATABASE {mysql_config['database']}")
            print(f"✅ 数据库 '{mysql_config['database']}' 创建成功")
        
        cursor.close()
        conn.close()
        
        # 连接到指定数据库
        print(f"尝试连接到数据库 '{mysql_config['database']}'...")
        conn = mysql.connector.connect(
            **mysql_config,
            auth_plugin='mysql_native_password',
            connection_timeout=10
        )
        print(f"✅ 成功连接到数据库 '{mysql_config['database']}'！")
        
        # 检查表是否存在
        cursor = conn.cursor()
        cursor.execute("SHOW TABLES")
        tables = [table[0] for table in cursor.fetchall()]
        
        required_tables = ['sku_predictions', 'sku_history']
        missing_tables = [table for table in required_tables if table not in tables]
        
        if missing_tables:
            print(f"以下表不存在: {missing_tables}")
            print("创建缺失的表...")
            
            if 'sku_predictions' in missing_tables:
                print("创建表 'sku_predictions'...")
                cursor.execute("""
                CREATE TABLE sku_predictions (
                    sku_id VARCHAR(50) PRIMARY KEY,
                    sku_name VARCHAR(100),
                    current_price DECIMAL(10,2),
                    predicted_prob FLOAT,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """)
                print("✅ 表 'sku_predictions' 创建成功")
            
            if 'sku_history' in missing_tables:
                print("创建表 'sku_history'...")
                cursor.execute("""
                CREATE TABLE sku_history (
                    sku_id VARCHAR(50),
                    date DATE,
                    price DECIMAL(10,2),
                    is_promotion BOOLEAN,
                    PRIMARY KEY (sku_id, date)
                )
                """)
                print("✅ 表 'sku_history' 创建成功")
        else:
            print("✅ 所有必要的表都已存在")
        
        cursor.close()
        conn.close()
        
        print("MySQL连接测试完成，一切正常！")
        return True
        
    except mysql.connector.Error as err:
        print(f"❌ MySQL连接错误: {err}")
        
        if err.errno == 1045:  # Access denied
            print("可能原因: 用户名或密码错误")
            print("解决方案: 检查配置文件中的用户名和密码")
        elif err.errno == 2003:  # Can't connect
            print("可能原因: 服务器地址错误或MySQL服务未运行")
            print("解决方案: 检查MySQL服务是否运行，以及主机名和端口是否正确")
        elif err.errno == 1049:  # Unknown database
            print("可能原因: 数据库不存在")
            print("解决方案: 创建数据库或检查数据库名称拼写")
        
        return False
    
    except Exception as e:
        print(f"❌ 其他错误: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_mysql_connection()
    sys.exit(0 if success else 1)