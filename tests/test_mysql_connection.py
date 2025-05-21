#!/usr/bin/env python3
"""
测试MySQL连接脚本
"""

import mysql.connector
import sys
import os
import json
import time
from datetime import datetime

def test_write_performance(conn):
    """测试MySQL写入性能"""
    try:
        # 检查连接是否有效
        if not conn.is_connected():
            print("重新连接到MySQL...")
            conn.reconnect(attempts=3, delay=5)
        
        cursor = conn.cursor()
        
        # 创建测试表
        print("创建性能测试表...")
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS performance_test (
            id INT AUTO_INCREMENT PRIMARY KEY,
            sku_id VARCHAR(50) NOT NULL,
            price DECIMAL(10,2) NOT NULL,
            timestamp DATETIME NOT NULL
        )
        """)
        conn.commit()
        print("✅ 测试表创建成功")
        
        # 生成测试数据
        print("生成测试数据...")
        test_data = []
        for i in range(1000):  # 生成1000条测试数据
            test_data.append((
                f"SKU_{i:05d}",  # SKU_00001 格式
                round(100 + i * 0.1, 2),  # 价格从100开始，每次增加0.1
                datetime.now()
            ))
        
        # 批量写入测试
        print("执行批量写入测试...")
        start_time = time.time()
        
        cursor.executemany(
            """
            INSERT INTO performance_test 
            (sku_id, price, timestamp) 
            VALUES (%s, %s, %s)
            """,
            test_data
        )
        conn.commit()
        
        end_time = time.time()
        elapsed = end_time - start_time
        records_per_second = len(test_data) / elapsed
        
        print(f"✅ 批量写入完成:")
        print(f"  - 总记录数: {len(test_data)}")
        print(f"  - 总耗时: {elapsed:.2f}秒")
        print(f"  - 写入速度: {records_per_second:.2f}条/秒")
        
        # 清理测试数据
        print("清理测试数据...")
        cursor.execute("TRUNCATE TABLE performance_test")
        conn.commit()
        print("✅ 测试数据已清理")
        
        return True
        
    except mysql.connector.Error as err:
        print(f"性能测试失败: {err}")
        return False
    except Exception as e:
        print(f"未预期的错误: {e}")
        return False
    
    # 测试批量写入
    print("测试批量写入性能...")
    try:
        start_time = time.time()
        data = [(f"SKU{i}", i*10.5, datetime.now()) for i in range(1000)]
        cursor.executemany(
            "INSERT INTO performance_test (sku_id, price, timestamp) VALUES (%s, %s, %s)",
            data
        )
        conn.commit()
        elapsed = time.time() - start_time
        print(f"✅ 批量写入1000条记录耗时: {elapsed:.3f}秒")
    except mysql.connector.Error as err:
        print(f"批量写入测试失败: {err}")
        return False
    
    # 清理测试表
    cursor.execute("DROP TABLE IF EXISTS performance_test")
    return True

def test_transaction_handling(conn):
    """测试MySQL事务处理"""
    try:
        # 检查连接是否有效
        if not conn.is_connected():
            print("重新连接到MySQL...")
            conn.reconnect(attempts=3, delay=5)
            
        cursor = conn.cursor()
        
        # 创建测试表
        print("创建事务测试表...")
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS transaction_test (
            id INT AUTO_INCREMENT PRIMARY KEY,
            operation VARCHAR(50) NOT NULL,
            amount DECIMAL(10,2) NOT NULL,
            status VARCHAR(20) NOT NULL
        )
        """)
        conn.commit()
        print("✅ 事务测试表创建成功")
        
        # 测试1: 事务提交
        print("\n测试事务提交...")
        try:
            # 确保没有活跃事务
            conn.rollback()
            
            # 开始新事务
            conn.start_transaction()
            
            # 插入测试数据
            cursor.execute("""
            INSERT INTO transaction_test 
            (operation, amount, status) 
            VALUES ('deposit', 100.00, 'pending')
            """)
            
            # 提交事务
            conn.commit()
            print("✅ 事务已提交")
            
            # 验证数据已提交
            cursor.execute("SELECT COUNT(*) FROM transaction_test WHERE operation='deposit'")
            count = cursor.fetchone()[0]
            
            if count == 1:
                print("✅ 事务提交测试通过")
            else:
                print(f"❌ 事务提交测试失败: 预期1条记录，实际{count}条")
                return False
                
        except mysql.connector.Error as err:
            print(f"❌ 事务提交测试失败: {err}")
            return False
            
        # 测试2: 事务回滚
        print("\n测试事务回滚...")
        try:
            # 确保没有活跃事务
            conn.rollback()
            
            # 开始新事务
            conn.start_transaction()
            
            # 插入测试数据
            cursor.execute("""
            INSERT INTO transaction_test 
            (operation, amount, status) 
            VALUES ('withdraw', 50.00, 'pending')
            """)
            
            # 回滚事务
            conn.rollback()
            print("✅ 事务已回滚")
            
            # 验证数据已回滚（不应该存在）
            cursor.execute("SELECT COUNT(*) FROM transaction_test WHERE operation='withdraw'")
            count = cursor.fetchone()[0]
            
            if count == 0:
                print("✅ 事务回滚测试通过")
            else:
                print(f"❌ 事务回滚测试失败: 预期0条记录，实际{count}条")
                return False
                
        except mysql.connector.Error as err:
            print(f"❌ 事务回滚测试失败: {err}")
            return False
            
        # 清理测试数据
        print("\n清理事务测试数据...")
        # 确保没有活跃事务
        try:
            conn.rollback()
        except:
            pass
            
        cursor.execute("TRUNCATE TABLE transaction_test")
        conn.commit()
        print("✅ 事务测试数据已清理")
        
        return True
        
    except mysql.connector.Error as err:
        print(f"事务测试失败: {err}")
        # 确保清理任何未完成的事务
        try:
            conn.rollback()
        except:
            pass
        return False
    except Exception as e:
        print(f"未预期的错误: {e}")
        # 确保清理任何未完成的事务
        try:
            conn.rollback()
        except:
            pass
        return False
    
    # 测试事务提交
    print("测试事务提交...")
    try:
        conn.start_transaction()
        cursor.execute(
            "INSERT INTO transaction_test (operation, amount) VALUES (%s, %s)",
            ("deposit", 100.00)
        )
        conn.commit()
        print("✅ 事务提交成功")
    except mysql.connector.Error as err:
        print(f"事务提交测试失败: {err}")
        return False
    
    # 测试事务回滚
    print("测试事务回滚...")
    try:
        conn.start_transaction()
        cursor.execute(
            "INSERT INTO transaction_test (operation, amount) VALUES (%s, %s)",
            ("withdraw", 50.00)
        )
        # 故意回滚
        conn.rollback()
        
        # 验证记录数
        cursor.execute("SELECT COUNT(*) FROM transaction_test")
        count = cursor.fetchone()[0]
        if count == 1:  # 只有deposit记录
            print("✅ 事务回滚成功")
        else:
            print(f"❌ 事务回滚失败，记录数: {count}")
            return False
    except mysql.connector.Error as err:
        print(f"事务回滚测试失败: {err}")
        return False
    
    # 清理测试表
    cursor.execute("DROP TABLE IF EXISTS transaction_test")
    return True

def test_mysql_connection():
    """测试MySQL连接"""
    print("开始测试MySQL连接...")
    
    # 尝试从JSON配置文件加载
    config_file = 'config/database.json'
    if os.path.exists(config_file):
        try:
            with open(config_file) as f:
                config = json.load(f)
            mysql_config = {
                'host': config['mysql']['host'],
                'port': int(config['mysql']['port']),
                'user': config['mysql']['user'],
                'password': config['mysql']['password'],
                'database': config['mysql']['database']
            }
            print(f"从JSON配置文件加载MySQL配置: {config_file}")
        except Exception as e:
            print(f"从JSON配置文件加载失败: {str(e)}")
            mysql_config = None
    else:
        print(f"JSON配置文件不存在: {config_file}")
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
        
        print("MySQL连接测试完成，一切正常！")
        
        # 新增性能测试
        print("\n开始数据写入性能测试...")
        try:
            test_write_performance(conn)
        except Exception as e:
            print(f"❌ 性能测试失败: {str(e)}")
        
        # 新增事务测试
        print("\n开始事务处理测试...")
        try:
            test_transaction_handling(conn)
        except Exception as e:
            print(f"❌ 事务测试失败: {str(e)}")
        
        # 清理测试表
        print("\n清理测试表...")
        try:
            cursor.execute("DROP TABLE IF EXISTS performance_test")
            cursor.execute("DROP TABLE IF EXISTS transaction_test")
            conn.commit()
            print("✅ 测试表已删除")
        except mysql.connector.Error as err:
            print(f"❌ 清理测试表失败: {err}")
        
        # 关闭连接
        cursor.close()
        conn.close()
        print("✅ MySQL连接已关闭")
        
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