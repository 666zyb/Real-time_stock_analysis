import mysql.connector
import json


def init_mysql_database():
    # 读取配置文件
    with open('../config/config.json', 'r', encoding='utf-8') as f:
        config = json.load(f)

    mysql_config = config['mysql_config']

    # 连接MySQL（不指定数据库）
    conn = mysql.connector.connect(
        host=mysql_config['host'],
        user=mysql_config['user'],
        password=mysql_config['password']
    )
    cursor = conn.cursor()

    try:
        # 创建数据库
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {mysql_config['database']}")
        cursor.execute(f"USE {mysql_config['database']}")

        # 创建股票基本信息表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS stocks (
                id INT PRIMARY KEY AUTO_INCREMENT,
                code VARCHAR(10) NOT NULL UNIQUE,
                name VARCHAR(50) NOT NULL,
                industry VARCHAR(50),
                listing_date DATE,
                total_shares BIGINT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
            )
        """)

        # 创建财务数据表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS financial_data (
                id INT PRIMARY KEY AUTO_INCREMENT,
                stock_code VARCHAR(10),
                report_date DATE,
                revenue DECIMAL(20,2),
                net_profit DECIMAL(20,2),
                eps DECIMAL(10,4),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (stock_code) REFERENCES stocks(code)
            )
        """)

        conn.commit()
        print("数据库初始化成功！")

    except Exception as e:
        print(f"初始化数据库时出错: {str(e)}")

    finally:
        cursor.close()
        conn.close()


if __name__ == "__main__":
    init_mysql_database()