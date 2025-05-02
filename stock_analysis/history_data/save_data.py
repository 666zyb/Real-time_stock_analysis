import csv
import mysql.connector
import os
import json


def load_config(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"错误: 未找到配置文件 {file_path}")
    except json.JSONDecodeError:
        print(f"错误: 配置文件 {file_path} 不是有效的 JSON 格式")
    return None


# 加载配置文件
config = load_config('../config/config.json')
if not config:
    exit(1)

# 连接到 MySQL
try:
    mysql_config = config['mysql_config']
    mydb = mysql.connector.connect(
        host=mysql_config['host'],
        port=mysql_config.get('port', 3306),
        user=mysql_config['user'],
        password=mysql_config['password'],
        database=mysql_config['database']
    )
    mycursor = mydb.cursor()
    print("成功连接到 MySQL")
except mysql.connector.Error as e:
    print(f"无法连接到 MySQL: {e}")


def create_mysql_table(stock_name):
    """创建 MySQL 表"""
    table_name = f"{stock_name}_history"
    create_table_sql = f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
        `id` INT AUTO_INCREMENT PRIMARY KEY,  -- 添加主键字段
        `日期` DATE,
        `开盘价` DECIMAL(10, 2),
        `收盘价` DECIMAL(10, 2),
        `最高价` DECIMAL(10, 2),
        `最低价` DECIMAL(10, 2),
        `成交量(手)` INT,
        `成交额(元)` DECIMAL(20, 2),
        `振幅(%)` DECIMAL(10, 2),
        `涨跌幅(%)` DECIMAL(10, 2),
        `涨跌额(元)` DECIMAL(10, 2),
        `换手率(%)` DECIMAL(10, 2),
        `市盈率` DECIMAL(10, 4),
        `市盈率TTM` DECIMAL(10, 4),
        `市净率` DECIMAL(10, 4),
        `股息率` DECIMAL(10, 4),
        `股息率TTM` DECIMAL(10, 4),
        `市销率` DECIMAL(10, 4),
        `市销率TTM` DECIMAL(10, 4),
        `总市值` DECIMAL(20, 4),
        `总市值(元)` DECIMAL(20, 2),
        `流通市值(元)` DECIMAL(20, 2),
        `总股本(股)` BIGINT,
        `流通股本` BIGINT,
        UNIQUE KEY unique_date (`日期`)  -- 添加唯一约束
    )
    """
    mycursor.execute(create_table_sql)
    mydb.commit()


def insert_to_mysql(stock_name, data):
    """插入数据到 MySQL 表"""
    table_name = f"{stock_name}_history"
    columns = ', '.join([f'`{col.strip(" \ufeff")}`' for col in data[0].keys()])
    placeholders = ', '.join(['%s'] * len(data[0]))
    # 使用 INSERT IGNORE 避免重复插入
    sql = f"INSERT IGNORE INTO {table_name} ({columns}) VALUES ({placeholders})"
    values = []
    for row in data:
        row_values = []
        for value in row.values():
            if value == '':
                row_values.append(None)
            else:
                row_values.append(value)
        values.append(tuple(row_values))
    try:
        mycursor.executemany(sql, values)
        mydb.commit()
    except mysql.connector.Error as e:
        print(f"插入 {stock_name} 数据到 MySQL 时发生错误: {e}")


def process_csv_files(folder_path):
    """处理 CSV 文件"""
    stocks = config['stocks']
    stock_mapping = {f"{stock['code']}_{stock['name']}": stock['name'] for stock in stocks}
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            full_stock_name = os.path.splitext(filename)[0]
            if full_stock_name in stock_mapping:
                stock_name = stock_mapping[full_stock_name]
                file_path = os.path.join(folder_path, filename)
                # 创建 MySQL 表
                create_mysql_table(stock_name)
                data = []
                try:
                    with open(file_path, 'r', encoding='utf-8', newline='') as csvfile:
                        reader = csv.DictReader(csvfile)
                        for row in reader:
                            data.append(row)
                    # 插入数据到 MySQL
                    insert_to_mysql(stock_name, data)
                    print(f"成功处理 {stock_name} 的数据")
                except FileNotFoundError:
                    print(f"错误: 文件 {file_path} 未找到。")
                except Exception as e:
                    print(f"处理 {stock_name} 数据时发生未知错误: {e}")


if __name__ == "__main__":
    folder_path = '../history_data'  # 请替换为实际的 CSV 文件所在文件夹路径
    process_csv_files(folder_path)

    # 关闭连接
    mycursor.close()
    mydb.close()