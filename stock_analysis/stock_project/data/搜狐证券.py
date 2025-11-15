import requests
import json
import re
import os
import mysql.connector
from datetime import datetime


def fetch_stock_history_data(code, start_date, end_date):
    headers = {
        "accept": "*/*",
        "accept-language": "zh-CN,zh;q=0.9",
        "cache-control": "no-cache",
        "cookie": "gidinf=x099980109ee1a721da17347c0008fe8f14b6a13b2bc; SUV=1743169752407ml0b80; BIZ_MyLBS=cn_600085^%^2C^%^u540C^%^u4EC1^%^u5802^%^7Ccn_600082^%^2C^%^u6D77^%^u6CF0^%^u53D1^%^u5C55; reqtype=pc; _dfp=D8o6JLPDBbJ8s4YGJZebiFjIl3BzA1VfUiHABJ8v0b0=; clt=1744184321; cld=20250409153841; t=1744184548614^",
        "pragma": "no-cache",
        "referer": "https://q.stock.sohu.com/cn/600082/lshq.shtml",
        "sec-ch-ua": "^\\^Chromium^^;v=^\\^9^^, ^\\^Not?A_Brand^^;v=^\\^8^^^",
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": "^\\^Windows^^^",
        "sec-fetch-dest": "script",
        "sec-fetch-mode": "no-cors",
        "sec-fetch-site": "same-origin",
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 SLBrowser/9.0.6.2081 SLBChan/105 SLBVPV/64-bit"
    }
    url = "https://q.stock.sohu.com/hisHq"
    params = {
        "code": code,
        "start": start_date,
        "end": end_date,
        "stat": "1",
        "order": "D",
        "period": "d",
        "callback": "historySearchHandler",
        "rt": "jsonp",
    }
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        print(f"请求出错: {e}")
        return None


def parse_history_data(response_text):
    """解析响应文本中的股票历史数据"""
    try:
        # 使用正则表达式提取JSON部分
        pattern = r'historySearchHandler\((.*)\)'
        match = re.search(pattern, response_text)
        if not match:
            print("未找到匹配的JSON数据")
            return None

        json_str = match.group(1)
        data = json.loads(json_str)

        if not data or not isinstance(data, list) or len(data) == 0:
            print("解析的数据格式不正确")
            return None

        # 检查状态码
        if data[0].get("status") != 0:
            print(f"接口返回错误状态: {data[0].get('status')}")
            return None

        # 获取历史行情数据
        hq_data = data[0].get("hq", [])
        if not hq_data:
            print("没有历史行情数据")
            return None

        return hq_data
    except Exception as e:
        print(f"解析数据时出错: {e}")
        return None


def get_stock_name_from_code(code):
    """从股票代码获取股票名称"""
    # 去除可能的前缀
    stock_code = code.replace("cn_", "")

    # 加载配置文件获取股票名称
    try:
        # 尝试从配置文件获取股票信息
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config', 'config.json')
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        stocks = config.get('stocks', [])
        for stock in stocks:
            if stock.get('code') == stock_code:
                return stock.get('name')
        # 如果在配置中找不到，返回代码作为名称
        return stock_code
    except Exception as e:
        print(f"获取股票名称失败: {e}")
        return stock_code


def save_to_database(stock_name, history_data):
    """将历史数据保存到MySQL数据库"""
    # 获取MySQL配置
    try:
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config', 'config.json')
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        mysql_config = config.get('mysql_config', {})
    except Exception as e:
        print(f"读取配置文件失败: {e}")
        # 使用默认配置
        mysql_config = {
            'host': '172.16.0.3',
            'port': 3306,
            'user': 'root',
            'password': 'zyb123456668866',
            'database': 'stock_analysis'
        }

    try:
        # 连接到MySQL数据库
        conn = mysql.connector.connect(
            host=mysql_config.get('host', '127.0.0.1'),
            port=mysql_config.get('port', 3306),
            user=mysql_config.get('user', 'root'),
            password=mysql_config.get('password', ''),
            database=mysql_config.get('database', 'stock_analysis')
        )
        cursor = conn.cursor()

        # 创建表名（使用股票名称）
        table_name = f"{stock_name}_history"

        # 检查表是否存在
        check_table_query = f"""
        SELECT COUNT(*) 
        FROM information_schema.tables 
        WHERE table_schema = DATABASE() 
        AND table_name = '{table_name}'
        """
        cursor.execute(check_table_query)
        table_exists = cursor.fetchone()[0] > 0

        if not table_exists:
            # 如果表不存在，创建表
            create_table_query = f"""
            CREATE TABLE IF NOT EXISTS `{table_name}` (
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
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """
            cursor.execute(create_table_query)
            print(f"表 {table_name} 创建成功")
        else:
            print(f"表 {table_name} 已存在")

        # 插入数据
        insert_query = f"""
        INSERT INTO `{table_name}` 
        (`日期`, `开盘价`, `收盘价`, `最高价`, `最低价`, `成交量(手)`, `成交额(元)`, `涨跌额(元)`, `涨跌幅(%)`, `换手率(%)`)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
        `开盘价` = VALUES(`开盘价`),
        `收盘价` = VALUES(`收盘价`),
        `最高价` = VALUES(`最高价`),
        `最低价` = VALUES(`最低价`),
        `成交量(手)` = VALUES(`成交量(手)`),
        `成交额(元)` = VALUES(`成交额(元)`),
        `涨跌额(元)` = VALUES(`涨跌额(元)`),
        `涨跌幅(%)` = VALUES(`涨跌幅(%)`),
        `换手率(%)` = VALUES(`换手率(%)`)
        """

        # 准备数据
        records = []
        for record in history_data:
            # 数据格式: [日期, 开盘价, 收盘价, 涨跌额, 涨跌幅, 最低价, 最高价, 成交量, 成交额, 换手率]
            if len(record) >= 10:
                # 转换日期格式
                date_obj = datetime.strptime(record[0], '%Y-%m-%d')

                # 处理百分比格式
                change_rate = record[4].replace('%', '')
                turnover_rate = record[9].replace('%', '')

                # 转换数值
                open_price = float(record[1])
                close_price = float(record[2])
                change = float(record[3])
                low_price = float(record[5])
                high_price = float(record[6])
                volume = int(record[7])
                amount = float(record[8])

                records.append((
                    date_obj.strftime('%Y-%m-%d'),
                    open_price,
                    close_price,
                    high_price,
                    low_price,
                    volume,
                    amount,
                    change,
                    change_rate,
                    turnover_rate
                ))

        # 执行批量插入
        cursor.executemany(insert_query, records)
        conn.commit()

        print(f"成功插入/更新 {cursor.rowcount} 条记录到表 {table_name}")

        # 关闭连接
        cursor.close()
        conn.close()

        return True
    except Exception as e:
        print(f"保存到数据库时出错: {e}")
        import traceback
        traceback.print_exc()
        return False


def process_stock_history(stock_code, start_date, end_date, stock_name=None):
    """处理股票历史数据的完整流程"""
    # 1. 获取原始数据
    raw_data = fetch_stock_history_data(stock_code, start_date, end_date)
    if not raw_data:
        print(f"获取股票 {stock_code} 的历史数据失败")
        return False

    # 2. 解析数据
    history_data = parse_history_data(raw_data)
    if not history_data:
        print(f"解析股票 {stock_code} 的历史数据失败")
        return False

    # 3. 获取股票名称（如果未提供）
    stock_code_clean = stock_code.replace("cn_", "")
    if stock_name is None:
        stock_name = get_stock_name_from_code(stock_code)

    # 4. 保存数据到数据库
    success = save_to_database(stock_name, history_data)
    if success:
        print(f"股票 {stock_name}({stock_code_clean}) 的历史数据已成功保存到数据库")
    else:
        print(f"保存股票 {stock_name}({stock_code_clean}) 的历史数据到数据库失败")

    return success


if __name__ == "__main__":
    stock_code = "cn_601158"  # 例如: 600362 江西铜业
    start = "20241203"
    end = "20250407"

    # 完整处理
    process_stock_history(stock_code, start, end)
