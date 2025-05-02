import time

import pandas as pd
import mysql.connector
import json
import asyncio
import logging
import os.path
import subprocess
import sys
from typing import Dict, Optional
from datetime import datetime, time as dt_time, timedelta
import redis
from 新浪财经股票数据 import Sina_stock
from 百度股市通每日行情 import OptimizedFinanceDataFetcher  # 导入百度股市通模块
from 搜狐证券分时数据 import process_stock_minute_data  # 导入异步版的处理函数
import numpy as np

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class StockAnalyzer:
    def __init__(self, config_path: str = '../config/config.json'):
        self.config_path = config_path
        self.config_last_modified = 0
        self.load_config(config_path)
        # 仅保留Redis连接
        self.redis_client = redis.Redis(
            host='localhost',
            port=6379,
            db=0,
            decode_responses=True
        )
        # 初始化数据源
        self.sina_source = Sina_stock()

        # 连接到 MySQL
        self.mysql_config = self.config['mysql_config']
        self.mydb = mysql.connector.connect(
            host=self.mysql_config['host'],
            port=self.mysql_config.get('port', 3306),
            user=self.mysql_config['user'],
            password=self.mysql_config['password'],
            database=self.mysql_config['database']
        )
        self.mycursor = self.mydb.cursor()
        # 跟踪每日汇总状态
        self.daily_summary_completed = {}

        # 用于存储每个股票的技术指标计算状态
        self.tech_indicators_state = {}
        # 记录上次初始化技术指标状态的日期
        self.last_init_date = None

    def load_config(self, config_path: str):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        # 保存配置文件的最后修改时间
        self.config_last_modified = os.path.getmtime(config_path)
        logger.info(f"配置文件加载成功，最后修改时间: {datetime.fromtimestamp(self.config_last_modified)}")

    async def process_technical_indicators(self, stock_name):
        """异步处理单个股票的技术指标数据"""
        try:
            # 获取技术指标分析脚本的路径
            script_dir = os.path.dirname(os.path.abspath(__file__))
            indicators_dir = os.path.join(script_dir, '../indicator_analysis')

            # 确保路径存在
            if not os.path.exists(indicators_dir):
                logger.error(f"技术指标目录不存在: {indicators_dir}")
                return False

            # 设置MySQL连接信息
            mysql_config = self.mysql_config

            # 创建一个单独的MySQL连接用于计算技术指标
            conn = mysql.connector.connect(
                host=mysql_config['host'],
                port=mysql_config.get('port', 3306),
                user=mysql_config['user'],
                password=mysql_config['password'],
                database=mysql_config['database']
            )

            # 读取历史数据
            cursor = conn.cursor()
            history_table_name = f"{stock_name}_history"
            query = f"SELECT `日期`, `开盘价`, `收盘价`, `最高价`, `最低价`, `成交量(手)` FROM {history_table_name} ORDER BY `日期` DESC LIMIT 700"
            cursor.execute(query)
            rows = cursor.fetchall()

            # 获取列名
            columns = ['日期', '开盘价', '收盘价', '最高价', '最低价', '成交量(手)']

            # 创建DataFrame
            data = pd.DataFrame(rows, columns=columns)

            # 确保日期列是日期类型
            data['日期'] = pd.to_datetime(data['日期'])

            # 按日期升序排序，以便正确计算技术指标
            data = data.sort_values('日期')

            # 转换收盘价为数值类型，确保可以进行计算
            data['收盘价'] = pd.to_numeric(data['收盘价'], errors='coerce')

            # 定义计算函数
            def calculate_macd(data, short_window=12, long_window=26, signal_window=9):
                short_ema = data['收盘价'].ewm(span=short_window, adjust=False).mean()
                long_ema = data['收盘价'].ewm(span=long_window, adjust=False).mean()
                macd = (short_ema - long_ema) * 2  # 使用*2倍增效果，与技术指标脚本保持一致
                signal = macd.ewm(span=signal_window, adjust=False).mean()
                hist = macd - signal
                return macd, signal, hist

            def calculate_rsi(data, period=14):
                delta = data['收盘价'].diff()
                up = delta.clip(lower=0)
                down = -delta.clip(upper=0)
                avg_gain = up.rolling(window=period).mean()
                avg_loss = down.rolling(window=period).mean()
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
                return rsi

            def calculate_bollinger_bands(data, window=20, std_dev=2):
                sma = data['收盘价'].rolling(window=window).mean()
                std = data['收盘价'].rolling(window=window).std()
                upper_band = sma + (std * std_dev)
                lower_band = sma - (std * std_dev)
                return upper_band, lower_band

            def calculate_ma(data, window=5):
                return data['收盘价'].rolling(window=window).mean()

            # 计算技术指标
            macd, signal, hist = calculate_macd(data)
            rsi = calculate_rsi(data)
            upper_band, lower_band = calculate_bollinger_bands(data)
            ma5 = calculate_ma(data, window=5)
            ma10 = calculate_ma(data, window=10)

            # 将计算结果添加到 DataFrame 中
            data['MACD'] = macd
            data['Signal'] = signal
            data['MACD_Hist'] = hist
            data['RSI'] = rsi
            data['Upper_Band'] = upper_band
            data['Lower_Band'] = lower_band
            data['MA5'] = ma5
            data['MA10'] = ma10

            # 使用前向填充和后向填充处理缺失值
            data = data.fillna(method='ffill').fillna(method='bfill')

            # 删除任何剩余的含有NaN的行
            data = data.dropna()

            # 只取最近的500条记录
            if len(data) > 500:
                data = data.iloc[-500:]

            # 创建新表
            technical_indicators_table_name = f"technical_indicators_{stock_name}"
            create_table_query = f"""
            CREATE TABLE IF NOT EXISTS `{technical_indicators_table_name}` (
                `日期` DATE,
                `MACD` DECIMAL(10, 4),
                `Signal` DECIMAL(10, 4),
                `MACD_Hist` DECIMAL(10, 4),
                `RSI` DECIMAL(10, 4),
                `Upper_Band` DECIMAL(10, 2),
                `Lower_Band` DECIMAL(10, 2),
                `MA5` DECIMAL(10, 2),
                `MA10` DECIMAL(10, 2),
                UNIQUE KEY unique_date (`日期`)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """
            cursor.execute(create_table_query)

            # 清空表中的数据
            cursor.execute(f"TRUNCATE TABLE `{technical_indicators_table_name}`")

            # 逐条插入数据，避免批量插入可能的问题
            success_count = 0
            for index, row in data.iterrows():
                try:
                    # 确保日期格式正确
                    date_str = row['日期'].strftime('%Y-%m-%d')

                    # 检查是否有NaN值并处理
                    if (pd.isna(row['MACD']) or pd.isna(row['Signal']) or
                            pd.isna(row['MACD_Hist']) or pd.isna(row['RSI']) or
                            pd.isna(row['Upper_Band']) or pd.isna(row['Lower_Band']) or
                            pd.isna(row['MA5']) or pd.isna(row['MA10'])):
                        continue

                    insert_query = f"""
                    INSERT INTO `{technical_indicators_table_name}` 
                    (`日期`, `MACD`, `Signal`, `MACD_Hist`, `RSI`, `Upper_Band`, `Lower_Band`, `MA5`, `MA10`)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """
                    values = (
                        date_str,
                        float(row['MACD']),
                        float(row['Signal']),
                        float(row['MACD_Hist']),
                        float(row['RSI']),
                        float(row['Upper_Band']),
                        float(row['Lower_Band']),
                        float(row['MA5']),
                        float(row['MA10'])
                    )
                    cursor.execute(insert_query, values)
                    success_count += 1
                except Exception as e:
                    logger.error(f"插入数据时出错: {e}, 股票: {stock_name}, 日期: {row['日期']}")
                    continue

            # 提交更改
            conn.commit()
            cursor.close()
            conn.close()

            logger.info(f"{stock_name} 技术指标计算并存储成功！共 {success_count} 条记录。")
            return True

        except Exception as e:
            logger.error(f"处理股票 {stock_name} 的技术指标数据时出错: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    async def reload_config_if_changed(self):
        """检查配置文件是否更新，如果更新则重新加载"""
        try:
            # 每5秒检查一次配置文件是否更新
            while True:
                current_modified_time = os.path.getmtime(self.config_path)
                if current_modified_time > self.config_last_modified:
                    logger.info(f"检测到配置文件已更新，重新加载配置...")
                    previous_config = self.config.copy()
                    self.load_config(self.config_path)

                    # 检查股票列表是否有变化
                    previous_stocks = {stock['code']: stock for stock in previous_config.get('stocks', [])}
                    current_stocks = {stock['code']: stock for stock in self.config.get('stocks', [])}

                    # 查找新增的股票
                    new_stocks = []
                    for code, stock in current_stocks.items():
                        if code not in previous_stocks:
                            new_stocks.append(stock)
                            logger.info(f"检测到新增股票: {stock['name']}({code})")

                    # 创建新增股票的数据表
                    for stock in new_stocks:
                        formatted_code = self.format_stock_code(stock['code'])
                        try:
                            # 1. 创建历史数据表
                            logger.info(f"为新增股票 {stock['name']}({stock['code']}) 创建历史数据表")
                            self.create_daily_summary_table(formatted_code)

                            # 2. 创建实时数据表
                            table_name = f"stock_{formatted_code}_realtime"
                            # create_table_sql = f"""
                            # CREATE TABLE IF NOT EXISTS {table_name} (
                            #     `时间` VARCHAR(255) PRIMARY KEY,
                            #     `今日开盘价` VARCHAR(255),
                            #     `昨日收盘价` VARCHAR(255),
                            #     `当前价格` VARCHAR(255),
                            #     `今日最低价` VARCHAR(255),
                            #     `竞买价` VARCHAR(255),
                            #     `竞卖价` VARCHAR(255),
                            #     `成交量(手)` VARCHAR(255),
                            #     `成交额(元)` VARCHAR(255),
                            #     `买一委托量` VARCHAR(255),
                            #     `买一报价` VARCHAR(255),
                            #     `买二委托量` VARCHAR(255),
                            #     `买二报价` VARCHAR(255),
                            #     `买三委托量` VARCHAR(255),
                            #     `买四委托量` VARCHAR(255),
                            #     `买四报价` VARCHAR(255),
                            #     `买五委托量` VARCHAR(255),
                            #     `买五报价` VARCHAR(255),
                            #     `卖一委托量` VARCHAR(255),
                            #     `卖一报价` VARCHAR(255),
                            #     `卖二报价` VARCHAR(255),
                            #     `卖三委托量` VARCHAR(255),
                            #     `卖三报价` VARCHAR(255),
                            #     `卖四委托量` VARCHAR(255),
                            #     `卖五委托量` VARCHAR(255),
                            #     `卖五报价` VARCHAR(255),
                            #     `日期` VARCHAR(255),
                            #     `其他保留字段` VARCHAR(255)
                            # )
                            # """
                            # 异步调用处理分时数据的函数
                            await process_stock_minute_data(stock['code'])

                            # self.mycursor.execute(create_table_sql)
                            self.mydb.commit()
                            logger.info(f"成功创建股票 {stock['name']}({stock['code']}) 的实时数据表")

                            # 3. 为新增股票创建技术指标表并计算技术指标
                            logger.info(f"开始处理股票 {stock['name']} 的技术指标数据...")
                            result = await self.process_technical_indicators(stock['name'])
                            if result:
                                logger.info(f"股票 {stock['name']} 的技术指标数据处理完成")
                            else:
                                logger.warning(f"股票 {stock['name']} 的技术指标数据处理失败")

                        except Exception as e:
                            logger.error(f"为新增股票 {stock['name']}({stock['code']}) 创建数据表失败: {str(e)}")

                    logger.info("配置更新完成，继续处理...")

                await asyncio.sleep(5)  # 等待10秒再检查

        except Exception as e:
            logger.error(f"监控配置文件变化时出错: {str(e)}")
            await asyncio.sleep(30)  # 出错后等待30秒再试

    async def _update_redis_data(self, stock_info, formatted_code):
        """更新Redis数据"""
        try:
            realtime_data = {
                "时间": stock_info["日期"] + '-' + stock_info["时间"],
                "今日开盘价": stock_info["今日开盘价"],
                "昨日收盘价": stock_info["昨日收盘价"],
                "当前价格": stock_info["当前价格"],
                "今日最低价": stock_info["今日最低价"],
                "竞买价": stock_info["竞买价"],
                "竞卖价": stock_info["竞卖价"],
                "成交量(手)": stock_info["成交量(手)"],
                "成交额(元)": stock_info["成交额(元)"],
                "买一委托量": stock_info["买一委托量"],
                "买一报价": stock_info["买一报价"],
                "买二委托量": stock_info["买二委托量"],
                "买二报价": stock_info["买二报价"],
                "买三委托量": stock_info["买三委托量"],
                "买四委托量": stock_info["买四委托量"],
                "买四报价": stock_info["买四报价"],
                "买五委托量": stock_info["买五委托量"],
                "买五报价": stock_info["买五报价"],
                "卖一委托量": stock_info["卖一委托量"],
                "卖一报价": stock_info["卖一报价"],
                "卖二报价": stock_info["卖二报价"],
                "卖三委托量": stock_info["卖三委托量"],
                "卖三报价": stock_info["卖三报价"],
                "卖四委托量": stock_info["卖四委托量"],
                "卖五委托量": stock_info["卖五委托量"],
                "卖五报价": stock_info["卖五报价"],
                "日期": stock_info["日期"],
                "其他保留字段": stock_info["其他保留字段"],
            }

            self.redis_client.hset(
                f"stock:realtime:{formatted_code}",
                mapping=realtime_data
            )

        except Exception as e:
            logger.error(f"更新Redis数据失败: {str(e)}")

    def format_stock_code(self, code: str) -> str:
        """格式化股票代码"""
        if not code.startswith(('sh', 'sz')):
            if code.startswith('6'):
                return f'sh{code}'
            elif code.startswith(('0', '3')):
                return f'sz{code}'
        return code

    async def collect_all_data(self):
        """收集实时数据"""
        while True:  # 改为持续运行的实时采集模式
            for stock in self.config['stocks']:
                try:
                    formatted_code = self.format_stock_code(stock['code'])
                    await self.collect_realtime_data(formatted_code)
                    await asyncio.sleep(self.config['settings']['realtime_interval'])
                except Exception as e:
                    logger.error(f"处理股票 {stock['code']} 时出错: {str(e)}")

    async def collect_realtime_data(self, formatted_code: str):
        """获取实时数据（新浪数据源）"""
        try:
            logger.info(f"开始获取股票 {formatted_code} 的实时数据")
            stock_info = self.sina_source.get_stock_data(formatted_code)

            if stock_info:
                await self._update_redis_data(stock_info, formatted_code)
                logger.info(f"成功写入股票 {formatted_code} 的实时数据")
                # 存储到 MySQL
                self.save_redis_to_mysql(formatted_code)

                # 获取股票名称
                stock_name = self.get_stock_name_from_config(formatted_code)

                # 计算并保存实时技术指标
                await self.calculate_realtime_technical_indicators(formatted_code, stock_name, stock_info)
            else:
                logger.warning(f"未获取到股票 {formatted_code} 的实时数据")

        except Exception as e:
            logger.error(f"获取实时数据失败: {str(e)}")

    async def calculate_realtime_technical_indicators(self, formatted_code, stock_name, current_stock_info):
        """增量计算并保存实时技术指标数据"""
        try:
            # 获取当前日期和时间
            current_date = datetime.now().date().strftime('%Y-%m-%d')

            # 检查今天是否已初始化技术指标状态
            if self.last_init_date is None or self.last_init_date.strftime('%Y-%m-%d') != current_date:
                # 每天早上检查并初始化
                await self.check_and_initialize_indicators()

            # 如果股票没有计算状态，进行初始化
            if stock_name not in self.tech_indicators_state:
                logger.info(f"股票 {stock_name} 的技术指标状态不存在，开始初始化...")
                success = self.initialize_technical_indicators_state(stock_name)
                if not success:
                    logger.warning(f"无法计算股票 {stock_name} 的实时技术指标，初始化状态失败")
                    return

            # 创建realtime_technical表
            table_name = f"realtime_technical_{stock_name}"
            create_table_sql = f"""
            CREATE TABLE IF NOT EXISTS `{table_name}` (
                `时间` VARCHAR(255) PRIMARY KEY,
                `当前价格` DECIMAL(10, 2),
                `MACD` DECIMAL(10, 4),
                `Signal` DECIMAL(10, 4),
                `MACD_Hist` DECIMAL(10, 4),
                `RSI` DECIMAL(10, 4),
                `Upper_Band` DECIMAL(10, 2),
                `Lower_Band` DECIMAL(10, 2),
                `MA5` DECIMAL(10, 2),
                `MA10` DECIMAL(10, 2),
                `日期` DATE
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """
            self.mycursor.execute(create_table_sql)
            self.mydb.commit()

            # 获取当前价格
            current_price = float(current_stock_info["当前价格"])
            time_val = current_stock_info["日期"] + '-' + current_stock_info["时间"]
            date_val = current_stock_info["日期"]

            # 获取技术指标计算状态
            state = self.tech_indicators_state[stock_name]

            # 使用增量计算更新技术指标
            # 1. 更新价格列表
            state['prices'].append(current_price)
            state['dates'].append(date_val)
            if len(state['prices']) > 100:  # 保持最近100个价格点
                state['prices'].pop(0)
                state['dates'].pop(0)

            # 2. 增量计算MACD
            short_span = 12  # 定义短期EMA周期
            long_span = 26  # 定义长期EMA周期
            signal_span = 9  # 定义信号线周期

            alpha_short = 2 / (short_span + 1)
            alpha_long = 2 / (long_span + 1)
            alpha_signal = 2 / (signal_span + 1)

            # 更新EMA
            new_short_ema = current_price * alpha_short + state['short_ema'] * (1 - alpha_short)
            new_long_ema = current_price * alpha_long + state['long_ema'] * (1 - alpha_long)
            new_macd = (new_short_ema - new_long_ema) * 2
            new_signal = new_macd * alpha_signal + state['signal'] * (1 - alpha_signal)
            macd_hist = new_macd - new_signal

            # 更新状态
            state['short_ema'] = new_short_ema
            state['long_ema'] = new_long_ema
            state['signal'] = new_signal

            # 3. 增量计算RSI
            last_price = state['last_price']
            price_change = current_price - last_price

            if price_change > 0:
                current_gain = price_change
                current_loss = 0
            else:
                current_gain = 0
                current_loss = -price_change

            # 更新平均gain和loss
            new_avg_gain = (state['avg_gain'] * 13 + current_gain) / 14
            new_avg_loss = (state['avg_loss'] * 13 + current_loss) / 14

            # 计算RSI
            if new_avg_loss == 0:
                rsi = 100
            else:
                rs = new_avg_gain / new_avg_loss
                rsi = 100 - (100 / (1 + rs))

            # 更新状态
            state['avg_gain'] = new_avg_gain
            state['avg_loss'] = new_avg_loss
            state['last_price'] = current_price

            # 4. 增量计算布林带
            # 使用完整窗口的价格列表计算布林带
            prices_array = np.array(state['prices'][-20:])
            if len(prices_array) < 20:
                # 如果数据不够，使用之前计算的值
                sma20 = state['sma20']
                upper_band = sma20 + (state['std20'] * 2)
                lower_band = sma20 - (state['std20'] * 2)
            else:
                # 重新计算SMA和标准差
                sma20 = np.mean(prices_array)
                std20 = np.std(prices_array)
                upper_band = sma20 + (std20 * 2)
                lower_band = sma20 - (std20 * 2)
                # 更新状态
                state['sma20'] = sma20
                state['std20'] = std20

            # 5. 增量计算MA
            state['ma5_values'].append(current_price)
            state['ma10_values'].append(current_price)
            if len(state['ma5_values']) > 5:
                state['ma5_values'].pop(0)
            if len(state['ma10_values']) > 10:
                state['ma10_values'].pop(0)

            ma5 = np.mean(state['ma5_values']) if len(state['ma5_values']) == 5 else None
            ma10 = np.mean(state['ma10_values']) if len(state['ma10_values']) == 10 else None

            # 更新最后计算时间
            state['last_update'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            # 检查是否已存在该时间点的记录
            check_sql = f"SELECT * FROM `{table_name}` WHERE `时间` = %s"
            self.mycursor.execute(check_sql, (time_val,))
            existing_record = self.mycursor.fetchone()

            if existing_record:
                # 更新现有记录
                update_sql = f"""
                UPDATE `{table_name}`
                SET `当前价格` = %s, `MACD` = %s, `Signal` = %s, `MACD_Hist` = %s, 
                    `RSI` = %s, `Upper_Band` = %s, `Lower_Band` = %s, `MA5` = %s, `MA10` = %s
                WHERE `时间` = %s
                """
                values = (
                    current_price,
                    new_macd,
                    new_signal,
                    macd_hist,
                    rsi,
                    upper_band,
                    lower_band,
                    ma5,
                    ma10,
                    time_val
                )
                self.mycursor.execute(update_sql, values)
            else:
                # 插入新记录
                insert_sql = f"""
                INSERT INTO `{table_name}` 
                (`时间`, `当前价格`, `MACD`, `Signal`, `MACD_Hist`, `RSI`, `Upper_Band`, `Lower_Band`, `MA5`, `MA10`, `日期`)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """
                values = (
                    time_val,
                    current_price,
                    new_macd,
                    new_signal,
                    macd_hist,
                    rsi,
                    upper_band,
                    lower_band,
                    ma5,
                    ma10,
                    date_val
                )
                self.mycursor.execute(insert_sql, values)

            self.mydb.commit()
            logger.info(f"成功保存股票 {stock_name} 的实时技术指标数据（增量计算）")

        except Exception as e:
            logger.error(f"计算并保存股票 {stock_name} 的实时技术指标时出错: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())

    async def realtime_monitor(self):
        """实时监控股票数据"""
        while True:
            try:
                current_time = datetime.now().time()
                current_date = datetime.now().date()
                date_key = current_date.strftime('%Y-%m-%d')  # 获取当前日期的字符串格式

                # 检查是否在15点之后
                if current_time >= dt_time(15, 0):
                    logger.info("当前时间在15点之后，检查历史数据中是否存在今日的股票行情并清除昨日数据...")

                    for stock in self.config['stocks']:
                        formatted_code = self.format_stock_code(stock['code'])
                        stock_name = self.get_stock_name_from_config(formatted_code)
                        table_name = f"{stock_name}_history"

                        # 检查今日数据是否存在
                        check_sql = f"SELECT * FROM `{table_name}` WHERE `日期` = %s"
                        self.mycursor.execute(check_sql, (current_date,))
                        result = self.mycursor.fetchone()
                        if not result:
                            logger.info(f"今日行情数据不存在，开始从百度股市通获取 {formatted_code} 的数据...")
                            fetcher = OptimizedFinanceDataFetcher(stock['code'])  # 创建爬虫实例
                            baidu_data = fetcher.get_data()  # 获取数据

                            if baidu_data and isinstance(baidu_data, dict):  # 确保数据有效
                                # 生成汇总数据并插入
                                summary_data = {
                                    '日期': current_date,
                                    '开盘价': baidu_data.get('今开', 0),
                                    '收盘价': baidu_data.get('今收', 0),
                                    '最高价': baidu_data.get('最高', 0),
                                    '最低价': baidu_data.get('最低', 0),
                                    '成交量(手)': baidu_data.get('成交量', 0),
                                    '振幅(%)': baidu_data.get('振幅', 0),
                                    '涨跌幅(%)': baidu_data.get('涨跌幅', 0),
                                    '换手率(%)': baidu_data.get('换手率', 0),
                                    '市盈率TTM': baidu_data.get('市盈(TTM)', 0),
                                    '市盈率': baidu_data.get('市盈(静)', 0),
                                    '市净率': baidu_data.get('市净率', 0),
                                    '总市值': str(round(float(baidu_data.get('总市值', 0)) / 10000, 4)),
                                    '总市值(元)': baidu_data.get('总市值', 0),
                                    '总股本(股)': baidu_data.get('总股本', 0),
                                    '流通股本': baidu_data.get('流通股', 0),
                                    '流通市值(元)': baidu_data.get('流通值', 0),
                                    '成交额(元)': baidu_data.get('成交额', 0),
                                    '涨跌额(元)': str(float(baidu_data.get('今收', 0)) - float(baidu_data.get('昨收', 0))),
                                    # 其他字段可以根据需要添加
                                }
                                # 保存到数据库
                                successd_status = self.save_daily_summary(formatted_code, summary_data)
                                if successd_status:
                                    self.clean_realtime_data(formatted_code)  # 只传递 formatted_code
                                    logger.info(f"已经删除{formatted_code}昨日实时数据")
                            else:
                                logger.warning(f"从百度股市通获取 {formatted_code} 的数据失败，无法插入今日行情数据")
                # 检查是否在交易时间内
                if self.is_trading_time(current_time):
                    logger.info("当前处于交易时间，开始获取实时数据...")
                    for stock in self.config['stocks']:
                        formatted_code = self.format_stock_code(stock['code'])
                        await self.collect_realtime_data(formatted_code)

                    # 等待指定的刷新间隔
                    await asyncio.sleep(self.config['settings']['realtime_interval'])
                else:
                    logger.info(f"当前时间 {current_time} 不在交易时间内，等待中...")
                    # 非交易时间，等待到下一个检查点
                    await asyncio.sleep(60)  # 每分钟检查一次是否进入交易时间

            except Exception as e:
                logger.error(f"实时监控出错: {str(e)}")
                await asyncio.sleep(5)  # 出错后等待5秒再重试

    def is_trading_time(self, current_time: dt_time) -> bool:
        """判断是否为交易时间"""
        # 定义交易时间段
        morning_start = dt_time(9, 30)
        morning_end = dt_time(11, 30)
        afternoon_start = dt_time(13, 0)
        afternoon_end = dt_time(15, 0)

        # 判断是否在交易时间内
        is_morning_session = morning_start <= current_time <= morning_end
        is_afternoon_session = afternoon_start <= current_time <= afternoon_end

        return is_morning_session or is_afternoon_session

    def save_redis_to_mysql(self, formatted_code):
        """将 Redis 数据存储到 MySQL，字段作为列，时间作为行"""
        try:
            # 从 Redis 读取数据
            redis_data = self.redis_client.hgetall(f"stock:realtime:{formatted_code}")
            time = redis_data['时间']

            # 构建列名和值
            columns = ', '.join([f'`{col}`' for col in redis_data.keys()])
            values_placeholders = ', '.join(['%s'] * len(redis_data))

            # 存储到 MySQL
            table_name = f"stock_{formatted_code}_realtime"
            create_table_sql = f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                `时间` VARCHAR(255) PRIMARY KEY,
                `今日开盘价` VARCHAR(255),
                `昨日收盘价` VARCHAR(255),
                `当前价格` VARCHAR(255),
                `今日最低价` VARCHAR(255),
                `竞买价` VARCHAR(255),
                `竞卖价` VARCHAR(255),
                `成交量(手)` VARCHAR(255),
                `成交额(元)` VARCHAR(255),
                `买一委托量` VARCHAR(255),
                `买一报价` VARCHAR(255),
                `买二委托量` VARCHAR(255),
                `买二报价` VARCHAR(255),
                `买三委托量` VARCHAR(255),
                `买四委托量` VARCHAR(255),
                `买四报价` VARCHAR(255),
                `买五委托量` VARCHAR(255),
                `买五报价` VARCHAR(255),
                `卖一委托量` VARCHAR(255),
                `卖一报价` VARCHAR(255),
                `卖二报价` VARCHAR(255),
                `卖三委托量` VARCHAR(255),
                `卖三报价` VARCHAR(255),
                `卖四委托量` VARCHAR(255),
                `卖五委托量` VARCHAR(255),
                `卖五报价` VARCHAR(255),
                `日期` VARCHAR(255),
                `其他保留字段` VARCHAR(255)
            )
            """
            self.mycursor.execute(create_table_sql)

            # 检查数据是否已存在
            check_sql = f"SELECT * FROM {table_name} WHERE `时间` = %s"
            self.mycursor.execute(check_sql, (time,))
            result = self.mycursor.fetchone()

            if not result:
                insert_sql = f"INSERT INTO {table_name} ({columns}) VALUES ({values_placeholders})"
                values = tuple(redis_data.values())
                self.mycursor.execute(insert_sql, values)
                self.mydb.commit()
                logger.info(f"成功将股票 {formatted_code} 的 Redis 数据存储到 MySQL")
            else:
                logger.info(f"股票 {formatted_code} 在 {time} 的数据已存在，跳过插入")
        except Exception as e:
            logger.error(f"将 Redis 数据存储到 MySQL 失败: {str(e)}")

    def create_daily_summary_table(self, formatted_code):
        """创建每日汇总数据表"""
        try:
            # 从配置中获取股票名称
            stock_name = self.get_stock_name_from_config(formatted_code)
            table_name = f"{stock_name}_history"

            create_table_sql = f"""
            CREATE TABLE IF NOT EXISTS `{table_name}` (
                `id` INT AUTO_INCREMENT PRIMARY KEY,
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
                UNIQUE KEY unique_date (`日期`)
            )
            """
            self.mycursor.execute(create_table_sql)
            self.mydb.commit()
            logger.info(f"成功创建或检查股票 {stock_name} 的历史数据表")
            return table_name
        except Exception as e:
            logger.error(f"创建历史数据表失败: {str(e)}")
            return None

    def get_stock_name_from_config(self, formatted_code):
        """从配置中获取股票名称"""
        try:
            # 移除前缀以匹配配置中的代码
            code = formatted_code.replace('sh', '').replace('sz', '')

            # 遍历配置中的股票列表找到匹配的名称
            for stock in self.config['stocks']:
                if stock['code'] == code:
                    return stock['name']

            # 如果找不到匹配的名称，返回代码作为名称
            logger.warning(f"在配置中找不到股票 {formatted_code} 的名称，使用代码作为表名")
            return f"stock_{formatted_code}"
        except Exception as e:
            logger.error(f"获取股票名称失败: {str(e)}")
            return f"stock_{formatted_code}"

    def save_daily_summary(self, formatted_code, summary_data):
        """保存每日汇总数据到数据库，将新数据插入到第一行"""
        try:
            # 获取股票名称作为表名前缀
            stock_name = self.get_stock_name_from_config(formatted_code)
            table_name = f"{stock_name}_history"

            # 检查是否已有当天的数据
            check_sql = f"SELECT * FROM `{table_name}` WHERE `日期` = %s"
            self.mycursor.execute(check_sql, (summary_data['日期'],))
            result = self.mycursor.fetchone()

            if not result:
                # 开始事务
                self.mycursor.execute("START TRANSACTION")

                # 1. 先将所有现有记录的ID加1，从最大ID开始递减处理，避免主键冲突
                update_ids_sql = f"""
                UPDATE `{table_name}` SET `id` = `id` + 1 
                ORDER BY `id` DESC
                """
                self.mycursor.execute(update_ids_sql)

                # 2. 重置AUTO_INCREMENT值
                reset_auto_increment_sql = f"""
                ALTER TABLE `{table_name}` AUTO_INCREMENT = 2
                """
                self.mycursor.execute(reset_auto_increment_sql)

                # 3. 构建插入语句，强制ID为1
                columns = ', '.join([f'`{col}`' for col in summary_data.keys()])
                placeholders = ', '.join(['%s'] * len(summary_data))

                # 插入新记录，强制ID=1
                insert_sql = f"INSERT INTO `{table_name}` (`id`, {columns}) VALUES (1, {placeholders})"
                values = tuple(summary_data.values())

                self.mycursor.execute(insert_sql, values)

                # 提交事务
                self.mydb.commit()
                logger.info(f"成功保存股票 {stock_name} 的历史数据到表的第一行")
            else:
                # 更新已有记录
                update_parts = ', '.join([f"`{col}` = %s" for col in summary_data.keys() if col != '日期'])
                update_values = [summary_data[col] for col in summary_data.keys() if col != '日期']
                update_values.append(summary_data['日期'])  # WHERE条件的值

                update_sql = f"UPDATE `{table_name}` SET {update_parts} WHERE `日期` = %s"

                self.mycursor.execute(update_sql, tuple(update_values))
                self.mydb.commit()
                logger.info(f"成功更新股票 {stock_name} 的历史数据")

            return True  # 返回成功标志

        except Exception as e:
            # 回滚事务
            try:
                self.mydb.rollback()
            except:
                pass
            logger.error(f"保存历史数据失败: {str(e)}")
            return False  # 返回失败标志

    def clean_realtime_data(self, formatted_code):
        """清理昨天的实时数据"""
        try:
            table_name = f"stock_{formatted_code}_realtime"
            # print(table_name)
            # 获取昨天的日期
            yesterday = (datetime.now() - timedelta(days=1)).date()
            # print(yesterday)

            # 删除昨天的实时数据
            delete_sql = f"DELETE FROM {table_name} WHERE DATE(`日期`) = %s"
            self.mycursor.execute(delete_sql, (yesterday,))
            affected_rows = self.mycursor.rowcount
            self.mydb.commit()

            logger.info(f"成功删除股票 {formatted_code} 在 {yesterday} 的实时数据，共 {affected_rows} 条记录")

            # # 可选：优化表结构
            # optimize_sql = f"OPTIMIZE TABLE {table_name}"
            # self.mycursor.execute(optimize_sql)
            # self.mycursor.fetchall()  # 读取所有结果，避免未读结果残留

            return affected_rows
        except Exception as e:
            logger.error(f"清理实时数据失败: {str(e)}")
            return 0

    def cleanup(self):
        self.redis_client.close()
        self.mycursor.close()
        self.mydb.close()

    def initialize_technical_indicators_state(self, stock_name):
        """初始化技术指标计算状态"""
        try:
            # 从数据库获取历史数据的后100条作为初始状态
            history_table_name = f"{stock_name}_history"
            query = f"""
            SELECT `日期`, `收盘价` 
            FROM `{history_table_name}` 
            ORDER BY `日期` DESC 
            LIMIT 100
            """
            self.mycursor.execute(query)
            rows = self.mycursor.fetchall()

            if not rows:
                logger.warning(f"没有找到股票 {stock_name} 的历史数据，无法初始化技术指标状态")
                return False

            # 创建DataFrame并按日期排序
            df = pd.DataFrame(rows, columns=['日期', '收盘价'])
            df['日期'] = pd.to_datetime(df['日期'])
            df['收盘价'] = pd.to_numeric(df['收盘价'], errors='coerce')
            df = df.sort_values('日期')

            # 计算初始技术指标状态
            prices = df['收盘价'].values
            dates = df['日期'].values

            # MACD计算状态
            short_span = 12
            long_span = 26
            signal_span = 9

            # 计算EMA初始值
            short_ema = df['收盘价'].ewm(span=short_span, adjust=False).mean().iloc[-1]
            long_ema = df['收盘价'].ewm(span=long_span, adjust=False).mean().iloc[-1]
            macd = (short_ema - long_ema) * 2

            # 计算信号线初始值
            macd_series = df['收盘价'].ewm(span=short_span, adjust=False).mean() - df['收盘价'].ewm(span=long_span, adjust=False).mean()
            macd_series = macd_series * 2
            signal = macd_series.ewm(span=signal_span, adjust=False).mean().iloc[-1]

            # RSI计算状态
            rsi_period = 14
            delta = df['收盘价'].diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            avg_gain = gain.rolling(window=rsi_period).mean().iloc[-1]
            avg_loss = loss.rolling(window=rsi_period).mean().iloc[-1]

            # 布林带计算状态
            bb_period = 20
            bb_std_dev = 2
            sma = df['收盘价'].rolling(window=bb_period).mean().iloc[-1]
            std = df['收盘价'].rolling(window=bb_period).std().iloc[-1]

            # MA计算状态
            ma5_values = df['收盘价'].rolling(window=5).mean().tail(5).values
            ma10_values = df['收盘价'].rolling(window=10).mean().tail(10).values

            # 保存计算状态
            self.tech_indicators_state[stock_name] = {
                'prices': list(prices[-30:]),  # 保留最近30个价格点
                'dates': [str(d)[:10] for d in dates[-30:]],  # 使用str()转换numpy.datetime64为字符串
                'last_price': float(prices[-1]),
                'short_ema': float(short_ema),
                'long_ema': float(long_ema),
                'signal': float(signal),
                'avg_gain': float(avg_gain) if not pd.isna(avg_gain) else 0.0,  # 添加空值检查
                'avg_loss': float(avg_loss) if not pd.isna(avg_loss) else 0.0,  # 添加空值检查
                'sma20': float(sma) if not pd.isna(sma) else 0.0,  # 添加空值检查
                'std20': float(std) if not pd.isna(std) else 0.0,  # 添加空值检查
                'ma5_values': [float(v) if not pd.isna(v) else 0.0 for v in ma5_values],  # 添加空值检查
                'ma10_values': [float(v) if not pd.isna(v) else 0.0 for v in ma10_values],  # 添加空值检查
                'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

            logger.info(f"成功初始化股票 {stock_name} 的技术指标计算状态")
            return True

        except Exception as e:
            logger.error(f"初始化股票 {stock_name} 的技术指标状态时出错: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    async def check_and_initialize_indicators(self):
        """检查并在每天开盘时初始化所有股票的技术指标状态"""
        try:
            current_date = datetime.now().date()
            current_time = datetime.now().time()

            # 如果是新的一天且当前时间在9:15到9:30之间(开盘前准备时间)
            if (self.last_init_date != current_date and
                    dt_time(9, 15) <= current_time <= dt_time(9, 30)):

                logger.info("开始初始化所有股票的技术指标计算状态...")
                self.tech_indicators_state = {}  # 清空旧状态

                for stock in self.config['stocks']:
                    stock_name = stock['name']
                    success = self.initialize_technical_indicators_state(stock_name)
                    if success:
                        logger.info(f"股票 {stock_name} 的技术指标状态初始化成功")
                    else:
                        logger.warning(f"股票 {stock_name} 的技术指标状态初始化失败")

                self.last_init_date = current_date
                logger.info("所有股票的技术指标计算状态初始化完成")

        except Exception as e:
            logger.error(f"检查并初始化技术指标状态时出错: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())


async def main():
    analyzer = StockAnalyzer()
    try:
        # 创建三个任务：配置文件监控、历史数据收集和实时监控
        tasks = [
            asyncio.create_task(analyzer.reload_config_if_changed()),
            asyncio.create_task(analyzer.collect_all_data()),
            asyncio.create_task(analyzer.realtime_monitor())
        ]

        # 等待所有任务完成
        await asyncio.gather(*tasks)
    except KeyboardInterrupt:
        logger.info("程序被用户中断")
    except Exception as e:
        logger.error(f"程序运行出错: {str(e)}")
        logger.error("详细错误信息: ", exc_info=True)
    finally:
        analyzer.cleanup()


if __name__ == "__main__":
    # 运行主程序
    asyncio.run(main())