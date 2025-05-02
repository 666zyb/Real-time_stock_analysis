import requests
import re
import ast
import mysql.connector
import json
import logging
import datetime
import os
from typing import List, Dict, Tuple, Optional

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class StockPageFetcher:
    def __init__(self, config_path: str = '../config/config.json'):
        self.headers = {
            "accept": "*/*",
            "accept-language": "zh-CN,zh;q=0.9",
            "cache-control": "no-cache",
            "pragma": "no-cache",
            "referer": "https://q.stock.sohu.com/",
            "sec-ch-ua": "\"Chromium\";v=\"9\", \"Not?A_Brand\";v=\"8\"",
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": "\"Windows\"",
            "sec-fetch-dest": "script",
            "sec-fetch-mode": "no-cors",
            "sec-fetch-site": "same-site",
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 SLBrowser/9.0.6.2081 SLBChan/105 SLBVPV/64-bit"
        }
        self.base_url = "https://hq.stock.sohu.com/cn/"
        self.stock_code = "cn_000001"
        self.config_path = config_path
        # 加载配置
        self.load_config()
        # 连接数据库
        self.connect_to_db()

    def load_config(self):
        """加载配置文件"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
            logger.info("配置文件加载成功")
        except Exception as e:
            logger.error(f"加载配置文件出错: {str(e)}")
            self.config = {
                "mysql_config": {
                    "host": "localhost",
                    "port": 3306,
                    "user": "root",
                    "password": "",
                    "database": "stock_analysis"
                },
                "stocks": []  # 确保有空的股票列表作为默认值
            }

    def connect_to_db(self):
        """连接到MySQL数据库"""
        try:
            mysql_config = self.config['mysql_config']
            self.db = mysql.connector.connect(
                host=mysql_config.get('host', 'localhost'),
                port=mysql_config.get('port', 3306),
                user=mysql_config.get('user', 'root'),
                password=mysql_config.get('password', ''),
                database=mysql_config.get('database', 'stock_analysis')
            )
            self.cursor = self.db.cursor()
            logger.info("数据库连接成功")
        except Exception as e:
            logger.error(f"数据库连接失败: {str(e)}")
            raise

    def set_headers(self, new_headers):
        self.headers = new_headers

    def set_stock_code(self, new_stock_code):
        """设置股票代码，如果不是cn_前缀格式，自动添加"""
        if not new_stock_code.startswith("cn_"):
            self.stock_code = f"cn_{new_stock_code}"
        else:
            self.stock_code = new_stock_code
        logger.info(f"设置股票代码: {self.stock_code}")

    def format_stock_code(self, code: str) -> str:
        """格式化股票代码，用于生成URL和表名"""
        # 去掉可能的cn_前缀
        clean_code = code.replace("cn_", "")

        if not clean_code.startswith(('sh', 'sz')):
            if clean_code.startswith('6'):
                return f'sh{clean_code}'
            elif clean_code.startswith(('0', '3')):
                return f'sz{clean_code}'
        return clean_code

    def get_stock_prev_close(self, stock_code: str) -> Optional[float]:
        """获取股票的昨日收盘价"""
        try:
            # 可以从新浪财经API获取昨日收盘价
            formatted_code = self.format_stock_code(stock_code)
            url = f"https://hq.sinajs.cn/list={formatted_code}"
            headers = {
                "Referer": "https://finance.sina.com.cn",
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }

            response = requests.get(url, headers=headers)
            response.raise_for_status()

            # 解析响应
            match = re.search(r'"([^"]+)"', response.text)
            if match:
                stock_data = match.group(1).split(',')
                # 索引2是昨日收盘价
                if len(stock_data) > 2:
                    return float(stock_data[2])

            logger.warning(f"无法获取股票 {stock_code} 的昨日收盘价")
            return None
        except Exception as e:
            logger.error(f"获取昨日收盘价失败: {str(e)}")
            return None

    def get_stock_pages_content(self):
        """获取股票的分时交易数据"""
        all_records = []
        # 提取股票代码最后三位作为URL的一部分
        code_suffix = self.stock_code.replace("cn_", "")[-3:]
        logger.info(f"开始获取股票 {self.stock_code} 的分时数据")

        for i in range(1, 17):
            if i == 16:
                url = f"{self.base_url}{code_suffix}/{self.stock_code}-3.html"
            else:
                url = f"{self.base_url}{code_suffix}/{self.stock_code}-3-{i}.html"

            try:
                response = requests.get(url, headers=self.headers)
                response.raise_for_status()
                # 提取 deal_data 中的数据
                deal_data_pattern = re.compile(r'deal_data\((.*?)\)')
                match = deal_data_pattern.search(response.text)
                if match:
                    deal_data_str = match.group(1)
                    try:
                        deal_data = ast.literal_eval(deal_data_str)
                        # 仅保留交易记录部分，跳过分组标识和时间段等信息
                        records = [r for r in deal_data[2:] if len(r) == 5]
                        all_records.extend(records)
                        logger.info(f"从 {url} 获取到 {len(records)} 条交易记录")
                    except (SyntaxError, ValueError) as e:
                        logger.error(f"解析交易数据时出错: {str(e)}")
            except requests.exceptions.HTTPError as http_err:
                logger.error(f"请求 {url} 时发生HTTP错误: {http_err}")
            except requests.exceptions.RequestException as req_err:
                logger.error(f"请求 {url} 时发生请求错误: {req_err}")
            except Exception as e:
                logger.error(f"处理 {url} 时发生错误: {e}")

        logger.info(f"共获取到 {len(all_records)} 条交易记录")
        return all_records

    def save_to_database(self, stock_code: str, records: List) -> bool:
        """将分时数据保存到数据库"""
        if not records:
            logger.warning("没有数据可保存")
            return False

        try:
            # 获取格式化后的股票代码(添加sh或sz前缀)
            pure_code = stock_code.replace("cn_", "")
            formatted_code = self.format_stock_code(pure_code)
            table_name = f"stock_{formatted_code}_realtime"
            logger.info(f"准备保存数据到表 {table_name}")

            # 获取昨日收盘价
            prev_close = self.get_stock_prev_close(pure_code)
            if not prev_close:
                prev_close = 0
                logger.warning(f"使用默认值0作为股票 {stock_code} 的昨日收盘价")

            # 确保表存在
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
            self.cursor.execute(create_table_sql)

            # 获取今日日期
            today_date = datetime.datetime.now().strftime("%Y-%m-%d")

            # 获取今日开盘价（第一条记录的价格）
            today_open = records[0][1] if records else "0"

            # 计算最低价（所有记录中的最小价格）
            try:
                low_price = min([float(item[1]) for item in records])
            except:
                low_price = 0

            # 保存每条记录
            inserted_count = 0
            for record in records:
                # 解析记录数据: ['时间', '成交价', '涨跌', '成交量(手)', '成交金额(万元)']
                time_str, price, change_percent, volume, amount = record

                # 构建完整时间戳
                timestamp = f"{today_date}-{time_str}"

                # 检查是否已存在
                check_sql = f"SELECT * FROM {table_name} WHERE `时间` = %s"
                self.cursor.execute(check_sql, (timestamp,))
                exists = self.cursor.fetchone()

                if exists:
                    continue

                # 构建数据记录
                data = {
                    "时间": timestamp,
                    "今日开盘价": abs(float(today_open)),
                    "昨日收盘价": str(prev_close),
                    "当前价格": abs(float(price)),
                    "今日最低价": abs(float(low_price)),
                    "竞买价": "0",
                    "竞卖价": "0",
                    "成交量(手)": volume,
                    "成交额(元)": amount,
                    "买一委托量": "0",
                    "买一报价": "0",
                    "买二委托量": "0",
                    "买二报价": "0",
                    "买三委托量": "0",
                    "买四委托量": "0",
                    "买四报价": "0",
                    "买五委托量": "0",
                    "买五报价": "0",
                    "卖一委托量": "0",
                    "卖一报价": "0",
                    "卖二报价": "0",
                    "卖三委托量": "0",
                    "卖三报价": "0",
                    "卖四委托量": "0",
                    "卖五委托量": "0",
                    "卖五报价": "0",
                    "日期": today_date,
                    "其他保留字段": change_percent
                }

                # 插入数据
                columns = ', '.join([f'`{col}`' for col in data.keys()])
                placeholders = ', '.join(['%s'] * len(data))

                insert_sql = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"
                self.cursor.execute(insert_sql, tuple(data.values()))
                inserted_count += 1

            # 提交事务
            self.db.commit()
            logger.info(f"成功将股票 {stock_code} 的分时数据保存到数据库，共 {inserted_count} 条记录")
            return True

        except Exception as e:
            self.db.rollback()
            logger.error(f"保存分时数据到数据库失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def close(self):
        """关闭数据库连接"""
        if hasattr(self, 'cursor') and self.cursor:
            self.cursor.close()
        if hasattr(self, 'db') and self.db:
            self.db.close()
        logger.info("数据库连接已关闭")

    def process_all_stocks(self):
        """处理配置文件中所有股票的分时数据"""
        success_count = 0
        failure_count = 0

        # 获取配置文件中的股票列表
        stocks = self.config.get('stocks', [])
        logger.info(f"从配置文件中获取到 {len(stocks)} 只股票")

        for stock in stocks:
            stock_code = stock.get('code')
            stock_name = stock.get('name')

            if not stock_code:
                logger.warning("发现缺少股票代码的配置项，跳过")
                continue

            logger.info(f"开始处理股票 {stock_name}({stock_code}) 的分时数据")

            try:
                # 设置股票代码
                self.set_stock_code(stock_code)

                # 获取分时数据
                records = self.get_stock_pages_content()

                if records:
                    # 保存到数据库
                    success = self.save_to_database(stock_code, records)
                    if success:
                        logger.info(f"股票 {stock_name}({stock_code}) 的分时数据处理成功")
                        success_count += 1
                    else:
                        logger.error(f"股票 {stock_name}({stock_code}) 的分时数据保存失败")
                        failure_count += 1
                else:
                    logger.warning(f"未获取到股票 {stock_name}({stock_code}) 的分时数据")
                    failure_count += 1
            except Exception as e:
                logger.error(f"处理股票 {stock_name}({stock_code}) 的分时数据时出错: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
                failure_count += 1

        logger.info(f"全部处理完成，成功: {success_count}，失败: {failure_count}")
        return success_count > 0


def process_stock_minute_data(stock_code: str) -> bool:
    """处理指定股票分时数据的主函数 - 用于外部调用"""
    fetcher = StockPageFetcher()
    try:
        # 设置股票代码
        fetcher.set_stock_code(stock_code)

        # 获取分时数据
        records = fetcher.get_stock_pages_content()

        if records:
            # 保存到数据库
            success = fetcher.save_to_database(stock_code, records)
            return success
        else:
            logger.warning(f"未获取到股票 {stock_code} 的分时数据")
            return False
    except Exception as e:
        logger.error(f"处理股票 {stock_code} 的分时数据失败: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False
    finally:
        fetcher.close()


def process_all_stocks_from_config(config_path: str = '../config/config.json') -> bool:
    """处理配置文件中所有股票分时数据的主函数"""
    fetcher = StockPageFetcher(config_path)
    try:
        return fetcher.process_all_stocks()
    except Exception as e:
        logger.error(f"处理所有股票的分时数据失败: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False
    finally:
        fetcher.close()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # 如果提供了参数，则作为指定的股票代码处理
        stock_code = sys.argv[1]
        logger.info(f"从命令行获取股票代码: {stock_code}")
        success = process_stock_minute_data(stock_code)

        if success:
            logger.info(f"股票 {stock_code} 的分时数据处理成功")
        else:
            logger.error(f"股票 {stock_code} 的分时数据处理失败")
    else:
        # 没有提供参数，则处理配置文件中的所有股票
        logger.info("没有提供股票代码，将处理配置文件中的所有股票")

        # 获取当前文件所在目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # 构建配置文件路径
        config_path = os.path.join(current_dir, '..', 'config', 'config.json')
        # 规范化路径
        config_path = os.path.normpath(config_path)

        logger.info(f"使用配置文件: {config_path}")
        success = process_all_stocks_from_config(config_path)

        if success:
            logger.info("所有股票的分时数据处理完成")
        else:
            logger.error("处理所有股票的分时数据时出现错误")

