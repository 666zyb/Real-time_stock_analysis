import re
import ast
import logging
import datetime
import os
import json
import asyncio
import aiohttp
import aiomysql
from typing import List, Dict, Tuple, Optional, Any, Union

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
        self.db_pool = None
        self.config = None

    async def load_config(self):
        """异步加载配置文件"""
        try:
            if not os.path.exists(self.config_path):
                logger.error(f"配置文件不存在: {self.config_path}")
                # 尝试查找config文件夹
                base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                potential_config = os.path.join(base_dir, 'config', 'config.json')
                if os.path.exists(potential_config):
                    logger.info(f"找到备选配置文件: {potential_config}")
                    self.config_path = potential_config
                else:
                    logger.error("无法找到有效的配置文件")

            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
            logger.info(f"配置文件 {self.config_path} 加载成功")
            # 检查配置结构
            if 'mysql_config' not in self.config:
                logger.warning("配置中缺少mysql_config部分")
            if 'stocks' not in self.config:
                logger.warning("配置中缺少stocks部分")
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

    async def connect_to_db(self):
        """异步连接到MySQL数据库，创建连接池"""
        if self.config is None:
            await self.load_config()

        try:
            mysql_config = self.config.get('mysql_config', {})
            logger.info(f"准备连接到数据库: {mysql_config.get('host')}:{mysql_config.get('port')}")

            self.db_pool = await aiomysql.create_pool(
                host=mysql_config.get('host', 'localhost'),
                port=mysql_config.get('port', 3306),
                user=mysql_config.get('user', 'root'),
                password=mysql_config.get('password', ''),
                db=mysql_config.get('database', 'stock_analysis'),
                autocommit=False,
                charset='utf8mb4'
            )
            logger.info("数据库连接池创建成功")

            # 测试连接
            async with self.db_pool.acquire() as conn:
                async with conn.cursor() as cursor:
                    await cursor.execute("SELECT 1")
                    result = await cursor.fetchone()
                    logger.info(f"数据库连接测试: {result}")

        except Exception as e:
            logger.error(f"数据库连接失败: {str(e)}")
            raise

    def set_headers(self, new_headers: Dict[str, str]):
        """设置HTTP请求头"""
        self.headers = new_headers

    def set_stock_code(self, new_stock_code: str):
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

    async def get_stock_prev_close(self, stock_code: str) -> Optional[float]:
        """异步获取股票的昨日收盘价"""
        try:
            # 可以从新浪财经API获取昨日收盘价
            formatted_code = self.format_stock_code(stock_code)
            url = f"https://hq.sinajs.cn/list={formatted_code}"
            headers = {
                "Referer": "https://finance.sina.com.cn",
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }

            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        text = await response.text()
                        # 解析响应
                        match = re.search(r'"([^"]+)"', text)
                        if match:
                            stock_data = match.group(1).split(',')
                            # 检查数据格式
                            if len(stock_data) > 3:
                                # 索引2是昨日收盘价
                                try:
                                    prev_close = float(stock_data[2])
                                    logger.info(f"获取到股票 {stock_code} 的昨日收盘价: {prev_close}")
                                    return abs(prev_close)  # 确保价格为正数
                                except (ValueError, TypeError) as e:
                                    logger.error(f"昨日收盘价格式错误: {stock_data[2]}")

            logger.warning(f"无法获取股票 {stock_code} 的昨日收盘价，将使用默认值")
            return None
        except Exception as e:
            logger.error(f"获取昨日收盘价失败: {str(e)}")
        return None

    async def get_stock_pages_content(self) -> List[List[str]]:
        """异步获取股票的分时交易数据"""
        all_records = []
        # 提取股票代码最后三位作为URL的一部分
        code_suffix = self.stock_code.replace("cn_", "")[-3:]
        logger.info(f"开始获取股票 {self.stock_code} 的分时数据")

        async with aiohttp.ClientSession() as session:
            tasks = []
            for i in range(1, 17):
                if i == 16:
                    url = f"{self.base_url}{code_suffix}/{self.stock_code}-3.html"
                else:
                    url = f"{self.base_url}{code_suffix}/{self.stock_code}-3-{i}.html"
                tasks.append(self.fetch_page(session, url, i))

            # 并发执行所有请求
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # 处理结果
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"获取页面 {i + 1} 时出错: {str(result)}")
                elif result:
                    all_records.extend(result)

        if not all_records:
            logger.warning(f"未能获取到股票 {self.stock_code} 的任何分时数据")
        else:
            logger.info(f"共获取到 {len(all_records)} 条交易记录")
            # 打印前几条数据作为示例
            if len(all_records) > 0:
                logger.info(f"数据示例: {all_records[0]}")

        return all_records

    async def fetch_page(self, session: aiohttp.ClientSession, url: str, page_num: int) -> List[List[str]]:
        """异步获取单个页面的分时数据"""
        try:
            logger.debug(f"开始请求页面 {page_num}，URL: {url}")
            async with session.get(url, headers=self.headers) as response:
                if response.status == 200:
                    logger.debug(f"页面 {page_num} 请求成功")
                    text = await response.text()
                    # 提取 deal_data 中的数据
                    deal_data_pattern = re.compile(r'deal_data\((.*?)\)')
                    match = deal_data_pattern.search(text)
                    if match:
                        deal_data_str = match.group(1)
                        try:
                            deal_data = ast.literal_eval(deal_data_str)
                            # 仅保留交易记录部分，跳过分组标识和时间段等信息
                            records = [r for r in deal_data[2:] if len(r) == 5]
                            logger.info(f"从页面 {page_num} 获取到 {len(records)} 条交易记录")
                            return records
                        except (SyntaxError, ValueError) as e:
                            logger.error(f"解析页面 {page_num} 的交易数据时出错: {str(e)}")
                    else:
                        logger.warning(f"页面 {page_num} 没有找到deal_data部分")
                else:
                    logger.error(f"请求页面 {page_num} 返回状态码: {response.status}")
        except Exception as e:
            logger.error(f"处理页面 {page_num} 时发生错误: {e}")
        return []

    def validate_price(self, price_str: str) -> float:
        """验证并转换价格字符串为有效数字，确保为正数"""
        try:
            price = float(price_str)
            return abs(price)  # 确保价格为正数
        except (ValueError, TypeError):
            logger.warning(f"价格转换失败，原始值: {price_str}，使用默认值0")
            return 0.0

    async def save_to_database(self, stock_code: str, records: List[List[str]]) -> bool:
        """异步将分时数据保存到数据库"""
        if not records:
            logger.warning("没有数据可保存")
            return False

        try:
            # 确保数据库连接池已创建
            if self.db_pool is None:
                await self.connect_to_db()

            # 获取格式化后的股票代码(添加sh或sz前缀)
            pure_code = stock_code.replace("cn_", "")
            formatted_code = self.format_stock_code(pure_code)
            table_name = f"stock_{formatted_code}_realtime"
            logger.info(f"准备保存数据到表 {table_name}")

            # 获取昨日收盘价
            prev_close = await self.get_stock_prev_close(pure_code)
            if prev_close is None:
                prev_close = 0
                logger.warning(f"使用默认值0作为股票 {stock_code} 的昨日收盘价")

            async with self.db_pool.acquire() as conn:
                async with conn.cursor() as cursor:
                    # 确保表存在
                    create_table_sql = f"""
                    CREATE TABLE IF NOT EXISTS {table_name} (
                        `时间` VARCHAR(255) PRIMARY KEY,
                        `今日开盘价` DECIMAL(10,2),
                        `昨日收盘价` DECIMAL(10,2),
                        `当前价格` DECIMAL(10,2),
                        `今日最低价` DECIMAL(10,2),
                        `竞买价` DECIMAL(10,2),
                        `竞卖价` DECIMAL(10,2),
                        `成交量(手)` INT,
                        `成交额(元)` DECIMAL(18,2),
                        `买一委托量` INT,
                        `买一报价` DECIMAL(10,2),
                        `买二委托量` INT,
                        `买二报价` DECIMAL(10,2),
                        `买三委托量` INT,
                        `买四委托量` INT,
                        `买四报价` DECIMAL(10,2),
                        `买五委托量` INT,
                        `买五报价` DECIMAL(10,2),
                        `卖一委托量` INT,
                        `卖一报价` DECIMAL(10,2),
                        `卖二报价` DECIMAL(10,2),
                        `卖三委托量` INT,
                        `卖三报价` DECIMAL(10,2),
                        `卖四委托量` INT,
                        `卖五委托量` INT,
                        `卖五报价` DECIMAL(10,2),
                        `日期` VARCHAR(255),
                        `其他保留字段` VARCHAR(255)
                    )
                    """
                    await cursor.execute(create_table_sql)

                    # 获取当前日期和时间
                    now = datetime.datetime.now()
                    current_time = now.time()

                    # 根据当前时间判断应该使用哪一天的日期
                    # 如果在9:30之前，使用昨天的日期；9:30之后使用今天的日期
                    if current_time < datetime.time(9, 30):
                        # 使用昨天的日期
                        target_date = (now - datetime.timedelta(days=1)).date()
                        logger.info(f"当前时间在9:30之前，使用昨天的日期: {target_date}")
                    else:
                        # 使用今天的日期
                        target_date = now.date()
                        logger.info(f"当前时间在9:30之后，使用今天的日期: {target_date}")

                    today_date = target_date.strftime("%Y-%m-%d")

                    # 获取今日开盘价（第一条记录的价格）
                    today_open = self.validate_price(records[0][1]) if records else 0

                    # 计算最低价（所有记录中的最小价格）
                    try:
                        # 首先转换并确保所有价格为正数
                        prices = [self.validate_price(item[1]) for item in records]
                        low_price = min(prices) if prices else 0
                    except Exception as e:
                        logger.error(f"计算最低价出错: {str(e)}")
                        low_price = 0

                    # 保存每条记录
                    inserted_count = 0
                    for record in records:
                        try:
                            # 解析记录数据: ['时间', '成交价', '涨跌', '成交量(手)', '成交金额(万元)']
                            time_str, price_str, change_percent, volume_str, amount_str = record

                            # 验证并转换数据
                            price = self.validate_price(price_str)
                            volume = int(float(volume_str)) if volume_str else 0

                            # 成交金额单位是万元，转换为元
                            try:
                                amount = float(amount_str) * 10000 if amount_str else 0
                            except (ValueError, TypeError):
                                amount = 0

                            # 构建完整时间戳
                            timestamp = f"{today_date}-{time_str}"

                            # 检查是否已存在
                            check_sql = f"SELECT * FROM {table_name} WHERE `时间` = %s"
                            await cursor.execute(check_sql, (timestamp,))
                            exists = await cursor.fetchone()

                            if exists:
                                logger.debug(f"记录 {timestamp} 已存在，跳过")
                                continue

                            # 构建数据记录
                            data = {
                                "时间": timestamp,
                                "今日开盘价": today_open,
                                "昨日收盘价": prev_close,
                                "当前价格": price,
                                "今日最低价": low_price,
                                "竞买价": 0,
                                "竞卖价": 0,
                                "成交量(手)": volume,
                                "成交额(元)": amount,
                                "买一委托量": 0,
                                "买一报价": 0,
                                "买二委托量": 0,
                                "买二报价": 0,
                                "买三委托量": 0,
                                "买四委托量": 0,
                                "买四报价": 0,
                                "买五委托量": 0,
                                "买五报价": 0,
                                "卖一委托量": 0,
                                "卖一报价": 0,
                                "卖二报价": 0,
                                "卖三委托量": 0,
                                "卖三报价": 0,
                                "卖四委托量": 0,
                                "卖五委托量": 0,
                                "卖五报价": 0,
                                "日期": today_date,
                                "其他保留字段": change_percent
                            }

                            # 打印部分关键数据用于调试
                            logger.debug(f"准备插入记录 - 时间: {timestamp}, 价格: {price}, 开盘价: {today_open}, 最低价: {low_price}")

                            # 插入数据
                            columns = ', '.join([f'`{col}`' for col in data.keys()])
                            placeholders = ', '.join(['%s'] * len(data))

                            insert_sql = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"
                            await cursor.execute(insert_sql, tuple(data.values()))
                            inserted_count += 1
                        except Exception as e:
                            logger.error(f"处理记录 {record} 时出错: {str(e)}")

                    # 提交事务
                    await conn.commit()
                    logger.info(f"成功将股票 {stock_code} 的分时数据保存到数据库，共 {inserted_count} 条记录")
                    return inserted_count > 0

        except Exception as e:
            logger.error(f"保存分时数据到数据库失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    async def close(self):
        """关闭数据库连接池"""
        if self.db_pool:
            self.db_pool.close()
            await self.db_pool.wait_closed()
            logger.info("数据库连接池已关闭")

    async def process_all_stocks(self):
        """异步处理配置文件中所有股票的分时数据"""
        # 确保配置已加载
        if self.config is None:
            await self.load_config()

        success_count = 0
        failure_count = 0

        # 获取配置文件中的股票列表
        stocks = self.config.get('stocks', [])
        logger.info(f"从配置文件中获取到 {len(stocks)} 只股票")

        # 创建任务列表
        tasks = []
        for stock in stocks:
            stock_code = stock.get('code')
            if stock_code:
                tasks.append(self.process_single_stock(stock_code, stock.get('name', '')))
            else:
                logger.warning("发现缺少股票代码的配置项，跳过")

        # 并发处理所有股票数据
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 统计成功和失败的数量
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"处理股票时出错: {str(result)}")
                failure_count += 1
            elif result:
                success_count += 1
            else:
                failure_count += 1

        logger.info(f"全部处理完成，成功: {success_count}，失败: {failure_count}")
        return success_count > 0

    async def process_single_stock(self, stock_code: str, stock_name: str = ""):
        """异步处理单个股票的分时数据"""
        logger.info(f"开始处理股票 {stock_name}({stock_code}) 的分时数据")

        try:
            # 设置股票代码
            self.set_stock_code(stock_code)

            # 获取分时数据
            records = await self.get_stock_pages_content()

            if records:
                # 保存到数据库
                success = await self.save_to_database(stock_code, records)
                if success:
                    logger.info(f"股票 {stock_name}({stock_code}) 的分时数据处理成功")
                    return True
                else:
                    logger.error(f"股票 {stock_name}({stock_code}) 的分时数据保存失败")
                    return False
            else:
                logger.warning(f"未获取到股票 {stock_name}({stock_code}) 的分时数据")
                return False
        except Exception as e:
            logger.error(f"处理股票 {stock_name}({stock_code}) 的分时数据时出错: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            raise


async def process_stock_minute_data(stock_code: str) -> bool:
    """异步处理指定股票分时数据的主函数 - 用于外部异步调用"""
    fetcher = StockPageFetcher()
    try:
        # 加载配置和连接数据库
        await fetcher.load_config()
        await fetcher.connect_to_db()

        # 设置股票代码
        fetcher.set_stock_code(stock_code)

        # 获取分时数据
        records = await fetcher.get_stock_pages_content()

        if records:
            # 保存到数据库
            success = await fetcher.save_to_database(stock_code, records)
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
        await fetcher.close()


async def process_all_stocks_from_config(config_path: str = '../config/config.json') -> bool:
    """异步处理配置文件中所有股票分时数据的主函数"""
    fetcher = StockPageFetcher(config_path)
    try:
        # 加载配置和连接数据库
        await fetcher.load_config()
        await fetcher.connect_to_db()

        return await fetcher.process_all_stocks()
    except Exception as e:
        logger.error(f"处理所有股票的分时数据失败: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False
    finally:
        await fetcher.close()


# 同步包装函数 - 用于兼容同步调用环境
def sync_process_stock_minute_data(stock_code: str) -> bool:
    """同步处理指定股票分时数据的包装函数"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(process_stock_minute_data(stock_code))
    finally:
        loop.close()


def sync_process_all_stocks_from_config(config_path: str = '../config/config.json') -> bool:
    """同步处理配置文件中所有股票分时数据的包装函数"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(process_all_stocks_from_config(config_path))
    finally:
        loop.close()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # 如果提供了参数，则作为指定的股票代码处理
        stock_code = sys.argv[1]
        logger.info(f"从命令行获取股票代码: {stock_code}")
        success = sync_process_stock_minute_data(stock_code)

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
        success = sync_process_all_stocks_from_config(config_path)

        if success:
            logger.info("所有股票的分时数据处理完成")
        else:
            logger.error("处理所有股票的分时数据时出现错误")

