import json
import logging
import mysql.connector
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import argparse
import concurrent.futures
from functools import partial

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


class StockDecisionAnalyzer:
    """
    股票决策分析器
    分析股票的实时技术指标、日线技术指标和基本面指标，判断短期是否可买入
    """

    def __init__(self, config_path="../config/config.json"):
        """初始化股票决策分析器"""
        self.config_path = config_path
        self.config = self.load_config()

        # 初始化配置监控状态
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config_content = f.read()
            current_config = json.loads(config_content)

        # 提取当前股票列表
        current_main_stocks = current_config.get('stocks', [])
        current_other_stocks = current_config.get('other_stocks', [])
        self.last_config_stock_codes = {stock.get('code') for stock in current_main_stocks + current_other_stocks if stock.get('code')}
        self.last_config = current_config

        self.conn = None
        self.cursor = None
        self.connect_to_db()

        # 技术分析阈值
        self.thresholds = {
            'macd_hist_positive': 0.0,  # MACD柱状线为正的阈值
            'rsi_buy': 50,  # RSI买入信号阈值
            'rsi_overbought': 70,  # RSI超买阈值
            'rsi_oversold': 30,  # RSI超卖阈值
            'ma5_above_ma10': True,  # MA5是否应该在MA10上方
            'buy_threshold': 60  # 综合得分买入阈值
        }

        # 线程池，用于并行处理股票分析
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=10)

    def load_config(self):
        """加载配置文件"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            logger.info("配置文件加载成功")
            return config
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            raise

    def connect_to_db(self):
        """连接到数据库"""
        try:
            self.conn = mysql.connector.connect(
                host=self.config['mysql_config']['host'],
                user=self.config['mysql_config']['user'],
                password=self.config['mysql_config']['password'],
                database=self.config['mysql_config']['database']
            )
            self.cursor = self.conn.cursor(dictionary=True)
            logger.info("成功连接到数据库")
        except Exception as e:
            logger.error(f"连接数据库失败: {e}")
            raise

    def close_db_connection(self):
        """关闭数据库连接"""
        try:
            if self.cursor:
                self.cursor.close()
            if self.conn:
                self.conn.close()
            # 关闭线程池
            if hasattr(self, 'thread_pool'):
                self.thread_pool.shutdown(wait=False)
            logger.info("成功关闭数据库连接和线程池")
        except Exception as e:
            logger.error(f"关闭数据库连接失败: {e}")

    def get_stocks_from_config(self):
        """从配置文件中获取所有股票"""
        stocks = []
        try:
            # 获取其他股票列表
            other_stocks = self.config.get('other_stocks', [])
            stocks.extend(other_stocks)

            logger.info(f"从配置文件中获取到 {len(stocks)} 只股票")
            return stocks
        except Exception as e:
            logger.error(f"获取股票列表失败: {e}")
            return []

    def check_table_exists(self, table_name):
        """检查表是否存在，使用完全独立的连接避免游标冲突"""
        # 创建全新的连接和游标，与主连接完全隔离
        check_conn = None
        check_cursor = None
        try:
            # 创建全新连接
            check_conn = mysql.connector.connect(
                host=self.config['mysql_config']['host'],
                user=self.config['mysql_config']['user'],
                password=self.config['mysql_config']['password'],
                database=self.config['mysql_config']['database']
            )
            # 创建游标
            check_cursor = check_conn.cursor(dictionary=True)

            # 执行检查
            check_query = f"""
            SELECT COUNT(*) as count 
            FROM information_schema.tables 
            WHERE table_schema = DATABASE()  
            AND table_name = '{table_name}'
            """
            check_cursor.execute(check_query)
            result = check_cursor.fetchone()
            exists = result and result['count'] > 0

            return exists
        except Exception as e:
            logger.error(f"检查表 {table_name} 是否存在时出错: {e}")
            return False
        finally:
            # 无论如何，确保资源被正确释放
            try:
                if check_cursor:
                    check_cursor.close()
                if check_conn:
                    check_conn.close()
            except Exception as e:
                logger.error(f"关闭表检查资源时出错: {e}")
                pass

    def get_realtime_indicators(self, stock_name, limit=100):
        """获取实时技术指标，获取多条记录用于分析趋势

        Args:
            stock_name: 股票名称
            limit: 获取的记录数量，默认100条
        """
        table_name = f"realtime_technical_{stock_name}"
        if not self.check_table_exists(table_name):
            logger.warning(f"表 {table_name} 不存在")
            return None

        try:
            query = f"""
            SELECT * FROM `{table_name}`
            ORDER BY 时间 DESC
            LIMIT {limit}
            """
            self.cursor.execute(query)
            results = self.cursor.fetchall()

            if results:
                logger.info(f"成功获取 {stock_name} 的实时技术指标，共 {len(results)} 条记录")
                return results
            else:
                logger.warning(f"未找到 {stock_name} 的实时技术指标")
                return None
        except Exception as e:
            logger.error(f"获取 {stock_name} 的实时技术指标失败: {e}")
            return None

    def get_daily_indicators(self, stock_name, days=60):
        """获取日线技术指标"""
        table_name = f"technical_indicators_{stock_name}"
        if not self.check_table_exists(table_name):
            logger.warning(f"表 {table_name} 不存在")
            return None

        try:
            query = f"""
            SELECT * FROM `{table_name}`
            ORDER BY 日期 DESC
            LIMIT {days}
            """
            self.cursor.execute(query)
            results = self.cursor.fetchall()

            if results:
                logger.info(f"成功获取 {stock_name} 的日线技术指标，共 {len(results)} 条记录")
                return results
            else:
                logger.warning(f"未找到 {stock_name} 的日线技术指标")
                return None
        except Exception as e:
            logger.error(f"获取 {stock_name} 的日线技术指标失败: {e}")
            return None

    def get_fundamental_data(self, stock_name):
        """获取基本面数据"""
        table_name = f"{stock_name}_history"
        if not self.check_table_exists(table_name):
            logger.warning(f"表 {table_name} 不存在")
            return None

        try:
            # 获取最新的基本面数据
            query = f"""
            SELECT * FROM `{table_name}`
            ORDER BY 日期 DESC
            LIMIT 60
            """
            self.cursor.execute(query)
            result = self.cursor.fetchone()

            if result:
                logger.info(f"成功获取 {stock_name} 的基本面数据")
                return result
            else:
                logger.warning(f"未找到 {stock_name} 的基本面数据")
                return None
        except Exception as e:
            logger.error(f"获取 {stock_name} 的基本面数据失败: {e}")
            return None

    def get_realtime_price(self, stock_code, limit=5):
        """从stock_{股票代码}_realtime表中获取当前价格和历史价格

        Args:
            stock_code: 股票代码
            limit: 获取的记录数量，默认5条

        Returns:
            一个字典，包含当前价格和价格列表
        """
        # 处理股票代码格式，添加市场前缀
        if stock_code.startswith('6'):
            formatted_code = f"sh{stock_code}"
        else:
            formatted_code = f"sz{stock_code}"

        table_name = f"stock_{formatted_code}_realtime"

        # 创建全新连接和游标，完全独立于主连接
        price_conn = None
        price_cursor = None

        try:
            # 创建全新连接
            price_conn = mysql.connector.connect(
                host=self.config['mysql_config']['host'],
                user=self.config['mysql_config']['user'],
                password=self.config['mysql_config']['password'],
                database=self.config['mysql_config']['database']
            )
            # 创建游标
            price_cursor = price_conn.cursor(dictionary=True)

            # 先自己检查表是否存在
            check_query = f"""
            SELECT COUNT(*) as count 
            FROM information_schema.tables 
            WHERE table_schema = DATABASE()  
            AND table_name = '{table_name}'
            """
            price_cursor.execute(check_query)
            result = price_cursor.fetchone()
            exists = result and result['count'] > 0

            if not exists:
                logger.warning(f"表 {table_name} 不存在")
                return None

            # 获取价格数据
            query = f"""
            SELECT `当前价格`, `时间` FROM `{table_name}`
            ORDER BY `时间` DESC
            LIMIT {limit}
            """
            price_cursor.execute(query)
            results = price_cursor.fetchall()

            if not results:
                logger.warning(f"未找到 {stock_code} 的价格数据")
                return None

            prices = []
            current_price = None

            for row in results:
                if '当前价格' in row and row['当前价格']:
                    try:
                        price = float(row['当前价格'])
                        prices.append(price)
                        # 保存第一条记录的价格作为当前价格
                        if current_price is None:
                            current_price = price
                    except (ValueError, TypeError):
                        logger.warning(f"无法将价格转换为数值: {row['当前价格']}")

            if current_price is None:
                logger.warning(f"无法获取 {stock_code} 的有效价格")
                return None

            logger.info(f"成功获取 {stock_code} 的价格数据，当前价格: {current_price}, 共 {len(prices)} 条记录")
            return {
                'current_price': current_price,
                'prices': prices
            }
        except Exception as e:
            logger.error(f"获取 {stock_code} 的价格数据失败: {e}")
            return None
        finally:
            # 确保资源被正确释放
            try:
                if price_cursor:
                    price_cursor.close()
                if price_conn:
                    price_conn.close()
            except Exception as e:
                logger.error(f"关闭价格查询资源时出错: {e}")
                pass

    def analyze_realtime_indicators(self, indicators_data):
        """分析实时技术指标

        Args:
            indicators_data: 实时技术指标数据列表
        """
        if not indicators_data or not isinstance(indicators_data, list):
            return None

        # 使用最新的一条数据作为主要分析对象
        indicators = indicators_data[0]
        score = 50  # 基础得分
        reasons = []

        # 分析MACD
        if 'MACD_Hist' in indicators:
            if indicators['MACD_Hist'] > self.thresholds['macd_hist_positive']:
                score += 10
                reasons.append(f"MACD柱状线为正({indicators['MACD_Hist']:.4f})，显示上涨趋势")
            elif indicators['MACD_Hist'] < 0:
                score -= 10
                reasons.append(f"MACD柱状线为负({indicators['MACD_Hist']:.4f})，显示下跌趋势")

            # 分析MACD趋势 (至少需要2条记录)
            if len(indicators_data) >= 3:
                macd_trend = [data['MACD_Hist'] for data in indicators_data[:3] if 'MACD_Hist' in data]
                if len(macd_trend) >= 3 and all(macd_trend[i] > macd_trend[i + 1] for i in range(len(macd_trend) - 1)):
                    score += 8
                    reasons.append(f"MACD柱状线连续上升，动能增强")
                elif len(macd_trend) >= 3 and all(macd_trend[i] < macd_trend[i + 1] for i in range(len(macd_trend) - 1)):
                    score -= 8
                    reasons.append(f"MACD柱状线连续下降，动能减弱")

        # 分析RSI
        if 'RSI' in indicators:
            if indicators['RSI'] > self.thresholds['rsi_buy'] and indicators['RSI'] < self.thresholds['rsi_overbought']:
                score += 10
                reasons.append(f"RSI({indicators['RSI']:.2f})处于买入区间，显示上涨动能")
            elif indicators['RSI'] >= self.thresholds['rsi_overbought']:
                score -= 10
                reasons.append(f"RSI({indicators['RSI']:.2f})处于超买区间，可能回调")
            elif indicators['RSI'] <= self.thresholds['rsi_oversold']:
                score += 5
                reasons.append(f"RSI({indicators['RSI']:.2f})处于超卖区间，可能反弹")

            # 分析RSI趋势
            if len(indicators_data) >= 3:
                rsi_trend = [data['RSI'] for data in indicators_data[:3] if 'RSI' in data]
                if len(rsi_trend) >= 3 and all(rsi_trend[i] > rsi_trend[i + 1] for i in range(len(rsi_trend) - 1)):
                    score += 8
                    reasons.append(f"RSI连续上升，看涨动能增强")
                elif len(rsi_trend) >= 3 and all(rsi_trend[i] < rsi_trend[i + 1] for i in range(len(rsi_trend) - 1)):
                    score -= 8
                    reasons.append(f"RSI连续下降，看涨动能减弱")

        # 分析均线
        if 'MA5' in indicators and 'MA10' in indicators:
            if indicators['MA5'] > indicators['MA10']:
                score += 10
                reasons.append(f"5日均线({indicators['MA5']:.2f})在10日均线({indicators['MA10']:.2f})上方，短期趋势向上")
            else:
                score -= 10
                reasons.append(f"5日均线({indicators['MA5']:.2f})在10日均线({indicators['MA10']:.2f})下方，短期趋势向下")

            # 分析均线趋势
            if len(indicators_data) >= 3:
                ma5_trend = [data['MA5'] for data in indicators_data[:3] if 'MA5' in data]
                if len(ma5_trend) >= 3 and all(ma5_trend[i] > ma5_trend[i + 1] for i in range(len(ma5_trend) - 1)):
                    score += 8
                    reasons.append(f"5日均线连续上升，短期趋势加强")
                elif len(ma5_trend) >= 3 and all(ma5_trend[i] < ma5_trend[i + 1] for i in range(len(ma5_trend) - 1)):
                    score -= 8
                    reasons.append(f"5日均线连续下降，短期趋势减弱")

        # 分析布林带
        price_field = '当前价格'
        if price_field in indicators and 'Upper_Band' in indicators and 'Lower_Band' in indicators:
            if indicators[price_field] > indicators['Upper_Band']:
                score -= 10
                reasons.append(f"价格({indicators[price_field]:.2f})突破上轨({indicators['Upper_Band']:.2f})，可能超买")
            elif indicators[price_field] < indicators['Lower_Band']:
                score += 10
                reasons.append(f"价格({indicators[price_field]:.2f})跌破下轨({indicators['Lower_Band']:.2f})，可能超卖")

        # 限制分数范围
        score = max(0, min(100, score))

        # 判断是否可以买入
        can_buy = score >= self.thresholds['buy_threshold']

        return {
            'score': score,
            'can_buy': can_buy,
            'reasons': reasons
        }

    def analyze_daily_indicators(self, indicators, stock_code=None, stock_name=None):
        """分析日线技术指标

        Args:
            indicators: 日线技术指标数据
            stock_code: 股票代码，用于获取当前价格
            stock_name: 股票名称，用于日志
        """
        if not indicators or len(indicators) < 5:  # 要求至少5天数据
            logger.warning(f"股票 {stock_name}({stock_code}) 的日线技术指标数据不足")
            return None

        # 最新的日线数据
        latest = indicators[0]
        previous = indicators[1] if len(indicators) > 1 else None

        score = 50  # 基础得分
        reasons = []

        # 分析MACD
        if 'MACD_Hist' in latest:
            if latest['MACD_Hist'] > self.thresholds['macd_hist_positive']:
                score += 10
                reasons.append(f"MACD柱状线为正({latest['MACD_Hist']:.4f})，显示上涨趋势")
            elif latest['MACD_Hist'] < 0:
                score -= 10
                reasons.append(f"MACD柱状线为负({latest['MACD_Hist']:.4f})，显示下跌趋势")

            # 分析MACD趋势
            macd_trend = [indicators[i]['MACD_Hist'] for i in range(5) if 'MACD_Hist' in indicators[i]]
            if len(macd_trend) >= 3 and all(macd_trend[i] > macd_trend[i + 1] for i in range(len(macd_trend) - 1)):
                score += 8
                reasons.append("MACD柱状线连续上升，上涨趋势增强")
            elif len(macd_trend) >= 3 and all(macd_trend[i] < macd_trend[i + 1] for i in range(len(macd_trend) - 1)):
                score -= 8
                reasons.append("MACD柱状线连续下降，下跌趋势增强")

        # 分析MACD金叉/死叉
        if previous and 'MACD' in latest and 'Signal' in latest:
            if previous['MACD'] < previous['Signal'] and latest['MACD'] > latest['Signal']:
                score += 15
                reasons.append("MACD金叉形成，买入信号")
            elif previous['MACD'] > previous['Signal'] and latest['MACD'] < latest['Signal']:
                score -= 15
                reasons.append("MACD死叉形成，卖出信号")

        # 分析RSI
        if 'RSI' in latest:
            if latest['RSI'] > self.thresholds['rsi_buy'] and latest['RSI'] < self.thresholds['rsi_overbought']:
                score += 10
                reasons.append(f"RSI({latest['RSI']:.2f})处于买入区间，显示上涨动能")
            elif latest['RSI'] >= self.thresholds['rsi_overbought']:
                score -= 10
                reasons.append(f"RSI({latest['RSI']:.2f})处于超买区间，可能回调")
            elif latest['RSI'] <= self.thresholds['rsi_oversold']:
                score += 5
                reasons.append(f"RSI({latest['RSI']:.2f})处于超卖区间，可能反弹")

            # 分析RSI趋势
            rsi_trend = [indicators[i]['RSI'] for i in range(5) if 'RSI' in indicators[i]]
            if len(rsi_trend) >= 3 and all(rsi_trend[i] > rsi_trend[i + 1] for i in range(len(rsi_trend) - 1)):
                score += 8
                reasons.append("RSI连续上升，买入动能增强")
            elif len(rsi_trend) >= 3 and all(rsi_trend[i] < rsi_trend[i + 1] for i in range(len(rsi_trend) - 1)):
                score -= 8
                reasons.append("RSI连续下降，买入动能减弱")

        # 分析均线
        if 'MA5' in latest and 'MA10' in latest:
            if latest['MA5'] > latest['MA10']:
                score += 10
                reasons.append(f"5日均线({latest['MA5']:.2f})在10日均线({latest['MA10']:.2f})上方，短期趋势向上")
            else:
                score -= 10
                reasons.append(f"5日均线({latest['MA5']:.2f})在10日均线({latest['MA10']:.2f})下方，短期趋势向下")

            # 分析均线趋势
            ma5_trend = [indicators[i]['MA5'] for i in range(5) if 'MA5' in indicators[i]]
            if len(ma5_trend) >= 3 and all(ma5_trend[i] > ma5_trend[i + 1] for i in range(len(ma5_trend) - 1)):
                score += 8
                reasons.append("5日均线连续上升，上涨趋势增强")
            elif len(ma5_trend) >= 3 and all(ma5_trend[i] < ma5_trend[i + 1] for i in range(len(ma5_trend) - 1)):
                score -= 8
                reasons.append("5日均线连续下降，下跌趋势增强")

        # 分析均线金叉/死叉
        if previous and 'MA5' in latest and 'MA10' in latest:
            if previous['MA5'] < previous['MA10'] and latest['MA5'] > latest['MA10']:
                score += 15
                reasons.append("均线金叉形成，买入信号")
            elif previous['MA5'] > previous['MA10'] and latest['MA5'] < latest['MA10']:
                score -= 15
                reasons.append("均线死叉形成，卖出信号")

        # 分析布林带 - 从stock_{股票代码}_realtime表获取当前价格
        price_data = None
        if stock_code:
            price_data = self.get_realtime_price(stock_code)

        if 'Upper_Band' in latest and 'Lower_Band' in latest:
            # 分析布林带宽度
            band_width = latest['Upper_Band'] - latest['Lower_Band']
            band_percent = band_width / ((latest['Upper_Band'] + latest['Lower_Band']) / 2) * 100

            if band_percent > 5:  # 带宽超过均值的5%
                # 带宽较大，波动较大
                score -= 5
                reasons.append(f"布林带宽度较大({band_percent:.2f}%)，市场波动加剧")
            else:
                # 带宽较小，波动较小
                score += 5
                reasons.append(f"布林带宽度较小({band_percent:.2f}%)，市场波动减弱")

            # 如果有当前价格，分析价格与布林带的关系
            if price_data and 'current_price' in price_data:
                current_price = price_data['current_price']
                if current_price > latest['Upper_Band']:
                    score -= 10
                    reasons.append(f"价格({current_price:.2f})突破上轨({latest['Upper_Band']:.2f})，可能超买")
                elif current_price < latest['Lower_Band']:
                    score += 10
                    reasons.append(f"价格({current_price:.2f})跌破下轨({latest['Lower_Band']:.2f})，可能超卖")

                # 分析价格趋势
                if 'prices' in price_data and len(price_data['prices']) >= 3:
                    prices = price_data['prices']
                    if all(prices[i] > prices[i + 1] for i in range(len(prices) - 1)):
                        score += 10
                        reasons.append(f"价格连续上涨，短期看涨")
                    elif all(prices[i] < prices[i + 1] for i in range(len(prices) - 1)):
                        score -= 10
                        reasons.append(f"价格连续下跌，短期看跌")

        # 限制分数范围
        score = max(0, min(100, score))

        # 判断是否可以买入
        can_buy = score >= self.thresholds['buy_threshold']

        return {
            'score': score,
            'can_buy': can_buy,
            'reasons': reasons
        }

    def analyze_fundamental_data(self, data):
        """分析基本面数据"""
        if not data:
            return None

        score = 50  # 基础得分
        reasons = []

        # 分析市盈率
        if '市盈率' in data and data['市盈率'] is not None:
            if data['市盈率'] < 15:
                score += 10
                reasons.append(f"市盈率({data['市盈率']:.2f})较低，可能被低估")
            elif data['市盈率'] > 30:
                score -= 10
                reasons.append(f"市盈率({data['市盈率']:.2f})较高，可能被高估")

        # 分析市净率
        if '市净率' in data and data['市净率'] is not None:
            if data['市净率'] < 1.5:
                score += 10
                reasons.append(f"市净率({data['市净率']:.2f})较低，可能被低估")
            elif data['市净率'] > 3:
                score -= 10
                reasons.append(f"市净率({data['市净率']:.2f})较高，可能被高估")

        # 分析股息率
        if '股息率' in data and data['股息率'] is not None:
            if data['股息率'] > 3:
                score += 10
                reasons.append(f"股息率({data['股息率']:.2f}%)较高，有稳定收益")

        # 分析涨跌幅
        if '涨跌幅(%)' in data and data['涨跌幅(%)'] is not None:
            if data['涨跌幅(%)'] > 5:
                score += 5
                reasons.append(f"涨幅({data['涨跌幅(%)']:.2f}%)较大，短期表现强势")
            elif data['涨跌幅(%)'] < -5:
                score += 5  # 大跌后可能会反弹
                reasons.append(f"跌幅({data['涨跌幅(%)']:.2f}%)较大，可能存在反弹机会")

        # 分析振幅
        if '振幅(%)' in data and data['振幅(%)'] is not None:
            if data['振幅(%)'] > 5:
                score -= 5
                reasons.append(f"振幅({data['振幅(%)']:.2f}%)较大，波动风险高")

        # 分析换手率
        if '换手率(%)' in data and data['换手率(%)'] is not None:
            if data['换手率(%)'] > 10:
                score += 5
                reasons.append(f"换手率({data['换手率(%)']:.2f}%)较高，交易活跃")
            elif data['换手率(%)'] < 1:
                score -= 5
                reasons.append(f"换手率({data['换手率(%)']:.2f}%)较低，交易不活跃")

        # 限制分数范围
        score = max(0, min(100, score))

        # 判断是否可以买入
        can_buy = score >= self.thresholds['buy_threshold']

        return {
            'score': score,
            'can_buy': can_buy,
            'reasons': reasons
        }

    def calculate_comprehensive_score(self, realtime_analysis, daily_analysis, fundamental_analysis):
        """计算综合得分"""
        # 加权计算
        weights = {
            'realtime': 0.4,  # 实时技术指标权重
            'daily': 0.3,  # 日线技术指标权重
            'fundamental': 0.3  # 基本面指标权重
        }

        total_score = 0
        effective_weight = 0

        if realtime_analysis:
            total_score += realtime_analysis['score'] * weights['realtime']
            effective_weight += weights['realtime']

        if daily_analysis:
            total_score += daily_analysis['score'] * weights['daily']
            effective_weight += weights['daily']

        if fundamental_analysis:
            total_score += fundamental_analysis['score'] * weights['fundamental']
            effective_weight += weights['fundamental']

        if effective_weight == 0:
            return {
                'score': 0,
                'can_buy': False,
                'reasons': ["无足够数据进行分析"]
            }

        # 计算加权平均分
        final_score = total_score / effective_weight

        # 判断是否可以买入
        can_buy = final_score >= self.thresholds['buy_threshold']

        # 收集所有原因
        all_reasons = []
        if realtime_analysis and realtime_analysis['reasons']:
            all_reasons.extend([f"实时: {r}" for r in realtime_analysis['reasons'][:2]])

        if daily_analysis and daily_analysis['reasons']:
            all_reasons.extend([f"日线: {r}" for r in daily_analysis['reasons'][:2]])

        if fundamental_analysis and fundamental_analysis['reasons']:
            all_reasons.extend([f"基本面: {r}" for r in fundamental_analysis['reasons'][:2]])

        return {
            'score': final_score,
            'can_buy': can_buy,
            'reasons': all_reasons[:5]  # 最多显示5条原因
        }

    def analyze_stock(self, stock_code, stock_name):
        """分析单只股票，确保所有数据库操作使用独立连接和游标"""
        logger.info(f"开始分析股票: {stock_name}({stock_code})")

        # 获取实时技术指标（获取10条记录）
        # 使用独立连接和游标
        realtime_conn = None
        realtime_cursor = None
        realtime_indicators = None
        try:
            realtime_conn = mysql.connector.connect(
                host=self.config['mysql_config']['host'],
                user=self.config['mysql_config']['user'],
                password=self.config['mysql_config']['password'],
                database=self.config['mysql_config']['database']
            )
            realtime_cursor = realtime_conn.cursor(dictionary=True)

            # 查询实时技术指标
            table_name = f"realtime_technical_{stock_name}"
            # 先检查表是否存在
            check_query = f"""
            SELECT COUNT(*) as count 
            FROM information_schema.tables 
            WHERE table_schema = DATABASE()  
            AND table_name = '{table_name}'
            """
            realtime_cursor.execute(check_query)
            result = realtime_cursor.fetchone()
            # 确保完全读取结果
            realtime_cursor.fetchall()

            exists = result and result['count'] > 0

            if exists:
                query = f"""
                SELECT * FROM `{table_name}`
                ORDER BY 时间 DESC
                LIMIT 100
                """
                realtime_cursor.execute(query)
                realtime_indicators = realtime_cursor.fetchall()
                # 确保完全消费所有结果
                realtime_cursor.fetchall()

                if realtime_indicators:
                    logger.info(f"成功获取 {stock_name} 的实时技术指标，共 {len(realtime_indicators)} 条记录")
                else:
                    logger.warning(f"未找到 {stock_name} 的实时技术指标")
            else:
                logger.warning(f"表 {table_name} 不存在")
        except Exception as e:
            logger.error(f"获取 {stock_name} 的实时技术指标失败: {e}")
        finally:
            # 关闭资源
            if realtime_cursor:
                realtime_cursor.close()
            if realtime_conn:
                realtime_conn.close()

        # 获取日线技术指标（获取30条记录）
        # 使用独立连接和游标
        daily_conn = None
        daily_cursor = None
        daily_indicators = None
        try:
            daily_conn = mysql.connector.connect(
                host=self.config['mysql_config']['host'],
                user=self.config['mysql_config']['user'],
                password=self.config['mysql_config']['password'],
                database=self.config['mysql_config']['database']
            )
            daily_cursor = daily_conn.cursor(dictionary=True)

            # 查询日线技术指标
            table_name = f"technical_indicators_{stock_name}"
            # 先检查表是否存在
            check_query = f"""
            SELECT COUNT(*) as count 
            FROM information_schema.tables 
            WHERE table_schema = DATABASE()  
            AND table_name = '{table_name}'
            """
            daily_cursor.execute(check_query)
            result = daily_cursor.fetchone()
            # 确保完全读取结果
            daily_cursor.fetchall()

            exists = result and result['count'] > 0

            if exists:
                query = f"""
                SELECT * FROM `{table_name}`
                ORDER BY 日期 DESC
                LIMIT 60
                """
                daily_cursor.execute(query)
                daily_indicators = daily_cursor.fetchall()
                # 确保完全消费所有结果
                daily_cursor.fetchall()

                if daily_indicators:
                    logger.info(f"成功获取 {stock_name} 的日线技术指标，共 {len(daily_indicators)} 条记录")
                else:
                    logger.warning(f"未找到 {stock_name} 的日线技术指标")
            else:
                logger.warning(f"表 {table_name} 不存在")
        except Exception as e:
            logger.error(f"获取 {stock_name} 的日线技术指标失败: {e}")
        finally:
            # 关闭资源
            if daily_cursor:
                daily_cursor.close()
            if daily_conn:
                daily_conn.close()

        # 获取基本面数据
        # 使用独立连接和游标
        fund_conn = None
        fund_cursor = None
        fundamental_data = None
        try:
            fund_conn = mysql.connector.connect(
                host=self.config['mysql_config']['host'],
                user=self.config['mysql_config']['user'],
                password=self.config['mysql_config']['password'],
                database=self.config['mysql_config']['database']
            )
            fund_cursor = fund_conn.cursor(dictionary=True)

            # 查询基本面数据
            table_name = f"{stock_name}_history"
            # 先检查表是否存在
            check_query = f"""
            SELECT COUNT(*) as count 
            FROM information_schema.tables 
            WHERE table_schema = DATABASE()  
            AND table_name = '{table_name}'
            """
            fund_cursor.execute(check_query)
            result = fund_cursor.fetchone()
            # 确保完全读取结果
            fund_cursor.fetchall()

            exists = result and result['count'] > 0

            if exists:
                query = f"""
                SELECT * FROM `{table_name}`
                ORDER BY 日期 DESC
                LIMIT 60
                """
                fund_cursor.execute(query)
                fundamental_data = fund_cursor.fetchone()
                # 确保完全消费所有结果
                remaining_results = fund_cursor.fetchall()  # 消费剩余结果

                if fundamental_data:
                    logger.info(f"成功获取 {stock_name} 的基本面数据")
                else:
                    logger.warning(f"未找到 {stock_name} 的基本面数据")
            else:
                logger.warning(f"表 {table_name} 不存在")
        except Exception as e:
            logger.error(f"获取 {stock_name} 的基本面数据失败: {e}")
        finally:
            # 关闭资源
            if fund_cursor:
                fund_cursor.close()
            if fund_conn:
                fund_conn.close()

        # 分析实时技术指标
        realtime_analysis = self.analyze_realtime_indicators(realtime_indicators)

        # 分析日线技术指标
        daily_analysis = self.analyze_daily_indicators(daily_indicators, stock_code, stock_name)

        # 分析基本面数据
        fundamental_analysis = self.analyze_fundamental_data(fundamental_data)

        # 计算综合得分
        comprehensive_result = self.calculate_comprehensive_score(
            realtime_analysis, daily_analysis, fundamental_analysis
        )

        # 构建结果
        result = {
            'stock_code': stock_code,
            'stock_name': stock_name,
            'can_buy': comprehensive_result['can_buy'],
            'score': comprehensive_result['score'],
            'reasons': comprehensive_result['reasons'],
            'realtime_analysis': realtime_analysis,
            'daily_analysis': daily_analysis,
            'fundamental_analysis': fundamental_analysis,
            'analysis_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        # 输出结果摘要
        buy_status = "建议买入 ✓" if comprehensive_result['can_buy'] else "不建议买入 ✗"
        logger.info(f"股票 {stock_name}({stock_code}) 分析结果: {buy_status}, 得分: {comprehensive_result['score']:.2f}")

        return result

    def create_db_connection(self):
        """创建新的数据库连接，用于多线程环境"""
        try:
            conn = mysql.connector.connect(
                host=self.config['mysql_config']['host'],
                user=self.config['mysql_config']['user'],
                password=self.config['mysql_config']['password'],
                database=self.config['mysql_config']['database']
            )
            cursor = conn.cursor(dictionary=True)
            return conn, cursor
        except Exception as e:
            logger.error(f"创建新的数据库连接失败: {e}")
            return None, None

    def analyze_stock_threaded(self, stock):
        """在独立线程中分析单个股票"""
        # 为每个线程创建独立的数据库连接
        conn_tuple = self.create_db_connection()
        try:
            logger.info(f"线程开始分析股票: {stock['name']}({stock['code']})")
            result = self._analyze_single_stock(stock, conn_tuple)

            # 如果是买入信号，则保存
            if result.get('can_buy', False):
                self.save_buy_signal(result)
                logger.info(f"已识别买入信号: {stock['name']}({stock['code']})")

            return result
        except Exception as e:
            logger.error(f"线程分析股票 {stock['name']}({stock['code']}) 时出错: {e}")
            return None
        finally:
            # 关闭数据库连接
            if conn_tuple and conn_tuple[0]:
                conn_tuple[0].close()

    def _analyze_single_stock(self, stock, conn):
        """分析单只股票的内部方法，使用传入的数据库连接

        Args:
            stock: 股票信息，包含code和name
            conn: 数据库连接，为元组(connection, cursor)

        Returns:
            分析结果字典
        """
        if not conn or not isinstance(conn, tuple) or len(conn) != 2:
            logger.error(f"无效的数据库连接对象: {conn}")
            return None

        connection, cursor = conn
        if not connection:
            logger.error("数据库连接无效")
            return None

        code = stock.get('code')
        name = stock.get('name')

        if not code or not name:
            logger.error(f"股票信息不完整: {stock}")
            return None

        # 对单只股票进行分析，与analyze_stock方法类似，但使用传入的数据库连接
        logger.info(f"开始分析股票: {name}({code})")

        # 获取实时技术指标（获取10条记录）
        # 使用传入的连接
        realtime_indicators = None
        try:
            # 查询实时技术指标
            table_name = f"realtime_technical_{name}"
            # 先检查表是否存在
            check_query = f"""
            SELECT COUNT(*) as count 
            FROM information_schema.tables 
            WHERE table_schema = DATABASE()  
            AND table_name = '{table_name}'
            """
            cursor.execute(check_query)
            result = cursor.fetchone()
            # 确保完全读取结果
            cursor.fetchall()

            exists = result and result['count'] > 0

            if exists:
                query = f"""
                SELECT * FROM `{table_name}`
                ORDER BY 时间 DESC
                LIMIT 100
                """
                cursor.execute(query)
                realtime_indicators = cursor.fetchall()
                # 确保完全消费所有结果
                cursor.fetchall()

                if realtime_indicators:
                    logger.info(f"成功获取 {name} 的实时技术指标，共 {len(realtime_indicators)} 条记录")
                else:
                    logger.warning(f"未找到 {name} 的实时技术指标")
            else:
                logger.warning(f"表 {table_name} 不存在")
        except Exception as e:
            logger.error(f"获取 {name} 的实时技术指标失败: {e}")

        # 获取日线技术指标（获取30条记录）
        daily_indicators = None
        try:
            # 查询日线技术指标
            table_name = f"technical_indicators_{name}"
            # 先检查表是否存在
            check_query = f"""
            SELECT COUNT(*) as count 
            FROM information_schema.tables 
            WHERE table_schema = DATABASE()  
            AND table_name = '{table_name}'
            """
            cursor.execute(check_query)
            result = cursor.fetchone()
            # 确保完全读取结果
            cursor.fetchall()

            exists = result and result['count'] > 0

            if exists:
                query = f"""
                SELECT * FROM `{table_name}`
                ORDER BY 日期 DESC
                LIMIT 60
                """
                cursor.execute(query)
                daily_indicators = cursor.fetchall()
                # 确保完全消费所有结果
                cursor.fetchall()

                if daily_indicators:
                    logger.info(f"成功获取 {name} 的日线技术指标，共 {len(daily_indicators)} 条记录")
                else:
                    logger.warning(f"未找到 {name} 的日线技术指标")
            else:
                logger.warning(f"表 {table_name} 不存在")
        except Exception as e:
            logger.error(f"获取 {name} 的日线技术指标失败: {e}")

        # 获取基本面数据
        fundamental_data = None
        try:
            # 查询基本面数据
            table_name = f"{name}_history"
            # 先检查表是否存在
            check_query = f"""
            SELECT COUNT(*) as count 
            FROM information_schema.tables 
            WHERE table_schema = DATABASE()  
            AND table_name = '{table_name}'
            """
            cursor.execute(check_query)
            result = cursor.fetchone()
            # 确保完全读取结果
            cursor.fetchall()

            exists = result and result['count'] > 0

            if exists:
                query = f"""
                SELECT * FROM `{table_name}`
                ORDER BY 日期 DESC
                LIMIT 1
                """
                cursor.execute(query)
                fundamental_data = cursor.fetchone()
                # 确保完全消费所有结果
                cursor.fetchall()

                if fundamental_data:
                    logger.info(f"成功获取 {name} 的基本面数据")
                else:
                    logger.warning(f"未找到 {name} 的基本面数据")
            else:
                logger.warning(f"表 {table_name} 不存在")
        except Exception as e:
            logger.error(f"获取 {name} 的基本面数据失败: {e}")

        # 将元组转换为列表，以确保analyze_realtime_indicators方法能正确处理
        realtime_indicators_list = list(realtime_indicators) if realtime_indicators else None

        # 同样为daily_indicators转换类型
        daily_indicators_list = list(daily_indicators) if daily_indicators else None

        # 基本面数据不需要转换为列表，因为它是单个记录

        # 分析实时技术指标
        realtime_analysis = self.analyze_realtime_indicators(realtime_indicators_list)

        # 分析日线技术指标
        daily_analysis = self.analyze_daily_indicators(daily_indicators_list, code, name)

        # 分析基本面数据
        fundamental_analysis = self.analyze_fundamental_data(fundamental_data)

        # 计算综合得分
        comprehensive_result = self.calculate_comprehensive_score(
            realtime_analysis, daily_analysis, fundamental_analysis
        )

        # 构建结果
        result = {
            'stock_code': code,
            'stock_name': name,
            'can_buy': comprehensive_result['can_buy'],
            'score': comprehensive_result['score'],
            'reasons': comprehensive_result['reasons'],
            'realtime_analysis': realtime_analysis,
            'daily_analysis': daily_analysis,
            'fundamental_analysis': fundamental_analysis,
            'analysis_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        # 输出结果摘要
        buy_status = "建议买入 ✓" if comprehensive_result['can_buy'] else "不建议买入 ✗"
        logger.info(f"股票 {name}({code}) 分析结果: {buy_status}, 得分: {comprehensive_result['score']:.2f}")

        return result

    def realtime_monitor(self, interval=60, non_trading_interval=60, duration=None, config_check_interval=1):
        """实时监控股票数据并判断能否买入

        Args:
            interval: 交易时间内检查间隔，单位为秒，默认60秒
            non_trading_interval: 非交易时间检查间隔，单位为秒，默认60秒
            duration: 监控持续时间，单位为秒，默认为None（持续运行）
            config_check_interval: 配置文件检查间隔，单位为秒，默认1秒
        """
        logger.info(f"开始实时监控，交易时间检查间隔：{interval}秒，非交易时间检查间隔：{non_trading_interval}秒，配置检查间隔：{config_check_interval}秒")

        # 初始获取股票列表
        stocks = self.get_stocks_from_config()
        if not stocks:
            logger.error("没有找到要监控的股票")
            return

        # 修复f-string中的反斜杠问题
        stock_display = ', '.join([f"{s.get('name')}({s.get('code')})" for s in stocks])
        logger.info(f"监控 {len(stocks)} 只股票: {stock_display}")

        # 存储上次买入建议，用于检测变化
        last_recommendations = {}
        start_time = time.time()
        last_config_check_time = time.time()

        try:
            while True:
                current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                logger.info(f"========== 检查时间: {current_time} ==========")

                # 定期检查配置文件是否有变化
                current_time = time.time()
                if current_time - last_config_check_time >= config_check_interval:
                    if self.reload_config_if_changed():
                        # 重新获取股票列表
                        stocks = self.get_stocks_from_config()
                        logger.info(f"更新后监控 {len(stocks)} 只股票")
                    last_config_check_time = current_time

                # 检查是否应该结束监控
                if duration and (time.time() - start_time) > duration:
                    logger.info(f"监控时间到达 {duration} 秒，结束监控")
                    break

                # 检查是否是交易时间
                is_trading = self.is_trading_time()

                if is_trading:
                    # 交易时间内使用多线程并行分析所有股票
                    logger.info(f"当前处于交易时间，开始并行分析 {len(stocks)} 只股票...")
                    start_analysis = time.time()

                    # 使用线程池并行分析所有股票
                    future_to_stock = {self.thread_pool.submit(self.analyze_stock_threaded, stock): stock for stock in stocks}

                    # 收集分析结果
                    buy_recommendations = []
                    results = []

                    for future in concurrent.futures.as_completed(future_to_stock):
                        stock = future_to_stock[future]
                        try:
                            result = future.result()
                            if result:
                                results.append(result)
                                code = stock.get('code', '')
                                name = stock.get('name', '')
                                stock_key = f"{name}({code})"

                                if result['can_buy']:
                                    buy_recommendations.append({
                                        'code': code,
                                        'name': name,
                                        'score': result['score'],
                                        'reasons': result['reasons'][:3]  # 取前3个原因
                                    })

                                    # 检查是否是新的买入建议
                                    if stock_key not in last_recommendations or not last_recommendations[stock_key]:
                                        print(f"\n[!] 新买入信号: {name}({code}) - 评分: {result['score']:.2f}")
                                        print(f"买入理由: {', '.join(result['reasons'][:3])}")

                                    # 保存买入信号到数据库
                                    self.save_buy_signal(result)

                                    # 更新上次建议
                                    last_recommendations[stock_key] = result['can_buy']
                        except Exception as e:
                            logger.error(f"处理 {stock.get('name')}({stock.get('code')}) 的分析结果时出错: {e}")

                    analysis_time = time.time() - start_analysis
                    logger.info(f"完成所有股票分析，耗时: {analysis_time:.2f}秒")

                    # 输出当前可买入的股票
                    if buy_recommendations:
                        print(f"\n当前可买入的股票 ({len(buy_recommendations)}):")
                        for rec in buy_recommendations:
                            print(f"  {rec['name']}({rec['code']}) - 评分: {rec['score']:.2f}")
                            print(f"  理由: {', '.join(rec['reasons'])}")
                    else:
                        print("\n当前没有可买入的股票")

                    # 在交易时间使用交易时间间隔
                    current_interval = interval
                else:
                    # 非交易时间，输出等待消息
                    wait_time = self.get_next_trading_time_wait()
                    hours, remainder = divmod(wait_time, 3600)
                    minutes, seconds = divmod(remainder, 60)
                    wait_str = ""
                    if hours > 0:
                        wait_str += f"{int(hours)}小时"
                    if minutes > 0:
                        wait_str += f"{int(minutes)}分钟"
                    if seconds > 0 or (hours == 0 and minutes == 0):
                        wait_str += f"{int(seconds)}秒"

                    logger.info(f"当前不是交易时间，距离下一个交易时间还有{wait_str}")
                    print(f"当前不是交易时间，将在{wait_str}后进入交易时间")

                    # 在非交易时间使用非交易时间间隔
                    current_interval = non_trading_interval

                # 等待下一次检查
                logger.info(f"等待 {current_interval} 秒后再次检查...")
                time.sleep(current_interval)

        except KeyboardInterrupt:
            logger.info("接收到键盘中断，停止监控")
        finally:
            logger.info("实时监控结束")
            self.thread_pool.shutdown(wait=True)

    def analyze_all_stocks(self):
        """分析所有股票，使用并行处理提高速度"""
        stocks = self.get_stocks_from_config()
        if not stocks:
            logger.error("没有找到要分析的股票")
            return []

        logger.info(f"开始并行分析 {len(stocks)} 只股票...")
        start_time = time.time()

        # 使用线程池并行分析所有股票
        future_to_stock = {self.thread_pool.submit(self.analyze_stock_threaded, stock): stock for stock in stocks}

        # 收集分析结果
        results = []

        for future in concurrent.futures.as_completed(future_to_stock):
            stock = future_to_stock[future]
            try:
                result = future.result()
                if result:
                    results.append(result)

                    # 打印详细结果
                    name = stock.get('name', '')
                    code = stock.get('code', '')
                    print(f"\n股票: {name}({code})")
                    if result['can_buy']:
                        print(f"【建议买入】综合评分: {result['score']:.2f}")
                    else:
                        print(f"【不建议买入】综合评分: {result['score']:.2f}")

                    print("分析原因:")
                    for reason in result['reasons']:
                        print(f"  - {reason}")

                    # 打印各项分析结果
                    if result['realtime_analysis']:
                        real_status = "支持买入 ✓" if result['realtime_analysis']['can_buy'] else "不建议买入 ✗"
                        print(f"实时技术分析 ({result['realtime_analysis']['score']:.2f}分): {real_status}")

                    if result['daily_analysis']:
                        daily_status = "支持买入 ✓" if result['daily_analysis']['can_buy'] else "不建议买入 ✗"
                        print(f"日线技术分析 ({result['daily_analysis']['score']:.2f}分): {daily_status}")

                    if result['fundamental_analysis']:
                        fund_status = "支持买入 ✓" if result['fundamental_analysis']['can_buy'] else "不建议买入 ✗"
                        print(f"基本面分析 ({result['fundamental_analysis']['score']:.2f}分): {fund_status}")

                    # 保存买入信号到数据库 - 使用独立游标
                    if result.get('can_buy', False):
                        save_cursor = None
                        try:
                            save_cursor = self.conn.cursor(dictionary=True)
                            self._save_buy_signal_with_cursor(result, save_cursor)
                        except Exception as e:
                            logger.error(f"保存买入信号时出错: {e}")
                        finally:
                            if save_cursor:
                                try:
                                    save_cursor.close()
                                except:
                                    pass
            except Exception as e:
                logger.error(f"处理 {stock.get('name')}({stock.get('code')}) 的分析结果时出错: {e}")
                import traceback
                logger.error(traceback.format_exc())

        analysis_time = time.time() - start_time
        logger.info(f"完成所有股票分析，耗时: {analysis_time:.2f}秒")

        # 整理可买入股票列表
        buyable_stocks = [r for r in results if r.get('can_buy', False)]
        if buyable_stocks:
            print("\n\n=== 以下股票短期建议买入 ===")
            for stock in buyable_stocks:
                print(f"{stock['stock_name']}({stock['stock_code']}): {stock['score']:.2f}分")
                print(f"  买入理由: {', '.join(stock['reasons'][:3])}")
        else:
            print("\n\n当前没有符合买入条件的股票")

        return results

    def _save_buy_signal_with_cursor(self, stock_result, cursor):
        """使用独立游标保存买入信号

        Args:
            stock_result: 股票分析结果
            cursor: 数据库游标
        """
        if not stock_result or not stock_result.get('can_buy', False):
            return False  # 不保存非买入信号

        # 确保表存在
        if not hasattr(self, 'signal_table_created'):
            self.signal_table_created = self.create_trading_signals_table()

        if not self.signal_table_created:
            logger.error("交易信号表未创建，无法保存买入信号")
            return False

        try:
            # 提取分析时间
            analysis_time = stock_result.get('analysis_time') or datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            # 获取当前价格
            current_price = None
            if stock_result.get('stock_code'):
                price_data = self.get_realtime_price(stock_result['stock_code'])
                if price_data and 'current_price' in price_data:
                    current_price = price_data['current_price']

            # 提取各项分析分数
            realtime_score = None
            if stock_result.get('realtime_analysis'):
                realtime_score = stock_result['realtime_analysis'].get('score')

            daily_score = None
            if stock_result.get('daily_analysis'):
                daily_score = stock_result['daily_analysis'].get('score')

            fundamental_score = None
            if stock_result.get('fundamental_analysis'):
                fundamental_score = stock_result['fundamental_analysis'].get('score')

            # 准备理由字符串
            reasons_str = '; '.join(stock_result.get('reasons', []))

            # 检查是否已存在相同股票和时间的记录
            check_query = """
            SELECT id FROM trading_signals 
            WHERE stock_code = %s AND DATE(analysis_time) = DATE(%s)
            """
            cursor.execute(check_query, (stock_result['stock_code'], analysis_time))
            existing_record = cursor.fetchone()

            if existing_record:
                # 更新现有记录
                update_query = """
                UPDATE trading_signals SET 
                score = %s,
                reasons = %s,
                realtime_score = %s,
                daily_score = %s,
                fundamental_score = %s,
                current_price = %s,
                analysis_time = %s
                WHERE id = %s
                """
                cursor.execute(update_query, (
                    stock_result['score'],
                    reasons_str,
                    realtime_score,
                    daily_score,
                    fundamental_score,
                    current_price,
                    analysis_time,
                    existing_record['id']
                ))
                logger.info(f"更新股票 {stock_result['stock_name']}({stock_result['stock_code']}) 的买入信号")
            else:
                # 插入新记录
                insert_query = """
                INSERT INTO trading_signals (
                    stock_code, 
                    stock_name, 
                    analysis_time, 
                    score, 
                    reasons, 
                    realtime_score,
                    daily_score,
                    fundamental_score,
                    current_price
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """
                cursor.execute(insert_query, (
                    stock_result['stock_code'],
                    stock_result['stock_name'],
                    analysis_time,
                    stock_result['score'],
                    reasons_str,
                    realtime_score,
                    daily_score,
                    fundamental_score,
                    current_price
                ))
                logger.info(f"添加股票 {stock_result['stock_name']}({stock_result['stock_code']}) 的买入信号")

            self.conn.commit()
            return True
        except Exception as e:
            logger.error(f"保存买入信号失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def is_trading_time(self):
        """检查当前是否是交易时间（工作日9:30-11:30, 13:00-15:00）"""
        now = datetime.now()

        # 检查是否是周末
        if now.weekday() >= 5:  # 5=周六, 6=周日
            return False

        # 检查时间段
        current_time = now.time()
        morning_start = datetime.strptime("9:30", "%H:%M").time()
        morning_end = datetime.strptime("11:30", "%H:%M").time()
        afternoon_start = datetime.strptime("13:00", "%H:%M").time()
        afternoon_end = datetime.strptime("15:00", "%H:%M").time()

        is_trading = (morning_start <= current_time <= morning_end) or (afternoon_start <= current_time <= afternoon_end)
        return is_trading

    def get_next_trading_time_wait(self):
        """计算距离下一个交易时间段的等待秒数"""
        now = datetime.now()
        current_time = now.time()

        # 定义交易时间段
        morning_start = datetime.strptime("9:30", "%H:%M").time()
        morning_end = datetime.strptime("11:30", "%H:%M").time()
        afternoon_start = datetime.strptime("13:00", "%H:%M").time()
        afternoon_end = datetime.strptime("15:00", "%H:%M").time()

        # 计算今天的各个时间点
        today = now.date()
        today_morning_start = datetime.combine(today, morning_start)
        today_morning_end = datetime.combine(today, morning_end)
        today_afternoon_start = datetime.combine(today, afternoon_start)
        today_afternoon_end = datetime.combine(today, afternoon_end)

        # 计算等待时间
        if current_time < morning_start:
            # 等待今天早上开盘
            wait_seconds = (today_morning_start - now).total_seconds()
        elif morning_end < current_time < afternoon_start:
            # 等待今天下午开盘
            wait_seconds = (today_afternoon_start - now).total_seconds()
        else:
            # 已经过了今天的交易时间，等待明天早上
            tomorrow = today + timedelta(days=1)
            # 跳过周末
            if tomorrow.weekday() >= 5:  # 5=周六, 6=周日
                # 计算下一个工作日
                days_to_add = 8 - tomorrow.weekday() if tomorrow.weekday() == 6 else 7 - tomorrow.weekday()
                tomorrow = today + timedelta(days=days_to_add)

            tomorrow_morning_start = datetime.combine(tomorrow, morning_start)
            wait_seconds = (tomorrow_morning_start - now).total_seconds()

        return int(wait_seconds)

    def reload_config_if_changed(self):
        """检查配置文件内容是否有变化，如果有则重新加载"""
        try:
            # 读取当前配置文件内容
            with open(self.config_path, 'r', encoding='utf-8') as f:
                current_config_content = f.read()
                current_config = json.loads(current_config_content)

            # 提取股票列表
            current_main_stocks = current_config.get('stocks', [])
            current_other_stocks = current_config.get('other_stocks', [])

            # 创建股票代码集合用于快速比较
            current_stock_codes = {stock.get('code') for stock in current_main_stocks + current_other_stocks if stock.get('code')}

            # 首次运行或配置内容变化检测
            if not hasattr(self, 'last_config_stock_codes'):
                # 首次运行，初始化存储
                self.last_config_stock_codes = current_stock_codes
                self.last_config = current_config
                logger.info(f"初始化配置文件监控，当前监控 {len(current_stock_codes)} 只股票")
                return False

            # 检查股票列表是否有变化
            if self.last_config_stock_codes != current_stock_codes:
                logger.info("检测到配置文件股票列表变更，重新加载配置...")

                # 找出新增的股票
                added_stock_codes = current_stock_codes - self.last_config_stock_codes

                # 找出移除的股票
                removed_stock_codes = self.last_config_stock_codes - current_stock_codes

                # 提取完整的新增股票信息
                added_stocks = []
                for stock in current_main_stocks + current_other_stocks:
                    if stock.get('code') in added_stock_codes:
                        added_stocks.append(stock)

                # 记录变更
                if added_stocks:
                    added_stock_names = [f"{stock.get('name')}({stock.get('code')})" for stock in added_stocks]
                    logger.info(f"新增股票: {', '.join(added_stock_names)}")
                    print(f"\n[配置更新] 新增监控 {len(added_stocks)} 只股票: {', '.join(added_stock_names)}")

                if removed_stock_codes:
                    removed_stock_codes_str = ', '.join(removed_stock_codes)
                    logger.info(f"移除股票代码: {removed_stock_codes_str}")
                    print(f"\n[配置更新] 停止监控 {len(removed_stock_codes)} 只股票: {removed_stock_codes_str}")

                # 更新配置
                self.config = current_config
                self.last_config_stock_codes = current_stock_codes
                self.last_config = current_config

                # 检查其他配置项是否有变化（可选）
                if self._check_other_config_changes():
                    logger.info("检测到其他配置项变更，已更新")

                # 刷新表存在缓存
                if hasattr(self, 'table_exists_cache'):
                    self.table_exists_cache = {}

                return True

            # 检查其他重要配置项是否变更（例如数据库配置、阈值设置等）
            if self._check_other_config_changes(current_config):
                logger.info("检测到其他配置项变更，更新配置...")
                self.config = current_config
                self.last_config = current_config
                return True

            return False
        except Exception as e:
            logger.error(f"检查配置文件变更时出错: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def _check_other_config_changes(self, new_config=None):
        """检查除股票列表外的其他配置项是否有变化"""
        if new_config is None:
            new_config = self.config

        if not hasattr(self, 'last_config'):
            return False

        # 检查技术分析阈值
        new_thresholds = new_config.get('thresholds', {})
        old_thresholds = self.last_config.get('thresholds', {})
        if new_thresholds != old_thresholds:
            # 更新阈值设置
            self.thresholds.update(new_thresholds)
            logger.info("技术分析阈值设置已更新")
            return True

        # 检查MySQL配置
        new_mysql = new_config.get('mysql_config', {})
        old_mysql = self.last_config.get('mysql_config', {})
        if new_mysql != old_mysql:
            logger.warning("MySQL配置已变更，需要重启程序才能生效")
            return True

        # 检查设置
        new_settings = new_config.get('settings', {})
        old_settings = self.last_config.get('settings', {})
        if new_settings != old_settings:
            logger.info("全局设置已更新")
            return True

        return False

    def create_trading_signals_table(self):
        """创建交易信号表，如果不存在的话"""
        # 使用独立连接和游标
        table_conn = None
        table_cursor = None
        try:
            # 创建新连接和游标
            table_conn = mysql.connector.connect(
                host=self.config['mysql_config']['host'],
                user=self.config['mysql_config']['user'],
                password=self.config['mysql_config']['password'],
                database=self.config['mysql_config']['database']
            )
            table_cursor = table_conn.cursor(dictionary=True)

            create_table_query = """
            CREATE TABLE IF NOT EXISTS trading_signals (
                id INT AUTO_INCREMENT PRIMARY KEY COMMENT '主键ID',
                stock_code VARCHAR(10) NOT NULL COMMENT '股票代码',
                stock_name VARCHAR(50) NOT NULL COMMENT '股票名称',
                analysis_time DATETIME NOT NULL COMMENT '分析时间',
                score FLOAT NOT NULL COMMENT '综合评分',
                reasons TEXT COMMENT '买入理由',
                is_bought BOOLEAN DEFAULT FALSE COMMENT '是否已买入',
                is_sold BOOLEAN DEFAULT FALSE COMMENT '是否已卖出',

                /* 交易执行信息 */
                buy_price FLOAT DEFAULT NULL COMMENT '买入价格',
                sell_price FLOAT DEFAULT NULL COMMENT '卖出价格',
                buy_quantity INT DEFAULT NULL COMMENT '买入数量(股)',
                buy_time DATETIME DEFAULT NULL COMMENT '买入时间',
                sell_time DATETIME DEFAULT NULL COMMENT '卖出时间',

                /* 账户和持仓信息 */
                account_id VARCHAR(50) DEFAULT NULL COMMENT '账户ID',
                account_name VARCHAR(100) DEFAULT NULL COMMENT '账户名称',
                available_cash DECIMAL(20,2) DEFAULT NULL COMMENT '可用资金',
                portfolio_ratio FLOAT DEFAULT NULL COMMENT '持仓比例',
                position_cost DECIMAL(20,2) DEFAULT NULL COMMENT '持仓成本',

                /* 交易策略信息 */
                target_position FLOAT DEFAULT NULL COMMENT '目标仓位',
                stop_loss_price DECIMAL(10,2) DEFAULT NULL COMMENT '止损价',
                take_profit_price DECIMAL(10,2) DEFAULT NULL COMMENT '止盈价',

                /* 技术分析分数 */
                realtime_score FLOAT DEFAULT NULL COMMENT '实时分析得分',
                daily_score FLOAT DEFAULT NULL COMMENT '日线分析得分',
                fundamental_score FLOAT DEFAULT NULL COMMENT '基本面分析得分',
                current_price FLOAT DEFAULT NULL COMMENT '当前价格',

                /* 凯利公式和仓位管理字段 */
                win_rate FLOAT DEFAULT NULL COMMENT '历史胜率',
                win_loss_ratio FLOAT DEFAULT NULL COMMENT '盈亏比',
                kelly_value FLOAT DEFAULT NULL COMMENT '凯利值',
                position_score_factor FLOAT DEFAULT NULL COMMENT '得分调整因子',
                final_position FLOAT DEFAULT NULL COMMENT '最终仓位比例',

                /* 交易状态信息 */
                transaction_id VARCHAR(50) DEFAULT NULL COMMENT '交易ID',
                order_type VARCHAR(20) DEFAULT NULL COMMENT '订单类型(市价/限价等)',
                trade_status VARCHAR(20) DEFAULT 'PENDING' COMMENT '交易状态(等待/完成/失败)',
                error_message TEXT COMMENT '错误信息',
                notes TEXT COMMENT '备注信息',
                last_update_time DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '最后更新时间',

                UNIQUE KEY (stock_code, analysis_time)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='股票交易信号表';
            """
            table_cursor.execute(create_table_query)
            table_conn.commit()
            logger.info("交易信号表创建或已存在")
            return True
        except Exception as e:
            logger.error(f"创建交易信号表失败: {e}")
            return False
        finally:
            # 关闭资源
            if table_cursor:
                table_cursor.close()
            if table_conn:
                table_conn.close()

    def save_buy_signal(self, stock_result):
        """保存买入信号到数据库"""
        # 检查是否为买入信号
        if not stock_result.get('can_buy', False):
            logger.debug(f"股票 {stock_result.get('stock_name', 'Unknown')}({stock_result.get('stock_code', 'Unknown')}) 不是买入信号，无需保存")
            return False

        # 确保trading_signals表已创建
        self.create_trading_signals_table()

        # 使用独立连接
        signal_conn = None
        signal_cursor = None
        try:
            # 创建新连接和游标
            signal_conn = mysql.connector.connect(
                host=self.config['mysql_config']['host'],
                user=self.config['mysql_config']['user'],
                password=self.config['mysql_config']['password'],
                database=self.config['mysql_config']['database']
            )
            signal_cursor = signal_conn.cursor(dictionary=True)

            stock_code = stock_result.get('stock_code')
            stock_name = stock_result.get('stock_name')
            score = stock_result.get('score', 0)
            analysis_time = stock_result.get('analysis_time', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            reasons = json.dumps(stock_result.get('reasons', []), ensure_ascii=False)

            # 获取技术分析分数
            realtime_score = None
            if stock_result.get('realtime_analysis'):
                realtime_score = stock_result['realtime_analysis'].get('score')

            daily_score = None
            if stock_result.get('daily_analysis'):
                daily_score = stock_result['daily_analysis'].get('score')

            fundamental_score = None
            if stock_result.get('fundamental_analysis'):
                fundamental_score = stock_result['fundamental_analysis'].get('score')

            current_price = None
            if stock_result.get('stock_code'):
                price_data = self.get_realtime_price(stock_result['stock_code'])
                if price_data and 'current_price' in price_data:
                    current_price = price_data['current_price']

            # 查询是否已有相同的记录（同一股票同一分析时间）
            check_query = "SELECT id FROM trading_signals WHERE stock_code = %s AND analysis_time = %s"
            signal_cursor.execute(check_query, (stock_code, analysis_time))
            # 消费所有结果
            result = signal_cursor.fetchall()

            if result:
                # 更新现有记录
                update_query = """
                UPDATE trading_signals 
                SET 
                    score = %s, 
                    reasons = %s,
                    realtime_score = %s,
                    daily_score = %s,
                    fundamental_score = %s,
                    current_price = %s
                WHERE stock_code = %s AND analysis_time = %s
                """
                signal_cursor.execute(update_query, (
                    score, reasons,
                    realtime_score, daily_score, fundamental_score, current_price,
                    stock_code, analysis_time
                ))
            else:
                # 检查当前未买入的不同股票数量是否已达到上限(5只)
                distinct_check_query = """
                SELECT COUNT(DISTINCT stock_code) as stock_count 
                FROM trading_signals 
                WHERE is_bought = FALSE
                """
                signal_cursor.execute(distinct_check_query)
                count_result = signal_cursor.fetchone()

                # 检查这只股票是否已经在未买入列表中
                stock_exists_query = """
                SELECT id FROM trading_signals 
                WHERE stock_code = %s AND is_bought = FALSE
                """
                signal_cursor.execute(stock_exists_query, (stock_code,))
                exists_result = signal_cursor.fetchall()

                # 从kelly_config.json配置中读取最大持有股票数
                try:
                    # 尝试读取kelly_config.json
                    kelly_config_path = "auto_trader/kelly_config.json"
                    with open(kelly_config_path, 'r', encoding='utf-8') as f:
                        kelly_config = json.load(f)
                    max_stocks = kelly_config.get('trade_settings', {}).get('max_stocks', 5)
                except Exception as e:
                    logger.warning(f"无法从kelly_config.json读取配置: {e}, 使用默认值5")
                    max_stocks = 5  # 默认最大持有5只股票

                # 如果已经达到最大持有股票数且当前股票不在列表中，则不插入
                if count_result and count_result['stock_count'] >= max_stocks and not exists_result:
                    logger.info(f"已达到最大持有股票数量({max_stocks})，不保存新的买入信号: {stock_name}({stock_code})")
                    # 找到评分最低的股票并替换
                    replace_lowest_query = """
                    SELECT id, stock_code, stock_name, score 
                    FROM trading_signals 
                    WHERE is_bought = FALSE 
                    ORDER BY score ASC LIMIT 1
                    """
                    signal_cursor.execute(replace_lowest_query)
                    lowest_signal = signal_cursor.fetchone()

                    if lowest_signal and lowest_signal['score'] < score:
                        # 删除评分最低的股票信号
                        delete_query = "DELETE FROM trading_signals WHERE id = %s"
                        signal_cursor.execute(delete_query, (lowest_signal['id'],))

                        # 插入新的信号
                        insert_query = """
                        INSERT INTO trading_signals (
                            stock_code, stock_name, analysis_time, score, reasons,
                            realtime_score, daily_score, fundamental_score, current_price
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """
                        signal_cursor.execute(insert_query, (
                            stock_code, stock_name, analysis_time, score, reasons,
                            realtime_score, daily_score, fundamental_score, current_price
                        ))

                        logger.info(f"已替换评分较低的股票 {lowest_signal['stock_name']}({lowest_signal['stock_code']})，"
                                    f"评分: {lowest_signal['score']} -> {stock_name}({stock_code})，评分: {score}")
                    return True
                else:
                    # 正常插入新记录
                    insert_query = """
                    INSERT INTO trading_signals (
                        stock_code, stock_name, analysis_time, score, reasons,
                        realtime_score, daily_score, fundamental_score, current_price
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """
                    signal_cursor.execute(insert_query, (
                        stock_code, stock_name, analysis_time, score, reasons,
                        realtime_score, daily_score, fundamental_score, current_price
                    ))

            # 提交事务
            signal_conn.commit()
            logger.info(f"已保存买入信号: {stock_name}({stock_code}) - 得分: {score}")
            return True

        except Exception as e:
            logger.error(f"保存买入信号失败: {e}")
            return False
        finally:
            # 关闭资源
            if signal_cursor:
                signal_cursor.close()
            if signal_conn:
                signal_conn.close()

    def get_pending_buy_signals(self):
        """获取所有未买入的买入信号"""
        try:
            query = """
            SELECT * FROM trading_signals 
            WHERE is_bought = FALSE 
            ORDER BY analysis_time DESC
            """
            self.cursor.execute(query)
            results = self.cursor.fetchall()

            if results:
                logger.info(f"找到 {len(results)} 条未买入的交易信号")
            else:
                logger.info("没有未买入的交易信号")

            return results
        except Exception as e:
            logger.error(f"获取未买入信号失败: {e}")
            return []

    def update_signal_status(self, signal_id, **kwargs):
        """更新交易信号状态

        Args:
            signal_id: 信号ID
            **kwargs: 需要更新的字段和值
        """
        try:
            # 有效字段列表
            valid_fields = [
                'is_bought', 'is_sold', 'buy_price', 'sell_price', 'buy_quantity',
                'buy_time', 'sell_time', 'account_id', 'account_name', 'available_cash',
                'portfolio_ratio', 'position_cost', 'target_position', 'stop_loss_price',
                'take_profit_price', 'transaction_id', 'order_type', 'trade_status',
                'error_message', 'notes'
            ]

            set_clauses = []
            params = []

            for field, value in kwargs.items():
                if field in valid_fields:
                    set_clauses.append(f"{field} = %s")
                    params.append(value)

            if not set_clauses:
                logger.warning("没有提供有效的更新字段")
                return False

            query = f"""
            UPDATE trading_signals
            SET {', '.join(set_clauses)}
            WHERE id = %s
            """

            params.append(signal_id)
            self.cursor.execute(query, params)
            self.conn.commit()

            logger.info(f"更新信号 ID {signal_id} 的状态")
            return True
        except Exception as e:
            logger.error(f"更新信号状态失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False


def main():
    """主函数，一次性分析所有股票"""
    print("开始股票决策分析...")
    analyzer = StockDecisionAnalyzer()

    try:
        # 分析所有股票
        analyzer.analyze_all_stocks()
    except Exception as e:
        logger.error(f"执行过程中出错: {e}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        # 关闭数据库连接
        analyzer.close_db_connection()
        print("\n分析完成!")


def start_realtime_monitor(interval=5, non_trading_interval=60, config_check_interval=1):
    """启动实时监控，在交易时间和非交易时间使用不同的检查间隔

    Args:
        interval: 交易时间内检查间隔，单位为秒，默认5秒
        non_trading_interval: 非交易时间检查间隔，单位为秒，默认60秒
        config_check_interval: 配置文件检查间隔，单位为秒，默认1秒
    """
    print(f"开始股票决策实时监控...")
    print(f"交易时间内每 {interval} 秒检查一次")
    print(f"非交易时间每 {non_trading_interval} 秒检查一次")
    print(f"配置文件每 {config_check_interval} 秒检查一次")
    print(f"使用多线程并行分析多支股票，将大幅提高处理速度")

    analyzer = StockDecisionAnalyzer()

    try:
        analyzer.realtime_monitor(interval=interval, non_trading_interval=non_trading_interval, config_check_interval=config_check_interval)
    except Exception as e:
        logger.error(f"监控过程中出错: {e}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        # 关闭数据库连接
        analyzer.close_db_connection()
        print("\n监控结束!")


if __name__ == "__main__":
    # 交易时间内每5秒检查一次，非交易时间每60秒检查一次，配置文件每1秒检查一次
    start_realtime_monitor(interval=5, non_trading_interval=60, config_check_interval=1)

    # 取消下面一行的注释以启用一次性分析模式（分析所有股票并显示结果）
    # main()