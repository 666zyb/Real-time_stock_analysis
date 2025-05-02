import matplotlib
import matplotlib.pyplot as plt
import mysql.connector
import pandas as pd
import mplfinance as mpf
import ta
import json
from datetime import datetime

# 设置支持中文的字体
matplotlib.rcParams['font.family'] = 'SimHei'  # 使用黑体字体，根据系统实际情况调整
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

matplotlib.use('TkAgg')  # 使用 TkAgg 后端


class StockChartAnalyzer:
    def __init__(self, config_path='../config/config.json'):
        """
        初始化 StockChartAnalyzer 类
        :param config_path: 配置文件的路径
        """
        self.config = self.load_config(config_path)
        self.mydb = self.connect_to_mysql()
        self.mycursor = self.mydb.cursor()

    def load_config(self, file_path):
        """
        加载配置文件
        :param file_path: 配置文件的路径
        :return: 配置文件的内容，如果文件不存在或格式错误则返回 None
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return json.load(file)
        except FileNotFoundError:
            print(f"错误: 未找到配置文件 {file_path}")
        except json.JSONDecodeError:
            print(f"错误: 配置文件 {file_path} 不是有效的 JSON 格式")
        return None

    def connect_to_mysql(self):
        """
        连接到 MySQL 数据库
        :return: 数据库连接对象，如果连接失败则返回 None
        """
        try:
            mysql_config = self.config['mysql_config']
            mydb = mysql.connector.connect(
                host=mysql_config['host'],
                port=mysql_config.get('port', 3306),
                user=mysql_config['user'],
                password=mysql_config['password'],
                database=mysql_config['database']
            )
            print("成功连接到 MySQL")
            return mydb
        except mysql.connector.Error as e:
            print(f"无法连接到 MySQL: {e}")
            return None

    def get_stock_data(self, stock_name):
        """
        获取指定股票的历史数据和实时数据，并合并处理
        :param stock_name: 股票名称
        :return: 合并后的股票数据 DataFrame，如果未找到股票代码则返回 None
        """
        # 根据股票名称获取股票代码，并格式化
        formatted_code = next((f"sh{stock['code']}" if stock['code'].startswith('6') else f"sz{stock['code']}"
                               for stock in self.config['stocks'] if stock['name'] == stock_name), None)
        if not formatted_code:
            print(f"未找到 {stock_name} 的代码")
            return None

        table_name_history = f"{stock_name}_history"
        table_name_realtime = f"stock_{formatted_code}_realtime"

        # 读取历史数据
        query_history = f"SELECT `日期`, `开盘价`, `收盘价`, `最高价`, `最低价`, `成交量(手)` FROM {table_name_history}"
        df_history = pd.read_sql(query_history, self.mydb)
        df_history['日期'] = pd.to_datetime(df_history['日期'])
        df_history.set_index('日期', inplace=True)
        df_history.rename(columns={'开盘价': 'Open', '收盘价': 'Close', '最高价': 'High', '最低价': 'Low', '成交量(手)': 'Volume'}, inplace=True)

        # 读取实时数据
        query_realtime = f"SELECT `时间`, `当前价格`, `成交量(手)` FROM {table_name_realtime}"
        df_realtime = pd.read_sql(query_realtime, self.mydb)
        df_realtime['时间'] = pd.to_datetime(df_realtime['时间'])
        df_realtime.set_index('时间', inplace=True)
        df_realtime.rename(columns={'当前价格': 'Close', '成交量(手)': 'Volume'}, inplace=True)

        # 合并历史数据和实时数据
        df_combined = pd.concat([df_history, df_realtime])

        # 转换数据类型
        df_combined['Close'] = pd.to_numeric(df_combined['Close'], errors='coerce')
        df_combined['Volume'] = pd.to_numeric(df_combined['Volume'], errors='coerce')
        df_combined['Open'] = pd.to_numeric(df_combined['Open'], errors='coerce')
        df_combined['High'] = pd.to_numeric(df_combined['High'], errors='coerce')
        df_combined['Low'] = pd.to_numeric(df_combined['Low'], errors='coerce')

        # 处理缺失值
        df_combined = df_combined.dropna(subset=['Close', 'Volume', 'Open', 'High', 'Low'])

        return df_combined

    def plot_line_chart(self, df, stock_name):
        """
        绘制股票收盘价的线图
        :param df: 股票数据 DataFrame
        :param stock_name: 股票名称
        """
        recent_df = df.tail(500).copy()  # 选取最近 60 条数据并创建副本
        plt.figure(figsize=(12, 6))
        recent_df['Close'].plot(title=f'{stock_name} 收盘价线图')
        plt.xlabel('日期')
        plt.ylabel('收盘价')

    def plot_bar_chart(self, df, stock_name):
        """
        绘制股票成交量的柱状图
        :param df: 股票数据 DataFrame
        :param stock_name: 股票名称
        """
        recent_df = df.tail(60).copy()  # 选取最近 60 条数据并创建副本
        plt.figure(figsize=(12, 6))
        # 对日期索引进行格式化
        recent_df.index = recent_df.index.strftime('%Y-%m-%d')
        recent_df['Volume'].plot(kind='bar', title=f'{stock_name} 成交量柱状图')
        plt.xlabel('日期')
        plt.ylabel('成交量')

    def plot_candlestick_chart(self, df, stock_name):
        """
        绘制股票的 K 线图
        :param df: 股票数据 DataFrame
        :param stock_name: 股票名称
        """
        recent_df = df.tail(200).copy()  # 选取最近 60 条数据并创建副本
        kwargs = dict(type='candle', volume=True, mav=(5, 10, 20), figratio=(12, 8), warn_too_much_data=len(recent_df) + 1)
        # 设置 mplfinance 的字体
        custom_rc = {
            'font.family': 'SimHei',
            'axes.unicode_minus': False
        }
        s = mpf.make_mpf_style(rc=custom_rc)
        mpf.plot(recent_df, **kwargs, title=f'{stock_name} 股票 K 线图', ylabel='价格', ylabel_lower='成交量', style=s)

    def plot_macd(self, df, stock_name):
        """
        绘制股票的 MACD 指标图
        :param df: 股票数据 DataFrame
        :param stock_name: 股票名称
        """
        recent_df = df.tail(200).copy()  # 选取最近 60 条数据并创建副本
        macd = ta.trend.MACD(recent_df['Close'])
        recent_df['MACD'] = macd.macd()
        recent_df['MACD_signal'] = macd.macd_signal()
        recent_df['MACD_diff'] = macd.macd_diff()

        plt.figure(figsize=(12, 6))
        plt.plot(recent_df.index, recent_df['MACD'], label='MACD')
        plt.plot(recent_df.index, recent_df['MACD_signal'], label='MACD Signal')
        plt.bar(recent_df.index, recent_df['MACD_diff'], label='MACD Diff')
        plt.title(f'{stock_name} MACD 指标', fontproperties=matplotlib.font_manager.FontProperties(family='SimHei'))
        plt.xlabel('日期', fontproperties=matplotlib.font_manager.FontProperties(family='SimHei'))
        plt.ylabel('MACD 值', fontproperties=matplotlib.font_manager.FontProperties(family='SimHei'))
        plt.legend(prop=matplotlib.font_manager.FontProperties(family='SimHei'))

    def plot_rsi(self, df, stock_name):
        """
        绘制股票的 RSI 指标图
        :param df: 股票数据 DataFrame
        :param stock_name: 股票名称
        """
        recent_df = df.tail(200).copy()  # 选取最近 60 条数据并创建副本
        rsi = ta.momentum.RSIIndicator(recent_df['Close'])
        recent_df['RSI'] = rsi.rsi()
        plt.figure(figsize=(12, 6))
        plt.plot(recent_df.index, recent_df['RSI'])
        plt.axhline(y=30, color='r', linestyle='--', label='超卖线')
        plt.axhline(y=70, color='g', linestyle='--', label='超买线')
        plt.title(f'{stock_name} RSI 指标', fontproperties=matplotlib.font_manager.FontProperties(family='SimHei'))
        plt.xlabel('日期', fontproperties=matplotlib.font_manager.FontProperties(family='SimHei'))
        plt.ylabel('RSI 值', fontproperties=matplotlib.font_manager.FontProperties(family='SimHei'))
        plt.legend(prop=matplotlib.font_manager.FontProperties(family='SimHei'))

    def plot_bollinger_bands(self, df, stock_name):
        """
        绘制股票的布林带指标图
        :param df: 股票数据 DataFrame
        :param stock_name: 股票名称
        """
        recent_df = df.tail(200).copy()  # 选取最近 60 条数据并创建副本
        indicator_bb = ta.volatility.BollingerBands(recent_df['Close'])
        recent_df['bb_bbm'] = indicator_bb.bollinger_mavg()
        recent_df['bb_bbh'] = indicator_bb.bollinger_hband()
        recent_df['bb_bbl'] = indicator_bb.bollinger_lband()

        plt.figure(figsize=(12, 6))
        plt.plot(recent_df.index, recent_df['Close'], label='收盘价')
        plt.plot(recent_df.index, recent_df['bb_bbm'], label='中轨')
        plt.plot(recent_df.index, recent_df['bb_bbh'], label='上轨')
        plt.plot(recent_df.index, recent_df['bb_bbl'], label='下轨')
        plt.title(f'{stock_name} 布林带指标', fontproperties=matplotlib.font_manager.FontProperties(family='SimHei'))
        plt.xlabel('日期', fontproperties=matplotlib.font_manager.FontProperties(family='SimHei'))
        plt.ylabel('价格', fontproperties=matplotlib.font_manager.FontProperties(family='SimHei'))
        plt.legend(prop=matplotlib.font_manager.FontProperties(family='SimHei'))

    def plot_moving_averages(self, df, stock_name):
        """
        绘制股票的移动平均线指标图
        :param df: 股票数据 DataFrame
        :param stock_name: 股票名称
        """
        recent_df = df.tail(500).copy()  # 选取最近 60 条数据并创建副本
        recent_df['MA_5'] = recent_df['Close'].rolling(window=5).mean()
        recent_df['MA_10'] = recent_df['Close'].rolling(window=10).mean()
        recent_df['MA_20'] = recent_df['Close'].rolling(window=20).mean()

        plt.figure(figsize=(12, 6))
        plt.plot(recent_df.index, recent_df['Close'], label='收盘价')
        plt.plot(recent_df.index, recent_df['MA_5'], label='5 日移动平均线')
        plt.plot(recent_df.index, recent_df['MA_10'], label='10 日移动平均线')
        plt.plot(recent_df.index, recent_df['MA_20'], label='20 日移动平均线')
        plt.title(f'{stock_name} 移动平均线指标', fontproperties=matplotlib.font_manager.FontProperties(family='SimHei'))
        plt.xlabel('日期', fontproperties=matplotlib.font_manager.FontProperties(family='SimHei'))
        plt.ylabel('价格', fontproperties=matplotlib.font_manager.FontProperties(family='SimHei'))
        plt.legend(prop=matplotlib.font_manager.FontProperties(family='SimHei'))

    def plot_time_sharing_chart(self, df, stock_name):
        """
        绘制股票的分时图
        :param df: 股票数据 DataFrame
        :param stock_name: 股票名称
        """
        recent_df = df.tail(500).copy()  # 选取最近 60 条数据并创建副本
        # 这里简单假设实时数据为分时数据
        plt.figure(figsize=(12, 6))
        recent_df['Close'].plot(title=f'{stock_name} 分时图')
        plt.xlabel('时间', fontproperties=matplotlib.font_manager.FontProperties(family='SimHei'))
        plt.ylabel('价格', fontproperties=matplotlib.font_manager.FontProperties(family='SimHei'))

    def plot_all_charts(self, df, stock_name):
        """
        绘制所有类型的股票图表
        :param df: 股票数据 DataFrame
        :param stock_name: 股票名称
        """
        self.plot_line_chart(df, stock_name)
        self.plot_bar_chart(df, stock_name)
        self.plot_candlestick_chart(df, stock_name)
        self.plot_macd(df, stock_name)
        self.plot_rsi(df, stock_name)
        self.plot_bollinger_bands(df, stock_name)
        self.plot_moving_averages(df, stock_name)
        self.plot_time_sharing_chart(df, stock_name)
        plt.show()

    def cleanup(self):
        """
        关闭数据库连接
        """
        self.mycursor.close()
        self.mydb.close()


if __name__ == "__main__":
    analyzer = StockChartAnalyzer()
    stock_name = "洪城环境"
    df = analyzer.get_stock_data(stock_name)
    if df is not None:
        analyzer.plot_all_charts(df, stock_name)
    analyzer.cleanup()
