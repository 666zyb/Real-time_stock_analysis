import sys
import os
from django.conf import settings
from ..models import StockHistory
import pandas as pd
import matplotlib

matplotlib.use('Agg')  # 使用非交互式后端，避免GUI窗口
import matplotlib.pyplot as plt
import mplfinance as mpf
import ta  # 确保安装了ta-lib
import datetime
import logging

# 导入现有的图表模块
sys.path.append(os.path.join(settings.BASE_DIR, 'stock_analysis'))
from News_analysis.news_stock_analysis import NewsStockAnalyzer

logger = logging.getLogger(__name__)


class ChartService:
    def __init__(self):
        # 不再需要chart_analyzer
        pass

    def generate_stock_charts(self, stock_name,period='day'):
        """生成股票图表，带缓存功能,支持多周期"""
        chart_dir = os.path.join('static', 'images', 'charts')
        os.makedirs(chart_dir, exist_ok=True)

        # 检查缓存是否存在且仍然有效,包含周期信息
        cache_valid = True
        for chart_type in ['candlestick', 'macd', 'rsi', 'bollinger', 'ma']:
            chart_path = os.path.join(chart_dir, f"{stock_name}_{period}_{chart_type}.png")
            if not os.path.exists(chart_path):
                cache_valid = False
                break

            # 检查文件是否是今天生成的
            file_time = os.path.getmtime(chart_path)
            if datetime.datetime.fromtimestamp(file_time).date() != datetime.datetime.now().date():
                cache_valid = False
                break

        # 如果缓存有效，直接返回图表路径
        if cache_valid:
            print(f"使用缓存的f{period}图表")
            return {
                'candlestick': f"/static/images/charts/{stock_name}_{period}candlestick.png",
                'macd': f"/static/images/charts/{stock_name}_macd.png",
                'rsi': f"/static/images/charts/{stock_name}_rsi.png",
                'bollinger': f"/static/images/charts/{stock_name}_bollinger.png",
                'ma': f"/static/images/charts/{stock_name}_ma.png"
            }

        # 缓存无效，重新生成图表
        # 获取股票历史数据
        history_data = self.get_stock_history(stock_name)

        if history_data is None or history_data.empty:
            print("无法获取历史数据")
            return None

        # 生成图表并返回路径
        chart_files = {}

        # 处理数据，准备绘图
        df = self.prepare_data_for_charts(history_data)

        # 生成K线图
        self.plot_candlestick_chart(df, stock_name, chart_dir)
        chart_files['candlestick'] = f"/static/images/charts/{stock_name}_candlestick.png"

        # 生成MACD图
        self.plot_macd(df, stock_name, chart_dir)
        chart_files['macd'] = f"/static/images/charts/{stock_name}_macd.png"

        # 生成RSI图
        self.plot_rsi(df, stock_name, chart_dir)
        chart_files['rsi'] = f"/static/images/charts/{stock_name}_rsi.png"

        # 生成布林带图
        self.plot_bollinger_bands(df, stock_name, chart_dir)
        chart_files['bollinger'] = f"/static/images/charts/{stock_name}_bollinger.png"

        # 生成移动平均线图
        self.plot_moving_averages(df, stock_name, chart_dir)
        chart_files['ma'] = f"/static/images/charts/{stock_name}_ma.png"

        return chart_files

    def get_stock_history(self, stock_name):
        """获取股票历史数据，优化查询"""
        from django.db import connection
        import pandas as pd

        # 使用参数化查询，避免SQL注入
        table_name = f"{stock_name}_history"
        query = f"""
            SELECT 
                `日期` AS date, 
                `开盘价` AS open_price, 
                `收盘价` AS close_price, 
                `最高价` AS high_price, 
                `最低价` AS low_price, 
                `成交量(手)` AS volume, 
                `成交额(元)` AS amount 
            FROM {table_name} 
            ORDER BY `日期` DESC 
            LIMIT 200;
        """

        try:
            # 使用上下文管理器自动管理连接
            with connection.cursor() as cursor:
                cursor.execute(query)
                columns = [col[0] for col in cursor.description]
                rows = cursor.fetchall()
                print(f"查询到 {len(rows)} 条数据")

            # 使用列名创建DataFrame
            df = pd.DataFrame(rows, columns=columns)
            return df
        except Exception as e:
            print(f"查询失败: {e}, SQL: {query}")
            return pd.DataFrame()  # 返回空DataFrame而不是None，避免类型错误

    def prepare_data_for_charts(self, df):
        """准备数据用于绘图"""
        # 确保 date 列是 datetime 类型
        df['date'] = pd.to_datetime(df['date'])

        # 设置日期为索引
        df = df.set_index('date')

        # 确保数据类型正确
        numeric_columns = ['open_price', 'close_price', 'high_price', 'low_price', 'volume', 'amount']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # 添加mplfinance需要的列名
        df = df.rename(columns={
            'open_price': 'Open',
            'close_price': 'Close',
            'high_price': 'High',
            'low_price': 'Low',
            'volume': 'Volume',
            'amount': 'Amount'
        })

        # 按日期升序排序，因为mplfinance更喜欢时间序列
        df = df.sort_index()

        return df

    def plot_candlestick_chart(self, df, stock_name, chart_dir):
        """绘制股票的 K 线图"""
        plt.figure(figsize=(12, 6))

        # 使用mplfinance绘制K线图
        kwargs = {
            'type': 'candle',
            'style': 'yahoo',
            'figsize': (12, 6),
            'title': f'{stock_name} 股票 K 线图',
            'ylabel': '价格',
            'ylabel_lower': '成交量',
            'savefig': f'{chart_dir}/{stock_name}_candlestick.png'
        }

        try:
            mpf.plot(df, **kwargs)
            print(f"成功保存K线图: {chart_dir}/{stock_name}_candlestick.png")
        except Exception as e:
            print(f"绘制K线图出错: {e}")
            # 创建一个简单的错误图像作为替代
            plt.figure(figsize=(12, 6))
            plt.text(0.5, 0.5, f'无法绘制K线图: {str(e)}', ha='center', va='center', fontsize=12)
            plt.savefig(f'{chart_dir}/{stock_name}_candlestick.png')
            plt.close()

    def plot_macd(self, df, stock_name, chart_dir):
        """绘制 MACD 图"""
        try:
            # 计算 MACD
            macd = ta.trend.MACD(df['Close'])

            plt.figure(figsize=(12, 6))
            plt.plot(macd.macd, label='MACD', color='blue')
            plt.plot(macd.macd_signal, label='Signal Line', color='red')
            plt.bar(df.index, macd.macd_diff, label='Histogram', color='green', alpha=0.5)
            plt.title(f'{stock_name} MACD')
            plt.legend()
            plt.grid(True)

            # 保存图片
            plt.savefig(f'{chart_dir}/{stock_name}_macd.png', dpi=100)
            plt.close()
            logger.info(f"成功保存MACD图: {chart_dir}/{stock_name}_macd.png")
        except Exception as e:
            logger.error(f"绘制MACD图出错: {e}", exc_info=True)
            # 创建一个简单的错误图像作为替代
            plt.figure(figsize=(12, 6))
            plt.text(0.5, 0.5, f'无法绘制MACD图: {str(e)}', ha='center', va='center', fontsize=12)
            plt.savefig(f'{chart_dir}/{stock_name}_macd.png')
            plt.close()

    def plot_rsi(self, df, stock_name, chart_dir):
        """绘制 RSI 图"""
        try:
            # 计算 RSI
            rsi = ta.momentum.RSIIndicator(df['Close']).rsi()

            plt.figure(figsize=(12, 6))
            plt.plot(rsi, label='RSI', color='purple')
            plt.axhline(70, linestyle='--', alpha=0.5, color='red')
            plt.axhline(30, linestyle='--', alpha=0.5, color='green')
            plt.title(f'{stock_name} RSI')
            plt.legend()
            plt.grid(True)

            # 保存图片
            plt.savefig(f'{chart_dir}/{stock_name}_rsi.png')
            plt.close()
            print(f"成功保存RSI图: {chart_dir}/{stock_name}_rsi.png")
        except Exception as e:
            print(f"绘制RSI图出错: {e}")
            # 创建一个简单的错误图像作为替代
            plt.figure(figsize=(12, 6))
            plt.text(0.5, 0.5, f'无法绘制RSI图: {str(e)}', ha='center', va='center', fontsize=12)
            plt.savefig(f'{chart_dir}/{stock_name}_rsi.png')
            plt.close()

    def plot_bollinger_bands(self, df, stock_name, chart_dir):
        """绘制布林带图"""
        try:
            # 计算布林带
            bollinger = ta.volatility.BollingerBands(df['Close'])

            plt.figure(figsize=(12, 6))
            plt.plot(df['Close'], label='价格', color='blue')
            plt.plot(bollinger.bollinger_hband(), label='上轨', color='red')
            plt.plot(bollinger.bollinger_mavg(), label='中轨', color='purple')
            plt.plot(bollinger.bollinger_lband(), label='下轨', color='green')
            plt.title(f'{stock_name} 布林带')
            plt.legend()
            plt.grid(True)

            # 保存图片
            plt.savefig(f'{chart_dir}/{stock_name}_bollinger.png')
            plt.close()
            print(f"成功保存布林带图: {chart_dir}/{stock_name}_bollinger.png")
        except Exception as e:
            print(f"绘制布林带图出错: {e}")
            # 创建一个简单的错误图像作为替代
            plt.figure(figsize=(12, 6))
            plt.text(0.5, 0.5, f'无法绘制布林带图: {str(e)}', ha='center', va='center', fontsize=12)
            plt.savefig(f'{chart_dir}/{stock_name}_bollinger.png')
            plt.close()

    def plot_moving_averages(self, df, stock_name, chart_dir):
        """绘制移动平均线图"""
        try:
            # 计算移动平均线
            ma5 = df['Close'].rolling(window=5).mean()
            ma10 = df['Close'].rolling(window=10).mean()
            ma20 = df['Close'].rolling(window=20).mean()
            ma60 = df['Close'].rolling(window=60).mean()

            plt.figure(figsize=(12, 6))
            plt.plot(df['Close'], label='价格', color='black')
            plt.plot(ma5, label='MA5', color='red')
            plt.plot(ma10, label='MA10', color='blue')
            plt.plot(ma20, label='MA20', color='green')
            plt.plot(ma60, label='MA60', color='purple')
            plt.title(f'{stock_name} 移动平均线')
            plt.legend()
            plt.grid(True)

            # 保存图片
            plt.savefig(f'{chart_dir}/{stock_name}_ma.png')
            plt.close()
            print(f"成功保存移动平均线图: {chart_dir}/{stock_name}_ma.png")
        except Exception as e:
            print(f"绘制移动平均线图出错: {e}")
            # 创建一个简单的错误图像作为替代
            plt.figure(figsize=(12, 6))
            plt.text(0.5, 0.5, f'无法绘制移动平均线图: {str(e)}', ha='center', va='center', fontsize=12)
            plt.savefig(f'{chart_dir}/{stock_name}_ma.png')
            plt.close()