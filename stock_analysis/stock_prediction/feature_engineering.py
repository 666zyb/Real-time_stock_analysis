import pandas as pd
import numpy as np
from ta import add_all_ta_features
from ta.trend import SMAIndicator, MACD
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands


class FeatureEngineering:
    def __init__(self):
        pass

    def sma(self, series, periods):
        """简单移动平均线"""
        return series.rolling(window=periods).mean()

    def ema(self, series, periods):
        """指数移动平均线"""
        return series.ewm(span=periods, adjust=False).mean()

    def rsi(self, close, periods=14):
        """相对强弱指数"""
        delta = close.diff()
        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)

        ema_up = self.ema(up, periods)
        ema_down = self.ema(down, periods)

        rs = ema_up / ema_down
        return 100 - (100 / (1 + rs))

    def macd(self, close, fast=12, slow=26, signal=9):
        """MACD指标"""
        fast_ema = self.ema(close, fast)
        slow_ema = self.ema(close, slow)
        macd_line = fast_ema - slow_ema
        signal_line = self.ema(macd_line, signal)
        histogram = macd_line - signal_line

        return macd_line, signal_line, histogram

    def bollinger_bands(self, close, window=20, std=2):
        """布林带"""
        middle = self.sma(close, window)
        sigma = close.rolling(window=window).std()

        upper = middle + std * sigma
        lower = middle - std * sigma

        return upper, middle, lower

    def stochastic_oscillator(self, high, low, close, k_window=14, d_window=3):
        """KDJ指标"""
        # 计算K值
        lowest_low = low.rolling(window=k_window).min()
        highest_high = high.rolling(window=k_window).max()

        # 避免除以零
        denom = highest_high - lowest_low
        denom = denom.replace(0, 0.000001)

        K = 100 * ((close - lowest_low) / denom)
        # 计算D值
        D = K.rolling(window=d_window).mean()
        # 计算J值
        J = 3 * K - 2 * D

        return K, D, J

    def add_technical_indicators(self, df):
        """添加技术指标"""
        # 先检查数据框是否为空
        if df is None or df.empty:
            print("警告: 输入的数据框为空，无法计算技术指标")
            return df

        # 打印数据框信息
        print(f"原始数据形状: {df.shape}")
        print(f"原始数据列: {df.columns.tolist()}")

        # 确保列名符合预期
        price_col = None
        high_col = None
        low_col = None
        volume_col = None

        # 寻找价格列
        for col in ['close', '收盘价', '当前价格']:
            if col in df.columns:
                price_col = col
                break

        # 寻找最高价列
        for col in ['high', '最高价', '今日最高价']:
            if col in df.columns:
                high_col = col
                break

        # 寻找最低价列
        for col in ['low', '最低价', '今日最低价']:
            if col in df.columns:
                low_col = col
                break

        # 寻找成交量列
        for col in ['volume', '成交量', '成交量(手)']:
            if col in df.columns:
                volume_col = col
                break

        print(f"使用的列: 价格={price_col}, 最高价={high_col}, 最低价={low_col}, 成交量={volume_col}")

        # 复制DataFrame以避免警告
        df = df.copy()

        # 确保关键列存在
        if price_col is None:
            print(f"找不到价格列，无法继续计算技术指标")
            return df

        # 将列转换为数值类型
        for col in [price_col, high_col, low_col, volume_col]:
            if col is not None and col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # 计算移动平均线
        if price_col in df.columns:
            for period in [5, 10, 20, 30, 60]:
                if len(df) >= period:  # 只有当数据量足够时才计算
                    df[f'MA{period}'] = self.sma(df[price_col], period)

            # 计算RSI
            if len(df) >= 14:
                df['RSI'] = self.rsi(df[price_col], 14)

            # 计算MACD
            if len(df) >= 26:
                macd_line, signal_line, histogram = self.macd(df[price_col])
                df['MACD'] = macd_line
                df['MACD_SIGNAL'] = signal_line
                df['MACD_HIST'] = histogram

            # 计算布林带
            if len(df) >= 20:
                upper, middle, lower = self.bollinger_bands(df[price_col])
                df['UPPER'] = upper
                df['MIDDLE'] = middle
                df['LOWER'] = lower

            # 计算价格变动百分比
            if len(df) >= 2:
                df['price_pct_change'] = df[price_col].pct_change() * 100

        # 计算KDJ
        if all(col is not None and col in df.columns for col in [high_col, low_col, price_col]) and len(df) >= 9:
            K, D, J = self.stochastic_oscillator(df[high_col], df[low_col], df[price_col], k_window=9, d_window=3)
            df['K'] = K
            df['D'] = D
            df['J'] = J

        # 成交量指标
        if volume_col is not None and volume_col in df.columns:
            if len(df) >= 5:
                df['volume_ma5'] = self.sma(df[volume_col], 5)
            if len(df) >= 10:
                df['volume_ma10'] = self.sma(df[volume_col], 10)
            if len(df) >= 2:
                df['volume_pct_change'] = df[volume_col].pct_change() * 100

        # 使用前向填充和后向填充结合的方式处理NaN
        df = df.fillna(method='ffill').fillna(method='bfill')

        # 如果仍有NaN值，用0填充
        df = df.fillna(0)

        print(f"处理后数据形状: {df.shape}")
        return df

    def add_sentiment_features(self, df, sentiment_df):
        """添加情感特征"""
        # 确定日期列名
        date_col = None
        if 'trade_date' in df.columns:
            date_col = 'trade_date'
        elif '日期' in df.columns:
            date_col = '日期'
        else:
            print("警告: 找不到日期列，无法添加情感特征")
            return df

        # 确保日期格式一致
        if isinstance(df.index, pd.DatetimeIndex):
            df_merged = df.copy()
            sentiment_df.index = pd.to_datetime(sentiment_df.index)
            for col in sentiment_df.columns:
                df_merged = df_merged.join(sentiment_df[[col]], how='left')
        else:
            df_merged = df.copy()
            df_merged[date_col] = pd.to_datetime(df_merged[date_col])
            sentiment_df.index = pd.to_datetime(sentiment_df.index)
            df_merged = df_merged.set_index(date_col)
            for col in sentiment_df.columns:
                df_merged = df_merged.join(sentiment_df[[col]], how='left')
            df_merged = df_merged.reset_index()

        # 填充缺失值
        if 'sentiment' in df_merged.columns:
            df_merged['sentiment'] = df_merged['sentiment'].fillna(0)

        return df_merged

    def normalize_features(self, df, exclude_cols=None):
        """标准化特征"""
        if exclude_cols is None:
            exclude_cols = []

        # 确定要标准化的列（排除不需要标准化的列）
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cols_to_normalize = [col for col in numeric_cols if col not in exclude_cols]

        # 标准化
        for col in cols_to_normalize:
            mean = df[col].mean()
            std = df[col].std()
            if std != 0:  # 避免除以零
                df[col] = (df[col] - mean) / std

        return df

    def prepare_ml_features(self, df, target_col='close', n_days=5):
        """准备机器学习特征和目标变量"""
        # 创建目标变量：未来n天的收盘价变动百分比
        price_col = target_col if target_col in df.columns else '收盘价'

        # 检查价格列是否存在
        if price_col not in df.columns:
            print(f"警告: 找不到价格列 {price_col}")
            print(f"可用的列: {', '.join(df.columns)}")
            for alt in ['close', '收盘价', '收盘', 'Close']:
                if alt in df.columns:
                    price_col = alt
                    print(f"使用 {alt} 作为价格列")
                    break
            else:
                raise ValueError(f"找不到可用的价格列，无法计算目标变量")

        # 创建目标变量
        df['target'] = df[price_col].shift(-n_days) / df[price_col] - 1

        # 处理日期特征
        date_col = None
        if 'trade_date' in df.columns:
            date_col = 'trade_date'
        elif '日期' in df.columns:
            date_col = '日期'

        if date_col:
            df[date_col] = pd.to_datetime(df[date_col])
            df['day_of_week'] = df[date_col].dt.dayofweek
            df['month'] = df[date_col].dt.month
            df['quarter'] = df[date_col].dt.quarter

        # 删除不需要的列和包含NaN的行
        cols_to_drop = ['stock_code', 'stock_name'] if 'stock_code' in df.columns else []
        df = df.drop(columns=cols_to_drop, errors='ignore')
        df = df.dropna()

        return df