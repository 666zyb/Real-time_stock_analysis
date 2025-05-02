import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
import itertools


class ARIMAModel:
    def __init__(self):
        self.model = None
        self.order = None
        self.seasonal_order = None

    def check_stationarity(self, timeseries):
        """检查时间序列是否平稳"""
        result = adfuller(timeseries.dropna())
        print('ADF统计量: %f' % result[0])
        print('p值: %f' % result[1])
        print('临界值:')
        for key, value in result[4].items():
            print('\t%s: %.3f' % (key, value))

        # 如果p值小于0.05，则序列是平稳的
        if result[1] <= 0.05:
            print("数据是平稳的，可以直接用于ARIMA模型")
        else:
            print("数据不是平稳的，可能需要差分")

    def auto_arima(self, timeseries, seasonal=False):
        """自定义的auto_arima函数，简化版本"""
        best_aic = float('inf')
        best_order = None
        best_seasonal_order = None

        # 定义参数范围
        p_values = range(0, 3)
        d_values = range(0, 2)
        q_values = range(0, 3)

        # 季节性参数
        if seasonal:
            P_values = range(0, 2)
            D_values = range(0, 1)
            Q_values = range(0, 2)
            m_values = [5]  # 每周5个交易日
        else:
            P_values = D_values = Q_values = [0]
            m_values = [0]

        # 遍历所有参数组合
        print("开始ARIMA参数搜索...")

        # 非季节性参数遍历
        for p, d, q in itertools.product(p_values, d_values, q_values):
            # 季节性参数遍历
            for P, D, Q, m in itertools.product(P_values, D_values, Q_values, m_values):

                # 跳过无季节项的m=0情况
                if m == 0 and (P != 0 or D != 0 or Q != 0):
                    continue

                seasonal_order = (P, D, Q, m) if m > 0 else None
                order = (p, d, q)

                try:
                    if seasonal_order:
                        model = SARIMAX(
                            timeseries,
                            order=order,
                            seasonal_order=seasonal_order,
                            enforce_stationarity=False,
                            enforce_invertibility=False
                        )
                    else:
                        model = ARIMA(timeseries, order=order)

                    result = model.fit()
                    aic = result.aic

                    if aic < best_aic:
                        best_aic = aic
                        best_order = order
                        best_seasonal_order = seasonal_order

                        print(f"新的最佳模型 - ARIMA{order}")
                        if seasonal_order:
                            print(f"季节性参数: {seasonal_order}")
                        print(f"AIC: {aic}")

                except Exception as e:
                    continue

        if best_order is None:
            print("无法找到有效的ARIMA模型，使用默认 (1,1,1)")
            best_order = (1, 1, 1)

        print(f"最佳ARIMA参数: {best_order}")
        if best_seasonal_order:
            print(f"最佳季节性参数: {best_seasonal_order}")

        self.order = best_order
        self.seasonal_order = best_seasonal_order

        # 创建并返回最佳模型
        if best_seasonal_order:
            model = SARIMAX(
                timeseries,
                order=best_order,
                seasonal_order=best_seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
        else:
            model = ARIMA(timeseries, order=best_order)

        return model

    def train(self, timeseries, order=None, seasonal_order=None):
        """训练ARIMA模型"""
        if order is None:
            if self.order is None:
                print("未指定ARIMA参数，将使用auto_arima自动确定")
                model = self.auto_arima(timeseries, seasonal=(seasonal_order is not None))
                self.result = model.fit()
                return self.result
            order = self.order

        if seasonal_order is not None:
            # 使用SARIMAX模型（带季节性）
            self.model = SARIMAX(timeseries,
                                 order=order,
                                 seasonal_order=seasonal_order,
                                 enforce_stationarity=False,
                                 enforce_invertibility=False)
        else:
            # 使用ARIMA模型（不带季节性）
            self.model = ARIMA(timeseries, order=order)

        self.result = self.model.fit()
        print(self.result.summary())
        return self.result

    def predict(self, steps=5):
        """预测未来n天的值"""
        if self.result is None:
            raise Exception("请先训练模型")

        forecast = self.result.forecast(steps=steps)
        return forecast

    def plot_prediction(self, timeseries, forecast, title='ARIMA预测'):
        """绘制预测结果"""
        plt.figure(figsize=(12, 6))
        plt.plot(timeseries, label='历史数据')

        # 预测起点
        forecast_start = len(timeseries)

        # 创建预测日期索引
        if isinstance(timeseries.index, pd.DatetimeIndex):
            last_date = timeseries.index[-1]
            forecast_index = pd.date_range(start=last_date, periods=len(forecast) + 1)[1:]
            plt.plot(forecast_index, forecast, color='red', label='预测')
        else:
            forecast_index = range(forecast_start, forecast_start + len(forecast))
            plt.plot(forecast_index, forecast, color='red', label='预测')

        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.show()


class SimpleARIMAModel:
    def __init__(self):
        self.model = None
        self.order = (1, 1, 1)  # 默认参数

    def check_stationarity(self, timeseries):
        """检查时间序列是否平稳"""
        result = adfuller(timeseries.dropna())
        print('ADF统计量: %f' % result[0])
        print('p值: %f' % result[1])

        # 如果p值小于0.05，则序列是平稳的
        if result[1] <= 0.05:
            print("数据是平稳的，可以直接用于ARIMA模型")
            return True
        else:
            print("数据不是平稳的，可能需要差分")
            return False

    def train(self, timeseries, order=None):
        """训练ARIMA模型"""
        if order is not None:
            self.order = order

        is_stationary = self.check_stationarity(timeseries)

        # 如果数据不是平稳的，使用默认差分d=1
        if not is_stationary and self.order[1] == 0:
            self.order = (self.order[0], 1, self.order[2])
            print(f"数据不平稳，调整参数为: {self.order}")

        # 使用ARIMA模型
        self.model = ARIMA(timeseries, order=self.order)
        self.result = self.model.fit()
        print(self.result.summary())
        return self.result

    def predict(self, steps=5):
        """预测未来n天的值"""
        if self.result is None:
            raise Exception("请先训练模型")

        forecast = self.result.forecast(steps=steps)
        return forecast

    def plot_prediction(self, timeseries, forecast, title='ARIMA预测'):
        """绘制预测结果"""
        plt.figure(figsize=(12, 6))
        plt.plot(timeseries, label='历史数据')

        # 预测起点
        forecast_start = len(timeseries)

        # 创建预测日期索引
        if isinstance(timeseries.index, pd.DatetimeIndex):
            last_date = timeseries.index[-1]
            forecast_index = pd.date_range(start=last_date, periods=len(forecast) + 1)[1:]
            plt.plot(forecast_index, forecast, color='red', label='预测')
        else:
            forecast_index = range(forecast_start, forecast_start + len(forecast))
            plt.plot(forecast_index, forecast, color='red', label='预测')

        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.show() 