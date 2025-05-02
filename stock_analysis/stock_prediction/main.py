import os
import pandas as pd
import numpy as np
import json
import matplotlib

matplotlib.use('Agg')  # 必须在导入 pyplot 之前设置
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

# 导入自定义模块
from data_loader import DataLoader
from feature_engineering import FeatureEngineering
from models.arima_model import ARIMAModel
from models.ml_model import MLModel
from models.dl_model import DLModel
from utils.sentiment_analysis import SentimentAnalyzer


def load_config():
    """加载配置文件"""
    with open('config.json', 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config


def predict_stock(stock_code, stock_name, industry, days=5):
    """对单支股票进行预测"""
    print(f"\n{'=' * 50}")
    print(f"开始预测股票: {stock_name}({stock_code}), 行业: {industry}")
    print(f"{'=' * 50}")

    # 加载数据
    data_loader = DataLoader()

    # 获取历史数据 (默认近两年数据，增加数据量)
    daily_data = data_loader.get_daily_data(stock_code, stock_name, days=730)

    # 关键检查：确保daily_data不为空
    if daily_data is None or daily_data.empty:
        print(f"警告: 股票 {stock_code} 的历史数据不足，无法进行可靠预测")
        return None

    print(f"成功加载历史数据，共 {len(daily_data)} 行")

    # 获取实时数据
    realtime_data = data_loader.get_realtime_data(stock_code)
    if realtime_data:
        print(f"最新实时数据: {realtime_data}")

        # 确保历史数据中的价格列名称与特征工程模块匹配
        price_col = '收盘价' if '收盘价' in daily_data.columns else None
        if price_col and '当前价格' in realtime_data:
            print(f"当前价格: {realtime_data['当前价格']}, 历史最后收盘价: {daily_data[price_col].iloc[-1]}")
    else:
        print("无法获取实时数据，将使用历史数据的最后一个交易日数据")

    # 获取情感数据
    try:
        news_sentiment = data_loader.get_news_sentiment(stock_code, days=30)
        print(f"成功获取情感数据，包含 {len(news_sentiment)} 天的数据")
    except Exception as e:
        print(f"获取情感数据失败: {e}")
        # 创建一个空的情感数据框架
        dates = [(datetime.now() - timedelta(days=i)).date() for i in range(30)]
        news_sentiment = pd.DataFrame({
            'date': dates,
            'sentiment': [0.0] * len(dates)
        }).set_index('date')

    # 特征工程
    try:
        fe = FeatureEngineering()
        # 添加技术指标
        print("开始添加技术指标...")
        daily_data = fe.add_technical_indicators(daily_data)
        print(f"技术指标添加完成，数据形状: {daily_data.shape}")

        # 重要检查：确保处理后的数据不为空
        if daily_data.empty:
            print("技术指标计算后数据为空，无法继续预测")
            return None

        # 如果有情感数据，添加情感特征
        if news_sentiment is not None and not news_sentiment.empty:
            try:
                print("开始添加情感特征...")
                daily_data = fe.add_sentiment_features(daily_data, news_sentiment)
                print("情感特征添加完成")
            except Exception as e:
                print(f"添加情感特征失败: {e}")
                print("继续使用仅技术指标的数据")
        else:
            print("跳过情感特征添加，使用仅技术指标的数据")

        # 准备机器学习特征
        ml_data = fe.prepare_ml_features(daily_data, target_col='收盘价', n_days=days)

        # 确保ml_data不为空
        if ml_data.empty:
            print("准备机器学习特征后数据为空，无法继续预测")
            return None
    except Exception as e:
        print(f"特征工程处理失败: {e}")
        import traceback
        traceback.print_exc()
        return None

    # 保存处理后的数据以便调试
    processed_data_dir = 'processed_data'
    os.makedirs(processed_data_dir, exist_ok=True)
    ml_data.to_csv(f'{processed_data_dir}/{stock_code}_processed.csv', index=False)

    print("\n1. 使用ARIMA模型预测")
    # 使用ARIMA模型预测
    try:
        arima = ARIMAModel()
        # 检查平稳性
        arima.check_stationarity(ml_data['收盘价'])

        # 使用auto_arima找到最佳参数
        best_model = arima.auto_arima(ml_data['收盘价'])

        # 训练模型
        arima_result = arima.train(ml_data['收盘价'], order=best_model.order,
                                   seasonal_order=getattr(best_model, 'seasonal_order', None))

        # 预测未来n天
        arima_forecast = arima.predict(steps=days)
        print(f"\nARIMA预测未来{days}天的价格:")
        print(arima_forecast)

        # 绘制预测结果
        arima.plot_prediction(ml_data['收盘价'][-60:], arima_forecast,
                              title=f'ARIMA预测 - {stock_name}({stock_code})')
    except Exception as e:
        print(f"ARIMA预测出错：{e}")

    print("\n2. 使用机器学习模型预测")
    # 准备特征和目标变量
    features = ml_data.drop(['target'], axis=1)
    target = ml_data['target']

    # 移除不需要的列
    cols_to_drop = ['trade_date', 'stock_code', 'stock_name', '收盘价']
    features = features.drop([col for col in cols_to_drop if col in features.columns], axis=1)

    # 移除非数值列
    features = features.select_dtypes(include=[np.number])

    # 使用随机森林模型
    try:
        rf_model = MLModel(model_type='rf')
        rf_results = rf_model.train(features, target, tune_hyperparams=True)

        # 使用最后的数据点预测未来
        last_data_point = features.iloc[-1:].copy()

        # 预测并输出结果
        prediction = rf_model.predict(last_data_point)[0]
        current_price = ml_data['收盘价'].iloc[-1]
        predicted_price = current_price * (1 + prediction)

        print(f"\n随机森林预测结果:")
        print(f"当前价格: {current_price:.2f}")
        print(f"预测变动百分比: {prediction * 100:.2f}%")
        print(f"预测 {days} 天后价格: {predicted_price:.2f}")

        # 展示特征重要性
        rf_model.plot_feature_importance(features)
    except Exception as e:
        print(f"随机森林预测出错：{e}")

    print("\n3. 使用深度学习模型预测")
    # 使用深度学习模型
    try:
        # 准备LSTM需要的数据
        dl_features = features.copy()
        dl_target = pd.Series(ml_data['target'].values, index=ml_data.index[:-days])

        # 训练深度学习模型
        dl_model = DLModel(model_type='lstm')
        dl_results = dl_model.train(dl_features, dl_target, time_steps=10, epochs=50, batch_size=32)

        # 预测未来n天
        future_pred = dl_model.predict_future(dl_features, steps=days, time_steps=10)

        print(f"\nLSTM预测未来{days}天的价格:")
        for i, pred in enumerate(future_pred):
            print(f"第{i + 1}天: {pred[0]:.2f}")

        # 绘制学习曲线
        dl_model.plot_learning_curve(dl_results['history'])

        # 绘制预测结果 - 修复维度不匹配问题
        last_actual = dl_target.iloc[-20:].values
        plt.figure(figsize=(12, 6))
        plt.plot(range(len(last_actual)), last_actual, label='历史价格')

        # 修复：确保x和y数组的长度相同
        x_pred = range(len(last_actual) - 1, len(last_actual) + len(future_pred) - 1)
        y_pred = [last_actual[-1]] + [p[0] for p in future_pred]

        # 检查并调整数组长度
        min_len = min(len(x_pred), len(y_pred))
        plt.plot(x_pred[:min_len], y_pred[:min_len], 'r--', label='预测价格')

        plt.title(f'LSTM预测 - {stock_name}({stock_code})')
        plt.legend()
        plt.grid(True)
        plt.show()

        # 保存LSTM预测结果供综合预测使用
        lstm_price = future_pred[-1][0]
        lstm_change = (lstm_price / current_price - 1) * 100
        print(f"LSTM预测最终价格: {lstm_price:.2f}, 变动: {lstm_change:.2f}%")

    except Exception as e:
        print(f"深度学习预测出错：{e}")
        lstm_price = None
        lstm_change = None

    # 输出综合预测结果
    try:
        print("\n综合预测结果:")
        current_price = ml_data['收盘价'].iloc[-1]

        # ARIMA预测结果
        arima_last_price = arima_forecast[-1]
        arima_change = (arima_last_price / current_price - 1) * 100

        # 随机森林预测结果
        rf_price = predicted_price
        rf_change = prediction * 100

        # LSTM预测结果
        lstm_price = lstm_price
        lstm_change = lstm_change

        # 计算综合预测（简单平均）
        ensemble_price = (arima_last_price + rf_price + lstm_price) / 3
        ensemble_change = (ensemble_price / current_price - 1) * 100

        print(f"当前价格: {current_price:.2f}")
        print(f"ARIMA预测: {arima_last_price:.2f} (变动: {arima_change:.2f}%)")
        print(f"随机森林预测: {rf_price:.2f} (变动: {rf_change:.2f}%)")
        print(f"LSTM预测: {lstm_price:.2f} (变动: {lstm_change:.2f}%)")
        print(f"综合预测: {ensemble_price:.2f} (变动: {ensemble_change:.2f}%)")

        # 预测趋势判断
        if ensemble_change > 2:
            trend = "强烈上涨"
        elif ensemble_change > 0.5:
            trend = "上涨"
        elif ensemble_change > -0.5:
            trend = "横盘整理"
        elif ensemble_change > -2:
            trend = "下跌"
        else:
            trend = "强烈下跌"

        print(f"\n未来 {days} 天预测趋势: {trend}")

        # 返回预测结果
        return {
            'stock_code': stock_code,
            'stock_name': stock_name,
            'current_price': current_price,
            'arima_price': arima_last_price,
            'rf_price': rf_price,
            'lstm_price': lstm_price,
            'ensemble_price': ensemble_price,
            'ensemble_change': ensemble_change,
            'trend': trend
        }
    except Exception as e:
        print(f"综合预测出错：{e}")
        return None


def main():
    # 加载配置
    config = load_config()
    stocks = config['stocks']

    # 创建结果目录
    results_dir = 'prediction_results'
    os.makedirs(results_dir, exist_ok=True)

    # 预测结果列表
    all_results = []

    # 对每只股票进行预测
    for stock in stocks:
        stock_code = stock['code']
        stock_name = stock['name']
        industry = stock['industry']

        result = predict_stock(stock_code, stock_name, industry)
        if result:
            all_results.append(result)

    # 生成综合报告
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_df = results_df.sort_values('ensemble_change', ascending=False)

        # 保存到Excel
        now = datetime.now().strftime('%Y%m%d_%H%M%S')
        excel_path = f'{results_dir}/stock_predictions_{now}.xlsx'
        results_df.to_excel(excel_path, index=False)

        print(f"\n\n{'=' * 60}")
        print(f"预测报告已保存至: {excel_path}")
        print(f"{'=' * 60}")

        # 打印投资建议
        print("\n投资建议:")
        for i, row in results_df.iterrows():
            print(f"{row['stock_name']}({row['stock_code']}): {row['trend']} - 预期变动: {row['ensemble_change']:.2f}%")


if __name__ == "__main__":
    main() 