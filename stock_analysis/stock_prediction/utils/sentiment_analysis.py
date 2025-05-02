import pandas as pd
import numpy as np
import re
import jieba
from sklearn.feature_extraction.text import CountVectorizer
from snownlp import SnowNLP


class SentimentAnalyzer:
    def __init__(self):
        # 股票相关词典，可以根据需要扩充
        self.pos_words = ['利好', '增长', '上涨', '盈利', '收益', '扩张', '繁荣', '突破', '强劲', '看涨']
        self.neg_words = ['利空', '下跌', '减少', '亏损', '萎缩', '衰退', '破位', '疲软', '看跌', '风险']

        # 行业特定词典（中药行业）
        self.pharma_pos_words = ['新药', '政策扶持', '销售增长', '专利', '研发突破', '中医药振兴']
        self.pharma_neg_words = ['副作用', '监管趋严', '药价下降', '集采影响', '原材料上涨', '假药']

    def analyze_news_sentiment(self, news_text, industry=None):
        """分析新闻情感倾向"""
        if not isinstance(news_text, str) or not news_text.strip():
            return 0.0

        try:
            # 使用SnowNLP进行情感分析
            s = SnowNLP(news_text)
            # SnowNLP的情感分值在0-1之间，1表示积极，0表示消极
            # 转换为-1到1的范围，便于后续计算
            sentiment_score = (s.sentiments - 0.5) * 2

            # 计算行业特定词汇的影响
            industry_sentiment = 0
            if industry == '中药':
                # 计算中药行业特定词汇的出现次数
                pos_count = sum([1 for word in self.pharma_pos_words if word in news_text])
                neg_count = sum([1 for word in self.pharma_neg_words if word in news_text])
                industry_sentiment = (pos_count - neg_count) * 0.1  # 权重因子

            # 股票市场常用词汇的影响
            market_pos_count = sum([1 for word in self.pos_words if word in news_text])
            market_neg_count = sum([1 for word in self.neg_words if word in news_text])
            market_sentiment = (market_pos_count - market_neg_count) * 0.1

            # 综合得分
            final_sentiment = sentiment_score + industry_sentiment + market_sentiment

            # 限制在-1到1的范围内
            final_sentiment = max(-1, min(1, final_sentiment))

            return final_sentiment
        except Exception as e:
            print(f"情感分析出错: {e}")
            return 0.0  # 出错时返回中性值

    def analyze_news_batch(self, news_df, text_column, industry=None):
        """批量分析新闻情感"""
        if text_column not in news_df.columns:
            raise ValueError(f"列 {text_column} 不存在于数据框中")

        # 应用情感分析函数到每条新闻
        news_df['sentiment'] = news_df[text_column].apply(
            lambda x: self.analyze_news_sentiment(x, industry)
        )

        return news_df

    def aggregate_daily_sentiment(self, news_df, date_column, sentiment_column='sentiment'):
        """聚合每日情感得分"""
        # 确保日期列是日期类型
        news_df[date_column] = pd.to_datetime(news_df[date_column])

        # 按日期聚合情感得分
        daily_sentiment = news_df.groupby(news_df[date_column].dt.date)[sentiment_column].agg(['mean', 'count', 'std'])
        daily_sentiment.columns = ['sentiment_avg', 'news_count', 'sentiment_std']

        # 计算情感变化率
        daily_sentiment['sentiment_change'] = daily_sentiment['sentiment_avg'].pct_change()

        return daily_sentiment

    def get_sentiment_trend(self, daily_sentiment, window=5):
        """计算情感趋势"""
        if len(daily_sentiment) < window:
            print(f"警告：数据点数量({len(daily_sentiment)})少于窗口大小({window})")
            window = len(daily_sentiment)

        # 计算移动平均
        daily_sentiment['sentiment_ma'] = daily_sentiment['sentiment_avg'].rolling(window=window).mean()

        # 计算情感趋势（1表示上升趋势，-1表示下降趋势，0表示盘整）
        daily_sentiment['sentiment_trend'] = 0
        daily_sentiment.loc[daily_sentiment['sentiment_ma'] > daily_sentiment['sentiment_ma'].shift(1), 'sentiment_trend'] = 1
        daily_sentiment.loc[daily_sentiment['sentiment_ma'] < daily_sentiment['sentiment_ma'].shift(1), 'sentiment_trend'] = -1

        return daily_sentiment