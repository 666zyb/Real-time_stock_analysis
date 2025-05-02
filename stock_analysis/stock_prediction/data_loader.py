import json
import pandas as pd
import numpy as np
import redis
import pymysql
from datetime import datetime, timedelta
import re


class DataLoader:
    def __init__(self, config_path='config.json'):
        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)

        # MySQL配置
        self.mysql_config = self.config['mysql_config']

        # Redis配置
        self.redis_config = self.config['redis_config']
        self.redis_client = redis.Redis(
            host=self.redis_config['host'],
            port=self.redis_config['port'],
            db=self.redis_config['db'],
            password=self.redis_config['password']
        )

        # 股票列表
        self.stocks = self.config['stocks']

    def get_mysql_connection(self):
        """创建MySQL连接"""
        conn = pymysql.connect(
            host=self.mysql_config['host'],
            port=self.mysql_config['port'],
            user=self.mysql_config['user'],
            password=self.mysql_config['password'],
            database=self.mysql_config['database'],
            charset='utf8mb4'
        )
        return conn

    def get_daily_data(self, stock_code, stock_name=None, start_date=None, end_date=None, days=365):
        """获取股票的每日行情数据

        根据您的结构，每日行情数据存储在[股票名字]_history表中，使用'日期'列
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')

        if start_date is None:
            start_date = (datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=days)).strftime('%Y-%m-%d')
            # 确保结束日期不超过今天
            now = datetime.now().strftime('%Y-%m-%d')
            if end_date > now:
                end_date = now
                print(f"警告: 结束日期调整为今天 {now}")

        # 如果没有提供股票名称，则从配置中查找
        if stock_name is None:
            for stock in self.stocks:
                if stock['code'] == stock_code:
                    stock_name = stock['name']
                    break

            if stock_name is None:
                print(f"找不到股票代码 {stock_code} 对应的股票名称")
                return None

        # 构建表名
        table_name = f"{stock_name}_history"

        # 使用'日期'列作为日期字段
        date_column = '日期'

        # 构建SQL查询
        query = f"""
            SELECT * FROM `{table_name}` 
            WHERE `{date_column}` BETWEEN '{start_date}' AND '{end_date}'
            ORDER BY `{date_column}` ASC
        """

        try:
            conn = self.get_mysql_connection()
            df = pd.read_sql(query, conn)
            conn.close()

            if df.empty:
                print(f"股票 {stock_name}({stock_code}) 在指定日期范围内没有数据。")
                return None

            # 添加股票代码和名称列
            df['stock_code'] = stock_code
            df['stock_name'] = stock_name

            # 将'日期'列转换为datetime类型
            df[date_column] = pd.to_datetime(df[date_column])

            # 添加标准化的trade_date列(如果后续代码需要这个列名)
            df['trade_date'] = df[date_column]

            return df
        except Exception as e:
            print(f"数据库查询出错: {e}")

            # 尝试获取表的所有列
            try:
                conn = self.get_mysql_connection()
                cursor = conn.cursor()
                cursor.execute(f"SELECT * FROM `{table_name}` LIMIT 1")
                column_names = [desc[0] for desc in cursor.description]
                cursor.close()
                conn.close()
                print(f"表 {table_name} 的列名: {', '.join(column_names)}")
            except Exception as e2:
                print(f"无法获取表结构: {e2}")

            return None

    def execute_query(self, query, params=None):
        """执行SQL查询并返回DataFrame"""
        try:
            conn = self.get_mysql_connection()
            df = pd.read_sql(query, conn, params=params)
            conn.close()
            return df
        except Exception as e:
            print(f"数据库查询出错: {e}")
            return pd.DataFrame()

    def execute_update(self, query, params=None):
        """执行SQL更新操作"""
        try:
            conn = self.get_mysql_connection()
            cursor = conn.cursor()
            cursor.execute(query, params)
            conn.commit()
            affected_rows = cursor.rowcount
            cursor.close()
            conn.close()
            return affected_rows
        except Exception as e:
            print(f"数据库更新出错: {e}")
            return 0

    def get_realtime_data(self, stock_code):
        """获取股票的实时数据"""
        # 检查股票代码是否已包含前缀（如sz）
        if not re.match(r'^[a-z]{2}\d+$', stock_code):
            # 根据交易所规则为股票添加前缀
            if stock_code.startswith('6'):  # 上海交易所
                prefixed_code = f"sh{stock_code}"
            elif stock_code.startswith(('0', '3')):  # 深圳交易所
                prefixed_code = f"sz{stock_code}"
            else:
                print(f"无法确定股票 {stock_code} 的交易所前缀")
                return None
        else:
            prefixed_code = stock_code

        # 构建表名
        table_name = f"stock_{prefixed_code}_realtime"

        try:
            conn = self.get_mysql_connection()

            # 使用"时间"字段排序获取最新数据
            query = f"""
                SELECT * FROM `{table_name}` 
                ORDER BY `时间` DESC LIMIT 1
            """

            df = pd.read_sql(query, conn)
            conn.close()

            if df.empty:
                print(f"股票 {prefixed_code} 没有实时数据。")
                return None

            # 转换为字典
            realtime_data = df.iloc[0].to_dict()

            # 添加处理后的字段
            if '当前价格' in realtime_data:
                realtime_data['current_price'] = float(realtime_data['当前价格'])

            if '收盘价' not in realtime_data and '昨日收盘价' in realtime_data:
                realtime_data['收盘价'] = float(realtime_data['昨日收盘价'])

            return realtime_data
        except Exception as e:
            print(f"获取实时数据出错: {e}")

            # 尝试打印表结构以便调试
            try:
                conn = self.get_mysql_connection()
                cursor = conn.cursor()
                cursor.execute(f"SHOW COLUMNS FROM `{table_name}`")
                columns = cursor.fetchall()
                column_names = [col[0] for col in columns]
                cursor.close()
                conn.close()
                print(f"实时数据表 {table_name} 的列: {', '.join(column_names)}")
            except Exception as e2:
                print(f"无法获取实时数据表结构: {e2}")

            return None

    def get_news_sentiment(self, stock_code=None, days=7):
        """获取新闻情感数据"""
        try:
            # 检查Redis中键的类型
            key_exists = self.redis_client.exists('stock:hot_news')
            if not key_exists:
                print("Redis中不存在'stock:hot_news'键")
                return self._create_default_sentiment(days)

            key_type = self.redis_client.type('stock:hot_news').decode('utf-8')
            print(f"Redis键'stock:hot_news'的类型: {key_type}")

            # 读取已分析过的新闻缓存
            sentiment_cache_key = 'stock:sentiment_cache'
            sentiment_cache = {}
            if self.redis_client.exists(sentiment_cache_key):
                cache_data = self.redis_client.hgetall(sentiment_cache_key)
                for news_id, sentiment_value in cache_data.items():
                    sentiment_cache[news_id.decode('utf-8')] = float(sentiment_value.decode('utf-8'))

            # 根据键类型选择不同的读取方式
            news_data = []

            if key_type == 'string':
                hot_news = self.redis_client.get('stock:hot_news')
                if hot_news:
                    try:
                        news_item = json.loads(hot_news.decode('utf-8'))
                        news_data.append(news_item)
                    except:
                        print(f"无法解析字符串类型的热点新闻")
            elif key_type == 'list':
                # 获取列表中的所有元素
                all_items = self.redis_client.lrange('stock:hot_news', 0, -1)
                for item in all_items:
                    try:
                        news_item = json.loads(item.decode('utf-8'))
                        news_data.append(news_item)
                    except:
                        print(f"无法解析列表项")
            elif key_type == 'hash':
                # 获取哈希表中的所有字段和值
                all_fields = self.redis_client.hgetall('stock:hot_news')
                for field, value in all_fields.items():
                    try:
                        field_str = field.decode('utf-8')
                        value_str = value.decode('utf-8')

                        # 尝试解析JSON
                        try:
                            news_item = json.loads(value_str)
                        except:
                            # 如果不是JSON，创建简单对象
                            news_item = {
                                'content': value_str,
                                'datetime': field_str if 'date' in field_str or 'time' in field_str else datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            }

                        news_data.append(news_item)
                    except:
                        print(f"无法解析哈希项")
            else:
                print(f"不支持的Redis键类型: {key_type}")
                return self._create_default_sentiment(days)

            # 如果没有获取到任何数据
            if not news_data:
                print("未能获取任何新闻数据")
                return self._create_default_sentiment(days)

            print(f"成功获取{len(news_data)}条新闻数据")

            # 标准化新闻数据的格式
            standardized_news = []
            for item in news_data:
                news_item = {}

                # 处理内容
                if 'content' in item:
                    news_item['content'] = item['content']
                else:
                    # 尝试找到可能包含内容的字段
                    for key in ['text', 'message', 'body', '内容']:
                        if key in item:
                            news_item['content'] = item[key]
                            break
                    else:
                        # 如果找不到内容字段，使用整个对象的字符串形式
                        news_item['content'] = str(item)

                # 处理日期时间
                if 'datetime' in item:
                    news_item['date'] = pd.to_datetime(item['datetime']).date()
                elif 'date' in item:
                    news_item['date'] = pd.to_datetime(item['date']).date()
                else:
                    # 尝试找到可能包含日期的字段
                    for key in ['time', 'timestamp', 'created_at', 'publish_time', '日期', '时间']:
                        if key in item:
                            try:
                                news_item['date'] = pd.to_datetime(item[key]).date()
                                break
                            except:
                                pass
                    else:
                        # 如果找不到日期字段，使用当前日期
                        news_item['date'] = datetime.now().date()

                # 为每条新闻生成唯一ID (使用内容的哈希值)
                news_content = news_item.get('content', '')
                if isinstance(news_content, str):
                    import hashlib
                    news_id = hashlib.md5(news_content.encode('utf-8')).hexdigest()
                    news_item['news_id'] = news_id
                else:
                    news_item['news_id'] = None

                standardized_news.append(news_item)

            # 转换为DataFrame
            news_df = pd.DataFrame(standardized_news)

            # 只对未分析过的新闻进行情感分析
            from utils.sentiment_analysis import SentimentAnalyzer
            sentiment_analyzer = SentimentAnalyzer()

            new_sentiments = {}

            def get_sentiment(row):
                news_id = row['news_id']
                content = row['content']

                # 如果已经在缓存中，直接使用缓存的结果
                if news_id and news_id in sentiment_cache:
                    return sentiment_cache[news_id]

                # 否则进行新的分析
                if isinstance(content, str):
                    sentiment = sentiment_analyzer.analyze_news_sentiment(content)
                    # 保存到临时字典，稍后一次性更新Redis
                    if news_id:
                        new_sentiments[news_id] = sentiment
                    return sentiment
                return 0

            news_df['sentiment'] = news_df.apply(get_sentiment, axis=1)

            # 将新分析的情感结果更新到Redis缓存
            if new_sentiments:
                self.redis_client.hmset(sentiment_cache_key, {k: str(v) for k, v in new_sentiments.items()})
                print(f"已将{len(new_sentiments)}条新闻的情感分析结果添加到缓存")

            # 按日期聚合情感得分
            news_df['date'] = pd.to_datetime(news_df['date'])
            daily_sentiment = news_df.groupby(news_df['date'].dt.date)['sentiment'].mean().reset_index()

            # 确保每一天都有数据
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=days - 1)
            all_dates = pd.DataFrame({
                'date': [start_date + timedelta(days=i) for i in range(days)]
            })

            daily_sentiment = pd.merge(all_dates, daily_sentiment, on='date', how='left')
            daily_sentiment['sentiment'] = daily_sentiment['sentiment'].fillna(0)

            # 设置日期为索引
            daily_sentiment.set_index('date', inplace=True)

            return daily_sentiment

        except Exception as e:
            print(f"获取新闻情感数据出错: {e}")
            import traceback
            traceback.print_exc()
            return self._create_default_sentiment(days)

    def _create_default_sentiment(self, days):
        """创建默认的情感数据框"""
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days - 1)

        dates = [start_date + timedelta(days=i) for i in range(days)]
        empty_sentiment = pd.DataFrame({
            'date': dates,
            'sentiment': [0.0] * len(dates)
        }).set_index('date')

        return empty_sentiment

    def get_all_stocks(self):
        """返回配置中所有股票代码"""
        return [stock['code'] for stock in self.stocks] 