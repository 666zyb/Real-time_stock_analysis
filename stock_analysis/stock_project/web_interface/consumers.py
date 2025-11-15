import json
import asyncio
from channels.generic.websocket import AsyncWebsocketConsumer
from channels.db import database_sync_to_async
from .services.stock_service import StockDataService
from .services.news_service import NewsService
from .models import Stock, StockRealTimeData, News
import redis
from django.conf import settings
import logging

logger = logging.getLogger(__name__)


class StockConsumer(AsyncWebsocketConsumer):
    """股票数据WebSocket消费者"""

    async def connect(self):
        self.group_name = 'stock_updates'
        self.stock_service = StockDataService()

        # 加入组
        await self.channel_layer.group_add(
            self.group_name,
            self.channel_name
        )

        await self.accept()

        # 启动后台任务发送股票更新
        asyncio.create_task(self.send_stock_updates())

    async def disconnect(self, close_code):
        # 离开组
        await self.channel_layer.group_discard(
            self.group_name,
            self.channel_name
        )

    async def receive(self, text_data):
        """接收WebSocket消息"""
        try:
            data = json.loads(text_data)
            message_type = data.get('type')

            if message_type == 'request_stock_data':
                stock_code = data.get('stock_code')
                if stock_code:
                    # 从MySQL获取特定股票数据
                    stock_data = await self.stock_service.get_realtime_data_from_mysql(stock_code)
                    await self.send(text_data=json.dumps({
                        'type': 'stock_detail',
                        'stock_code': stock_code,
                        'data': stock_data
                    }))
        except Exception as e:
            print(f"处理WebSocket消息时出错: {str(e)}")

    @database_sync_to_async
    def get_latest_stock_data(self):
        """获取最新的股票数据"""
        stocks = Stock.objects.all()
        result = []

        for stock in stocks:
            try:
                latest = StockRealTimeData.objects.filter(stock=stock).order_by('-time').first()
                if latest:
                    # 计算涨跌和涨跌幅
                    change = latest.current_price - latest.last_close
                    change_percent = (change / latest.last_close) * 100

                    result.append({
                        'code': stock.code,
                        'name': stock.name,
                        'industry': stock.industry,
                        'current_price': latest.current_price,
                        'change': change,
                        'change_percent': change_percent,
                        'volume': latest.volume,
                        'amount': latest.amount,
                        'time': latest.time.strftime('%Y-%m-%d %H:%M:%S')
                    })
            except Exception as e:
                print(f"获取股票 {stock.code} 数据时出错: {str(e)}")

        return result

    @database_sync_to_async
    def get_stock_data(self, stock_code):
        """获取特定股票的详细数据"""
        try:
            stock = Stock.objects.get(code=stock_code)
            latest = StockRealTimeData.objects.filter(stock=stock).order_by('-time').first()

            if latest:
                # 计算涨跌和涨跌幅
                change = latest.current_price - latest.last_close
                change_percent = (change / latest.last_close) * 100

                return {
                    'code': stock.code,
                    'name': stock.name,
                    'industry': stock.industry,
                    'current_price': latest.current_price,
                    'open_price': latest.open_price,
                    'last_close': latest.last_close,
                    'high_price': latest.high_price,
                    'low_price': latest.low_price,
                    'change': change,
                    'change_percent': change_percent,
                    'volume': latest.volume,
                    'amount': latest.amount,
                    'time': latest.time.strftime('%Y-%m-%d %H:%M:%S')
                }

            return None
        except Exception as e:
            print(f"获取股票 {stock_code} 详细数据时出错: {str(e)}")
            return None

    async def send_stock_updates(self):
        """发送股票更新"""
        while True:
            try:
                # 直接从MySQL获取最新数据
                stock_data = await self.stock_service.get_realtime_data_from_mysql()

                # 发送到WebSocket
                await self.send(text_data=json.dumps({
                    'type': 'stock_update',
                    'data': stock_data
                }))

                # 每5秒更新一次
                await asyncio.sleep(5)
            except Exception as e:
                print(f"发送股票更新时出错: {str(e)}")
                await asyncio.sleep(5)

    async def send_stock_update(self, event):
        stock_data = event['data']
        await self.send(text_data=json.dumps({
            'type': 'stock_update',
            'data': stock_data
        }))


class NewsConsumer(AsyncWebsocketConsumer):
    """新闻数据WebSocket消费者"""

    async def connect(self):
        """WebSocket连接建立时调用"""
        self.group_name = 'news_updates'

        # 加入组
        await self.channel_layer.group_add(
            self.group_name,
            self.channel_name
        )

        # 接受WebSocket连接
        await self.accept()
        logger.info("WebSocket连接已建立")

        # 启动后台任务发送新闻更新
        self.background_task = asyncio.create_task(self.send_news_updates())

    async def disconnect(self, close_code):
        """WebSocket连接关闭时调用"""
        # 离开组
        await self.channel_layer.group_discard(
            self.group_name,
            self.channel_name
        )

        # 取消后台任务
        if hasattr(self, 'background_task'):
            self.background_task.cancel()

        logger.info(f"WebSocket连接已关闭，代码：{close_code}")

    async def get_news_from_redis(self):
        """从Redis获取新闻数据"""
        try:
            # 加载配置
            config = getattr(settings, 'REDIS_CONFIG', {
                'host': '127.0.0.1',
                'port': 6379,
                'db': 0,
                'password': None
            })

            # 连接到Redis
            redis_client = redis.Redis(
                host=config.get('host', '127.0.0.1'),
                port=config.get('port', 6379),
                db=config.get('db', 0),
                password=config.get('password'),
                decode_responses=True
            )

            # 获取新闻数据
            hot_news_key = "stock:hot_news"
            news_list = redis_client.lrange(hot_news_key, 0, 99)  # 获取前100条

            if not news_list:
                logger.warning(f"Redis中未找到新闻数据，键：{hot_news_key}")
                return []

            result = []
            for news_item in news_list:
                try:
                    news_data = json.loads(news_item)

                    # 记录原始数据
                    logger.info(f"从Redis获取到的新闻数据: {news_data}")

                    # 从内容中简单推断来源
                    source_name = 'Unknown'
                    content = news_data.get('content', '').lower()

                    if '新浪' in content or '微博' in content:
                        source_name = '新浪财经'
                    elif '同花顺' in content:
                        source_name = '同花顺'
                    elif '36氪' in content or '36kr' in content:
                        source_name = '36氪'
                    elif '财联社' in content:
                        source_name = '财联社'

                    # 构建结果对象，保留原始字段
                    news_item_obj = {
                        'content': news_data.get('content', ''),
                        'source': source_name,
                    }

                    # 添加日期字段
                    if 'datetime' in news_data:
                        news_item_obj['datetime'] = news_data['datetime']
                    elif 'pub_time' in news_data:
                        news_item_obj['pub_time'] = news_data['pub_time']

                    # 特别确保保留color字段
                    if 'color' in news_data:
                        news_item_obj['color'] = news_data['color']
                        logger.info(f"新闻有color字段: {news_data['color']}")

                    result.append(news_item_obj)
                except Exception as e:
                    logger.error(f"解析新闻数据时出错: {str(e)}, 数据: {news_item}")
                    continue

            # 打印最终结果的前几条
            if result:
                logger.info(f"处理后的新闻数据示例: {result[:2]}")

            logger.info(f"从Redis获取到 {len(result)} 条新闻")
            return result

        except Exception as e:
            logger.error(f"从Redis获取新闻时出错: {str(e)}")
            return []

    async def send_news_updates(self):
        """定期发送新闻更新"""
        while True:
            try:
                # 从Redis获取最新新闻
                news_data = await self.get_news_from_redis()

                # 发送到WebSocket
                if news_data:
                    await self.send(text_data=json.dumps({
                        'type': 'news_update',
                        'data': news_data
                    }))
                    logger.info(f"已通过WebSocket发送 {len(news_data)} 条新闻")

                # 每30秒更新一次
                await asyncio.sleep(30)

            except asyncio.CancelledError:
                # 任务被取消
                logger.info("新闻更新任务已取消")
                break
            except Exception as e:
                logger.error(f"发送新闻更新时出错: {str(e)}")
                await asyncio.sleep(30)  # 出错后等待30秒再重试