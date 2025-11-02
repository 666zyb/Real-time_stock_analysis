import os
import json
import redis
from django.conf import settings
from datetime import datetime

# 导入现有的新闻爬虫

from web_interface.models import NewsSource, News


class NewsService:
    def __init__(self):
        # 加载配置
        with open(os.path.join(settings.BASE_DIR, 'config', 'config.json'), 'r', encoding='utf-8') as f:
            self.config = json.load(f)

        # 连接到Redis
        redis_config = self.config.get('redis_config', {})
        self.redis_client = redis.Redis(
            host=redis_config.get('host', '127.0.0.1'),
            port=redis_config.get('port', 6379),
            db=redis_config.get('db', 0),
            password=redis_config.get('password'),
            decode_responses=True
        )

        # 新闻数据的Redis键
        self.hot_news_key = "stock:hot_news"

        # 确保新闻来源存在
        self.ensure_sources()

    def ensure_sources(self):
        """确保新闻来源在数据库中存在"""
        sources = ['新浪财经', '同花顺', '36氪', '财联社']
        for source in sources:
            NewsSource.objects.get_or_create(name=source)

    async def get_news_from_redis(self):
        """从Redis获取新闻数据"""
        try:
            # 从Redis获取所有新闻
            news_list = self.redis_client.lrange(self.hot_news_key, 0, -1)
            result = []

            for news_item in news_list:
                news_data = json.loads(news_item)

                # 尝试从内容中推断来源
                source_name = self._infer_source(news_data)

                # 获取来源对象
                source, _ = NewsSource.objects.get_or_create(name=source_name)

                # 格式化结果
                news_obj = {
                    'source': source.name,
                    'content': news_data['content'],
                    'pub_time': news_data['datetime']
                }

                # 保留原有id字段
                if 'id' in news_data:
                    news_obj['id'] = news_data['id']
                    # 同时添加大写Id字段
                    news_obj['Id'] = news_data['id']
                elif 'Id' in news_data:
                    news_obj['Id'] = news_data['Id']

                result.append(news_obj)

            return result
        except Exception as e:
            print(f"从Redis获取新闻时出错: {str(e)}")
            return []

    def _infer_source(self, news_data):
        """从新闻内容推断来源"""
        # 尝试从新闻数据中推断来源
        # 这只是一个简单的实现，实际项目中可能需要更复杂的逻辑
        content = news_data['content'].lower()

        if '新浪' in content or '微博' in content:
            return '新浪财经'
        elif '同花顺' in content:
            return '同花顺'
        elif '36氪' in content or '36kr' in content:
            return '36氪'
        elif '财联社' in content:
            return '财联社'
        else:
            return '其他来源'  # 默认来源

    async def update_news_from_redis(self):
        """更新Redis中的新闻到数据库"""
        news_list = await self.get_news_from_redis()

        for news_item in news_list:
            try:
                source = NewsSource.objects.get(name=news_item['source'])

                # 检查是否存在相同内容的新闻
                if not News.objects.filter(content=news_item['content']).exists():
                    News.objects.create(
                        source=source,
                        content=news_item['content'],
                        pub_time=datetime.strptime(news_item['pub_time'], "%Y-%m-%d %H:%M:%S")
                    )
            except Exception as e:
                print(f"更新新闻到数据库时出错: {str(e)}")

        return news_list

    async def fetch_news(self, source_name=None):
        """获取新闻"""
        all_news = []

        for spider_info in self.spiders:
            if source_name and spider_info['name'] != source_name:
                continue

            try:
                # 获取新闻来源
                source = NewsSource.objects.get(name=spider_info['name'])

                # 获取新闻数据
                news_list = await spider_info['instance'].fetch_news()

                # 保存新闻数据
                for news_item in news_list:
                    content = news_item['content']

                    # 转换时间格式
                    try:
                        pub_time = datetime.strptime(news_item['datetime'], "%Y-%m-%d %H:%M:%S")
                    except:
                        pub_time = datetime.now()

                    # 检查是否存在相同内容的新闻
                    if not News.objects.filter(content=content, source=source).exists():
                        News.objects.create(
                            source=source,
                            content=content,
                            pub_time=pub_time
                        )

                        all_news.append({
                            'source': source.name,
                            'content': content,
                            'pub_time': pub_time.strftime("%Y-%m-%d %H:%M:%S")
                        })

            except Exception as e:
                print(f"获取 {spider_info['name']} 新闻数据时出错: {str(e)}")

        return all_news

    async def update_all_news(self):
        """更新所有新闻"""
        return await self.fetch_news()