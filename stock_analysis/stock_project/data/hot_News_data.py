import redis
import json
from datetime import datetime, timedelta
import logging
from typing import List, Dict
import asyncio
from abc import ABC, abstractmethod
import hashlib


# 定义爬虫基类
class NewsSpider(ABC):
    @abstractmethod
    async def fetch_news(self) -> List[Dict]:
        """
        抓取新闻的抽象方法
        :return: 新闻列表，每条新闻包含content和datetime字段
        """
        pass


class HotNewsStorage:
    def __init__(self, host='172.16.0.4', port=6379, db=0, max_days=30):
        """
        初始化Redis连接
        :param host: Redis主机地址
        :param port: Redis端口
        :param db: Redis数据库编号
        :param max_days: 保留的最大天数
        """
        self.redis_client = redis.Redis(
            host=host,
            port=port,
            db=db,
            decode_responses=True
        )
        self.hot_news_key = "stock:hot_news"
        self.backup_key = "stock:hot_news_backup"  # 添加备份键
        self.spiders: List[NewsSpider] = []  # 存储爬虫实例
        self.max_days = max_days

        # 设置日志
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def register_spider(self, spider: NewsSpider) -> None:
        """
        注册爬虫
        :param spider: 爬虫实例
        """
        self.spiders.append(spider)
        self.logger.info(f"注册爬虫: {spider.__class__.__name__}")

    async def fetch_all_news(self) -> None:
        """从所有注册的爬虫获取新闻"""
        try:
            all_news = []

            # 并发运行所有爬虫
            tasks = [spider.fetch_news() for spider in self.spiders]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # 处理每个爬虫的结果
            for spider, result in zip(self.spiders, results):
                if isinstance(result, Exception):
                    self.logger.error(f"爬虫 {spider.__class__.__name__} 运行出错: {str(result)}")
                    continue

                self.logger.info(f"爬虫 {spider.__class__.__name__} 获取到 {len(result)} 条新闻")
                all_news.extend(result)

            # 存储所有获取到的新闻
            if all_news:
                # 存储新闻会进行去重处理
                stored_news = await self.store_news(all_news)

                # 检查数据一致性并从备份恢复缺失数据
                main_count = await self.get_news_count()
                backup_count = await self.get_backup_count()

                if main_count < backup_count:
                    self.logger.warning(f"主数据({main_count})少于备份({backup_count})，正在从备份补充缺失数据...")
                    await self.restore_missing_from_backup()

                # 只备份去重后的新数据
                if stored_news:
                    await self.backup_new_data(stored_news)

        except Exception as e:
            self.logger.error(f"获取新闻数据时出错: {str(e)}")

    def _filter_old_news(self, news_list: List[Dict]) -> List[Dict]:
        """
        过滤掉超过指定天数的新闻
        """
        current_time = datetime.now()
        cutoff_time = current_time - timedelta(days=self.max_days)

        return [
            news for news in news_list
            if datetime.strptime(news['datetime'], '%Y-%m-%d %H:%M:%S') > cutoff_time
        ]

    async def store_news(self, news_list: List[Dict]) -> List[Dict]:
        """存储新闻数据到Redis，并确保只保留一个月内的数据"""
        try:
            # 获取现有的新闻数据
            existing_news = await self.get_all_news()

            # 合并新旧数据
            all_news = existing_news + news_list

            # 先去重，保留id字段
            unique_news = self._deduplicate_news(all_news)

            # 过滤掉旧数据
            filtered_news = self._filter_old_news(unique_news)

            # 按时间排序
            sorted_news = sorted(
                filtered_news,
                key=lambda x: datetime.strptime(x['datetime'], '%Y-%m-%d %H:%M:%S'),
                reverse=True
            )

            # 找出新添加的新闻（存在于sorted_news但不存在于existing_news中的新闻）
            existing_contents = {news['content'] for news in existing_news}
            new_added_news = [news for news in sorted_news if news['content'] not in existing_contents]

            # 使用管道批量写入以提高性能
            pipe = self.redis_client.pipeline()
            pipe.delete(self.hot_news_key)
            if sorted_news:
                pipe.rpush(self.hot_news_key, *[json.dumps(news, ensure_ascii=False) for news in sorted_news])
            pipe.execute()

            self.logger.info(
                f"成功存储 {len(sorted_news)} 条新闻，"
                f"新增 {len(new_added_news)} 条，"
                f"总新闻数 {len(sorted_news)} 条"
            )

            # 为每条新闻发布消息，触发实时分析
            for i, news in enumerate(new_added_news):
                try:
                    # 为新闻生成哈希值
                    content = news.get('content', '')
                    datetime_str = news.get('datetime', '')
                    hash_str = f"{content}|{datetime_str}"
                    news_hash = hashlib.md5(hash_str.encode('utf-8')).hexdigest()

                    # 查找新闻在列表中的索引
                    for idx, item in enumerate(sorted_news):
                        if item['content'] == news['content'] and item['datetime'] == news['datetime']:
                            # 发布新闻添加事件
                            event_data = {
                                'hash': news_hash,
                                'index': idx,
                                'timestamp': datetime.now().timestamp()
                            }
                            self.redis_client.publish('stock:hot_news:add', json.dumps(event_data))
                            self.logger.info(f"已发布新闻添加事件: {news_hash[:10]}...")
                            break
                except Exception as e:
                    self.logger.error(f"发布新闻事件时出错: {str(e)}")

            # 返回新添加的新闻列表，用于实时分析
            return new_added_news

        except Exception as e:
            self.logger.error(f"存储新闻数据时出错: {str(e)}")
            return []

    async def get_all_news(self) -> List[Dict]:
        """获取所有新闻"""
        try:
            # 使用管道批量读取以提高性能
            pipe = self.redis_client.pipeline()
            pipe.lrange(self.hot_news_key, 0, -1)
            result = pipe.execute()

            if result and result[0]:
                return [json.loads(item) for item in result[0]]
            return []

        except Exception as e:
            self.logger.error(f"获取新闻数据时出错: {str(e)}")
            return []

    def _deduplicate_news(self, news_list: List[Dict]) -> List[Dict]:
        """修复版：超快速内容去重，不使用对象属性"""
        if not news_list:
            return []

        # 按时间从旧到新排序（保留最早发布的新闻）
        sorted_news = sorted(
            news_list,
            key=lambda x: datetime.strptime(x['datetime'], '%Y-%m-%d %H:%M:%S'),
            reverse=False
        )

        # 创建一个索引表，记录每个新闻的特征
        news_index = []

        # 记录已添加的新闻内容的MD5哈希，用于精确去重
        content_hashes = set()

        # 为每条新闻提前计算特征，避免重复计算
        news_features = {}

        unique_news = []

        for news in sorted_news:
            content = news.get('content', '')

            # 1. 完全匹配去重 - 使用MD5哈希
            content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
            if content_hash in content_hashes:
                continue

            # 2. 快速特征比较 - 只在10分钟窗口内进行
            current_time = datetime.strptime(news['datetime'], '%Y-%m-%d %H:%M:%S')
            is_similar = False

            # 计算当前新闻的特征
            if content_hash not in news_features:
                # 使用前50个字符作为特征集
                news_features[content_hash] = set(content[:50])

            current_features = news_features[content_hash]

            # 只检查最近的新闻（最多20条）
            for idx in range(len(news_index) - 1, max(len(news_index) - 20, -1), -1):
                existing_news, existing_time, existing_feature_hash = news_index[idx]

                # 如果时间差超过10分钟，跳出循环
                time_diff = (current_time - existing_time).total_seconds() / 60
                if time_diff > 10:
                    break

                # 快速检查 - 内容长度相差太大的一定不相似
                if abs(len(content) - len(existing_news['content'])) > len(content) * 0.3:
                    continue

                # 获取已存储的特征
                existing_features = news_features[existing_feature_hash]

                # 快速特征比较 - 使用前50个字符的交集占比
                intersection = len(current_features.intersection(existing_features))
                if intersection > len(current_features) * 0.7:  # 如果70%以上字符相同，认为相似
                    is_similar = True
                    break

            # 如果不相似，添加到结果中
            if not is_similar:
                # 添加到唯一新闻列表
                unique_news.append(news)
                # 记录哈希值
                content_hashes.add(content_hash)
                # 添加到索引
                news_index.append((news, current_time, content_hash))

        self.logger.info(f"超快速去重: 原有 {len(news_list)} 条新闻，去重后 {len(unique_news)} 条")
        return unique_news

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """计算两段文本的Jaccard相似度"""
        try:
            # 如果两个文本完全相同，相似度为1
            if text1 == text2:
                return 1.0

            # 分词（简单按字符切分，使用字符集合计算相似度）
            # 对于中文，这种方法比较有效，因为中文的字符本身就有较强的语义
            # 计算共有字符占比
            chars1 = set(text1)
            chars2 = set(text2)

            # 计算Jaccard相似度: 交集大小 / 并集大小
            intersection = len(chars1.intersection(chars2))
            union = len(chars1.union(chars2))

            if union == 0:
                return 0

            return intersection / union
        except Exception as e:
            self.logger.error(f"计算文本相似度时出错: {str(e)}")
            return 0

    async def clean_old_news(self) -> None:
        """
        清理旧数据的独立方法
        """
        try:
            news_list = await self.get_all_news()
            if not news_list:
                return

            filtered_news = self._filter_old_news(news_list)

            if len(filtered_news) < len(news_list):
                pipe = self.redis_client.pipeline()
                pipe.delete(self.hot_news_key)
                if filtered_news:
                    pipe.rpush(self.hot_news_key, *[
                        json.dumps(news, ensure_ascii=False)
                        for news in filtered_news
                    ])
                pipe.execute()

                self.logger.info(f"清理了 {len(news_list) - len(filtered_news)} 条旧新闻")

        except Exception as e:
            self.logger.error(f"清理旧新闻时出错: {str(e)}")

    async def backup_new_data(self, processed_news: List[Dict] = None) -> None:
        """将去重后的新数据备份到Redis"""
        try:
            # 如果没有提供已处理的新闻列表，则获取所有主数据新闻
            if processed_news is None:
                main_news = await self.get_all_news()
                if not main_news:
                    self.logger.warning("没有新闻数据可备份")
                    return
            else:
                main_news = processed_news

            # 获取所有备份新闻
            backup_news = await self.get_backup_news()

            # 创建内容哈希集合，用于快速检查新闻是否已存在
            backup_content_set = {news['content'] for news in backup_news}

            # 找出需要备份的新数据
            new_news = [news for news in main_news if news['content'] not in backup_content_set]

            if not new_news:
                self.logger.info("没有新数据需要备份")
                return

            # 使用管道批量添加新数据到备份
            pipe = self.redis_client.pipeline()
            pipe.rpush(self.backup_key, *[json.dumps(news, ensure_ascii=False) for news in new_news])
            pipe.execute()

            self.logger.info(f"已将 {len(new_news)} 条去重后的热点新闻备份到Redis")
        except Exception as e:
            self.logger.error(f"备份热点新闻到Redis时出错: {str(e)}")

    async def get_backup_news(self) -> List[Dict]:
        """获取备份中的所有新闻"""
        try:
            # 使用管道批量读取以提高性能
            pipe = self.redis_client.pipeline()
            pipe.lrange(self.backup_key, 0, -1)
            result = pipe.execute()

            if result and result[0]:
                return [json.loads(item) for item in result[0]]
            return []

        except Exception as e:
            self.logger.error(f"获取备份新闻数据时出错: {str(e)}")
            return []

    async def restore_missing_from_backup(self) -> None:
        """从备份中恢复缺失的热点新闻"""
        try:
            # 获取主数据和备份数据
            main_news = await self.get_all_news()
            backup_news = await self.get_backup_news()

            if not backup_news:
                self.logger.warning("没有备份数据可用于恢复")
                return

            # 创建内容哈希集合，用于快速检查
            main_content_set = {news['content'] for news in main_news}

            # 找出缺失的数据（在备份中存在但在主数据中不存在）
            missing_news = [news for news in backup_news if news['content'] not in main_content_set]

            if not missing_news:
                self.logger.info("没有数据需要从备份恢复")
                return

            # 合并现有数据和缺失数据
            all_news = main_news + missing_news

            # 先去重
            unique_news = self._deduplicate_news(all_news)

            # 过滤掉旧数据
            filtered_news = self._filter_old_news(unique_news)

            # 按时间排序
            sorted_news = sorted(
                filtered_news,
                key=lambda x: datetime.strptime(x['datetime'], '%Y-%m-%d %H:%M:%S'),
                reverse=True
            )

            # 更新主数据
            pipe = self.redis_client.pipeline()
            pipe.delete(self.hot_news_key)
            if sorted_news:
                pipe.rpush(self.hot_news_key, *[json.dumps(news, ensure_ascii=False) for news in sorted_news])
            pipe.execute()

            self.logger.info(f"已从备份恢复 {len(missing_news)} 条缺失的热点新闻")
        except Exception as e:
            self.logger.error(f"从备份恢复缺失热点新闻时出错: {str(e)}")

    async def get_news_count(self) -> int:
        """获取当前热点新闻数量"""
        return self.redis_client.llen(self.hot_news_key)

    async def get_backup_count(self) -> int:
        """获取备份的热点新闻数量"""
        return self.redis_client.llen(self.backup_key)

    async def export_to_json(self, filename: str = None) -> None:
        """导出热点新闻到JSON文件"""
        try:
            news_data = await self.get_all_news()
            if not news_data:
                self.logger.warning("没有新闻数据可导出")
                return

            if filename is None:
                timestamp = int(datetime.now().timestamp())
                filename = f'hot_news_{timestamp}.json'

            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(news_data, f, ensure_ascii=False, indent=2)

            self.logger.info(f"已将 {len(news_data)} 条热点新闻导出到 {filename}")
        except Exception as e:
            self.logger.error(f"导出热点新闻到JSON文件时出错: {str(e)}")

    async def start_periodic_fetch(self, interval_minutes: int = 1):
        """定期获取新闻并清理旧数据"""
        while True:
            try:
                await self.fetch_all_news()
                await self.clean_old_news()  # 每次更新后清理旧数据
                await asyncio.sleep(interval_minutes * 10)
            except Exception as e:
                self.logger.error(f"定期获取新闻时出错: {str(e)}")
                await asyncio.sleep(60)


# 使用示例
async def main():
    # 创建存储实例
    storage = HotNewsStorage()

    # 注册爬虫
    from stock_analysis.News_crawler.kr_36氪 import Kr36Spider
    from stock_analysis.News_crawler.新浪财经 import SinaSpider
    from stock_analysis.News_crawler.同花顺 import ThsSpider
    from stock_analysis.News_crawler.财联社 import ClsSpider

    # 注册爬虫实例
    storage.register_spider(Kr36Spider())
    storage.register_spider(SinaSpider())
    storage.register_spider(ThsSpider())
    storage.register_spider(ClsSpider())

    # 启动定期获取
    await storage.start_periodic_fetch(interval_minutes=1)


if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # 运行
    asyncio.run(main())