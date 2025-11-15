import json
import redis
import logging
import os
from datetime import datetime
import time

# 设置日志 - 只输出到控制台，不生成日志文件
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class NewsAnalysisReceiver:
    """
    新闻分析接收器
    从Redis获取news_stock_analysis.py的分析结果
    """

    def __init__(self, config_path='config/config.json'):
        """初始化接收器"""
        self.config_path = config_path
        self.load_config(config_path)

        # 连接Redis
        self.redis_client = redis.Redis(
            host=self.redis_config.get('host', '172.16.0.4'),
            port=self.redis_config.get('port', 6379),
            db=self.redis_config.get('db', 0),
            password=self.redis_config.get('password', None),
            decode_responses=True
        )

        # 初始化已处理的新闻哈希集合
        self.processed_news_hashes = set()

        # 最大保留的处理过的新闻哈希数量
        self.max_processed_hashes = 2000

        # 跟踪已添加到other_stocks的股票代码，避免重复
        self.added_stock_codes = set()
        # 从配置文件加载已有的other_stocks
        self.load_other_stocks()

        # 待重试新闻哈希列表及重试次数记录
        self.retry_news = {}
        self.max_retry_times = 10

    def load_config(self, config_path: str):
        """加载配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)

            # 获取Redis配置
            self.redis_config = self.config.get('redis_config', {})

            # 新闻分析配置
            self.news_keys = {
                'all_analyses': 'stock:news_all_analyses',
                'hot_news': 'stock:hot_news',
                'processed_hashes': 'stock:news_indicators_processed_hashes',
                'processed_hashes_order': 'stock:news_indicators_processed_hashes_order',  # 用于记录处理顺序
                'retry_news': 'stock:news_indicators_retry_news'  # 用于存储待重试的新闻
            }

            logger.info("配置文件加载成功")
        except Exception as e:
            logger.error(f"加载配置文件出错: {str(e)}")
            raise

    def load_other_stocks(self):
        """加载other_stocks中的股票代码，避免重复添加"""
        try:
            other_stocks = self.config.get('other_stocks', [])
            for stock in other_stocks:
                self.added_stock_codes.add(stock['code'])
            logger.info(f"已从配置加载 {len(self.added_stock_codes)} 个other_stocks股票代码")
        except Exception as e:
            logger.error(f"加载other_stocks股票代码时出错: {str(e)}")

    def save_strong_impact_stocks(self, strong_stocks):
        """将强影响股票保存到config.json的other_stocks数组中"""
        try:
            # 重新加载最新的配置，避免覆盖其他进程的修改
            with open(self.config_path, 'r', encoding='utf-8') as f:
                current_config = json.load(f)

            # 获取当前的other_stocks数组
            other_stocks = current_config.get('other_stocks', [])

            # 标记是否有新股票添加
            has_new_stocks = False

            # 将新的强影响股票添加到other_stocks
            for stock in strong_stocks:
                code = stock.get('code', '')
                name = stock.get('name', '')

                # 跳过已在stocks中的股票
                if any(s['code'] == code for s in current_config.get('stocks', [])):
                    logger.info(f"股票 {name}({code}) 已在主要股票列表中，跳过添加")
                    continue

                # 跳过已添加的股票，避免重复
                if code in self.added_stock_codes:
                    continue

                # 添加新股票并记录
                other_stocks.append({"code": code, "name": name})
                self.added_stock_codes.add(code)
                has_new_stocks = True
                logger.info(f"添加新的强影响股票到other_stocks: {name}({code})")

            # 如果有新股票，更新配置文件
            if has_new_stocks:
                current_config['other_stocks'] = other_stocks
                with open(self.config_path, 'w', encoding='utf-8') as f:
                    json.dump(current_config, f, ensure_ascii=False, indent=2)
                logger.info(f"已将强影响股票添加到配置文件，当前other_stocks中有 {len(other_stocks)} 只股票")

            return has_new_stocks
        except Exception as e:
            logger.error(f"保存强影响股票到配置文件时出错: {str(e)}")
            return False

    def get_processed_news_hashes(self):
        """获取已处理的新闻哈希集合"""
        try:
            processed_hashes = self.redis_client.smembers(self.news_keys['processed_hashes'])
            return set(processed_hashes)
        except Exception as e:
            logger.error(f"获取已处理新闻哈希时出错: {str(e)}")
            return set()

    def load_retry_news(self):
        """从Redis加载待重试的新闻及其重试次数"""
        try:
            retry_data = self.redis_client.hgetall(self.news_keys['retry_news'])
            if retry_data:
                # 将字符串转换为整数
                self.retry_news = {hash_key: int(count) for hash_key, count in retry_data.items()}
                logger.info(f"已加载 {len(self.retry_news)} 条待重试新闻")
            else:
                self.retry_news = {}
        except Exception as e:
            logger.error(f"加载待重试新闻时出错: {str(e)}")
            self.retry_news = {}

    def save_retry_news(self):
        """保存待重试新闻到Redis"""
        try:
            # 先清除现有数据
            self.redis_client.delete(self.news_keys['retry_news'])

            # 如果有待重试的新闻，保存到Redis
            if self.retry_news:
                # 将整数转换为字符串
                retry_data = {hash_key: str(count) for hash_key, count in self.retry_news.items()}
                self.redis_client.hmset(self.news_keys['retry_news'], retry_data)
                logger.info(f"已保存 {len(self.retry_news)} 条待重试新闻到Redis")
        except Exception as e:
            logger.error(f"保存待重试新闻时出错: {str(e)}")

    def add_news_to_retry(self, news_hash, news_data):
        """添加新闻到重试队列"""
        # 初始化重试次数为1
        if news_hash not in self.retry_news:
            self.retry_news[news_hash] = 1
            logger.info(f"新闻 {news_hash[:8]}... 加入重试队列，首次重试")
        else:
            # 增加重试次数
            self.retry_news[news_hash] += 1
            logger.info(f"新闻 {news_hash[:8]}... 重试次数增加到 {self.retry_news[news_hash]}")

        # 将重试数据保存到Redis
        self.save_retry_news()

    def clean_old_processed_hashes(self):
        """当处理过的哈希数量超过限制时，删除最早的哈希"""
        try:
            # 获取当前处理过的哈希数量
            current_count = self.redis_client.scard(self.news_keys['processed_hashes'])

            # 如果超过最大限制，删除最早的哈希
            if current_count > self.max_processed_hashes:
                # 计算需要删除的数量
                to_remove_count = current_count - self.max_processed_hashes
                logger.info(f"处理过的新闻哈希数量({current_count})超过限制({self.max_processed_hashes})，将删除最早的{to_remove_count}条")

                # 从有序列表中获取最早的哈希
                oldest_hashes = self.redis_client.lrange(self.news_keys['processed_hashes_order'], 0, to_remove_count - 1)

                if oldest_hashes:
                    # 从集合中删除这些哈希
                    self.redis_client.srem(self.news_keys['processed_hashes'], *oldest_hashes)
                    # 从有序列表中删除
                    self.redis_client.ltrim(self.news_keys['processed_hashes_order'], to_remove_count, -1)
                    logger.info(f"成功删除{len(oldest_hashes)}条最早的处理记录")
        except Exception as e:
            logger.error(f"清理旧处理记录时出错: {str(e)}")

    def mark_news_as_processed(self, news_hash: str):
        """标记新闻为已处理"""
        try:
            # 先检查是否已处理过
            if self.redis_client.sismember(self.news_keys['processed_hashes'], news_hash):
                logger.info(f"新闻哈希 {news_hash[:8]}... 已处理过，无需再次标记")
                return

            # 添加到处理过的集合
            self.redis_client.sadd(self.news_keys['processed_hashes'], news_hash)
            # 添加到有序列表末尾，记录处理顺序
            self.redis_client.rpush(self.news_keys['processed_hashes_order'], news_hash)
            self.processed_news_hashes.add(news_hash)
            logger.info(f"标记新闻哈希 {news_hash[:8]}... 为已处理")

            # 如果该新闻在重试队列中，从重试队列移除
            if news_hash in self.retry_news:
                del self.retry_news[news_hash]
                self.save_retry_news()
                logger.info(f"从重试队列中移除已处理的新闻 {news_hash[:8]}...")

            # 检查并清理旧记录
            self.clean_old_processed_hashes()
        except Exception as e:
            logger.error(f"标记新闻为已处理时出错: {str(e)}")

    def get_news_content(self, news_hash, timeout=2):
        """获取新闻内容，添加超时机制"""
        try:
            start_time = time.time()
            # 获取所有新闻
            news_list = self.redis_client.lrange(self.news_keys['hot_news'], 0, -1)

            for news_item in news_list:
                # 检查超时
                if time.time() - start_time > timeout:
                    logger.warning(f"获取新闻内容超时 ({timeout}秒)")
                    return None

                try:
                    news_data = json.loads(news_item)
                    content = news_data.get('content', '')
                    datetime_str = news_data.get('datetime', '')
                    current_hash = self._generate_news_hash(content, datetime_str)

                    if current_hash == news_hash:
                        return {
                            'content': content[:100] + ('...' if len(content) > 100 else ''),
                            'datetime': datetime_str
                        }
                except Exception as inner_e:
                    logger.error(f"处理单条新闻时出错: {str(inner_e)}")
                    continue

            return None
        except Exception as e:
            logger.error(f"获取新闻内容时出错: {str(e)}")
            return None

    def _generate_news_hash(self, content, datetime_str):
        """生成新闻哈希值"""
        import hashlib
        hash_str = f"{content}|{datetime_str}"
        return hashlib.md5(hash_str.encode('utf-8')).hexdigest()

    def get_all_news_analyses(self, timeout=3):
        """获取全部的新闻分析结果，返回hash到分析的映射"""
        try:
            start_time = time.time()
            analyses_map = {}

            # 获取所有分析结果
            analyses_data = self.redis_client.hgetall(self.news_keys['all_analyses'])

            if not analyses_data:
                logger.warning("Redis中没有找到任何新闻分析结果")
                return {}

            # 将所有分析结果解析为JSON并保存到映射中
            for news_hash, analysis_json in analyses_data.items():
                # 检查超时
                if time.time() - start_time > timeout:
                    logger.warning(f"获取全部新闻分析超时 ({timeout}秒)")
                    break

                try:
                    analysis_data = json.loads(analysis_json)
                    analyses_map[news_hash] = analysis_data
                except Exception as e:
                    logger.error(f"解析分析结果JSON出错: {str(e)}")
                    continue

            return analyses_map
        except Exception as e:
            logger.error(f"获取全部新闻分析结果时出错: {str(e)}")
            return {}

    def get_latest_news(self, limit=5, timeout=3):
        """获取最新的几条新闻"""
        try:
            start_time = time.time()

            # 获取所有新闻
            all_news = []
            news_list = self.redis_client.lrange(self.news_keys['hot_news'], 0, limit - 1)

            for news_item in news_list:
                # 检查超时
                if time.time() - start_time > timeout:
                    logger.warning(f"获取最新新闻超时 ({timeout}秒)")
                    break

                try:
                    news_data = json.loads(news_item)
                    content = news_data.get('content', '')
                    datetime_str = news_data.get('datetime', '')
                    news_hash = self._generate_news_hash(content, datetime_str)

                    all_news.append({
                        'hash': news_hash,
                        'content': content,
                        'datetime': datetime_str
                    })
                except Exception as e:
                    logger.error(f"处理新闻JSON出错: {str(e)}")
                    continue

            # 按时间倒序排序
            all_news.sort(key=lambda x: x['datetime'], reverse=True)
            return all_news[:limit]
        except Exception as e:
            logger.error(f"获取最新新闻时出错: {str(e)}")
            return []

    def process_news_analysis(self):
        """处理新闻分析结果，只提取影响程度为强的上涨股票，只处理最新5条未处理的新闻"""
        try:
            # 获取已处理的新闻哈希集合
            self.processed_news_hashes = self.get_processed_news_hashes()
            logger.info(f"已有 {len(self.processed_news_hashes)} 条新闻被处理过")

            # 加载待重试的新闻
            self.load_retry_news()

            # 获取待处理的新闻列表
            news_to_process = []

            # 首先添加待重试的新闻
            retry_news_list = []
            for news_hash, retry_count in list(self.retry_news.items()):
                # 检查重试次数是否超过最大值
                if retry_count >= self.max_retry_times:
                    logger.info(f"新闻 {news_hash[:8]}... 已达到最大重试次数 {self.max_retry_times}，标记为已处理")
                    self.mark_news_as_processed(news_hash)
                    continue

                # 获取新闻内容
                news_content = self.get_news_content(news_hash)
                if news_content:
                    retry_news_list.append({
                        'hash': news_hash,
                        'content': news_content.get('content', ''),
                        'datetime': news_content.get('datetime', ''),
                        'retry_count': retry_count
                    })
                    logger.info(f"加载待重试新闻: {news_hash[:8]}..., 重试次数: {retry_count}")
                else:
                    logger.warning(f"无法获取待重试新闻 {news_hash[:8]}... 的内容，将移除")
                    del self.retry_news[news_hash]

            # 获取最新的5条新闻（减去待重试的数量）
            remaining_limit = max(1, 5 - len(retry_news_list))
            logger.info(f"正在获取最新的 {remaining_limit} 条新闻...")

            latest_news = self.get_latest_news(limit=remaining_limit)
            if not latest_news and not retry_news_list:
                print("没有找到任何新闻，也没有待重试的新闻")
                return False

            # 合并待重试和最新新闻
            news_to_process = retry_news_list + latest_news
            print(f"找到 {len(news_to_process)} 条需要处理的新闻（其中待重试 {len(retry_news_list)} 条）")

            # 获取所有新闻分析结果
            logger.info("正在获取所有新闻分析结果...")
            all_analyses = self.get_all_news_analyses()
            if not all_analyses:
                # 如果没有任何分析结果，将所有未处理的新闻加入重试队列
                for news in news_to_process:
                    if news['hash'] not in self.processed_news_hashes:
                        self.add_news_to_retry(news['hash'], news)
                print("没有找到任何新闻分析结果，已将未处理新闻加入重试队列")
                return False

            print(f"找到 {len(all_analyses)} 条新闻分析结果")

            # 用于收集所有强影响上涨股票（只关注上涨股票）
            all_strong_stocks = []

            # 处理每条新闻
            processed_count = 0
            for news in news_to_process:
                news_hash = news['hash']
                retry_count = news.get('retry_count', 0)

                # 检查是否已处理
                if news_hash in self.processed_news_hashes:
                    logger.info(f"新闻 {news_hash[:8]}... 已处理过，跳过")
                    # 如果在重试队列中，移除
                    if news_hash in self.retry_news:
                        del self.retry_news[news_hash]
                    continue

                # 检查是否有对应的分析结果
                if news_hash not in all_analyses:
                    logger.info(f"新闻 {news_hash[:8]}... 没有对应的分析结果")
                    # 添加到重试队列
                    if retry_count < self.max_retry_times:
                        self.add_news_to_retry(news_hash, news)
                        logger.info(f"新闻 {news_hash[:8]}... 加入重试队列，当前重试次数: {self.retry_news[news_hash]}")
                    else:
                        logger.warning(f"新闻 {news_hash[:8]}... 已达到最大重试次数 {self.max_retry_times}，标记为已处理")
                        self.mark_news_as_processed(news_hash)
                    continue

                # 获取分析结果
                analysis = all_analyses[news_hash]

                # 提取分析结果中影响程度为强的上涨股票
                potential_risers = analysis.get('potential_risers', [])
                strong_risers = [stock for stock in potential_risers if stock.get('influence', '') == '强']

                # 只关注强影响的上涨股票，忽略下跌股票
                all_strong_stocks.extend(strong_risers)

                # 输出结果
                retry_info = f"（重试第 {retry_count} 次）" if retry_count > 0 else ""
                print(f"\n处理新闻{retry_info}: {news['content'][:100]}...")
                print(f"发布时间: {news['datetime']}")

                if strong_risers:
                    print(f"发现 {len(strong_risers)} 只影响程度为强的潜在上涨股票:")
                    for stock in strong_risers:
                        code = stock.get('code', '')
                        name = stock.get('name', '')
                        reason = stock.get('reason', '')
                        print(f"  强影响上涨股票: {name}({code}), 原因: {reason}")
                else:
                    print("  此新闻中没有发现影响程度为强的上涨股票")

                # 标记该新闻为已处理
                self.mark_news_as_processed(news_hash)
                processed_count += 1

            # 将收集到的强影响上涨股票保存到配置文件中
            if all_strong_stocks:
                self.save_strong_impact_stocks(all_strong_stocks)

            # 更新重试队列
            self.save_retry_news()

            print(f"\n共处理了 {processed_count} 条新闻，还有 {len(self.retry_news)} 条等待下次重试")
            return True

        except Exception as e:
            logger.error(f"处理新闻分析时出错: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def cleanup(self):
        """清理资源"""
        try:
            # 保存待重试队列
            self.save_retry_news()

            # 关闭Redis连接
            self.redis_client.close()
            logger.info("已关闭Redis连接")
        except Exception as e:
            logger.error(f"清理资源时出错: {str(e)}")


def main():
    """主函数"""
    print("启动新闻分析接收器...")
    receiver = NewsAnalysisReceiver()

    try:
        # 添加循环，每10秒执行一次
        while True:
            # 设置全局超时
            start_time = time.time()
            timeout = 30  # 30秒超时

            # 记录开始执行时间
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f"\n===== {current_time} - 开始新一轮新闻分析 =====")

            # 处理新闻分析
            print("开始处理新闻分析...")
            success = receiver.process_news_analysis()

            if time.time() - start_time > timeout:
                print("处理超时，本轮分析中断")
            elif success:
                print("成功处理新闻分析结果")
            else:
                print("处理新闻分析结果失败")

            # 计算本轮耗时
            elapsed = time.time() - start_time
            print(f"本轮分析耗时: {elapsed:.2f}秒")

            # 计算需要等待的时间
            wait_time = max(0, 10 - elapsed)
            if wait_time > 0:
                print(f"等待{wait_time:.2f}秒后开始下一轮分析...")
                time.sleep(wait_time)
            else:
                print("本轮处理时间超过10秒，立即开始下一轮分析...")

    except KeyboardInterrupt:
        print("\n检测到Ctrl+C，程序正在退出...")
    except Exception as e:
        print(f"运行过程中出错: {str(e)}")
        import traceback
        print(traceback.format_exc())
    finally:
        print("清理资源...")
        receiver.cleanup()
        print("程序执行完毕")


if __name__ == "__main__":
    main()