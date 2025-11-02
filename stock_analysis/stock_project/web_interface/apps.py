from django.apps import AppConfig
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)  # 调整这个层级

print(f"当前目录: {current_dir}")
print(f"项目根目录: {project_root}")
print(f"项目根目录是否存在: {os.path.exists(project_root)}")

# 检查stock_analysis目录是否存在
stock_analysis_path = os.path.join(project_root, 'stock_analysis')
print(f"stock_analysis路径: {stock_analysis_path}")
print(f"stock_analysis是否存在: {os.path.exists(stock_analysis_path)}")

# 列出项目根目录下的所有文件和文件夹
print("项目根目录内容:", os.listdir(project_root))

# 添加项目根目录到Python路径
if project_root not in sys.path:
    sys.path.insert(0, project_root)

print("Python路径:", sys.path)

#from stock_analysis.stock_project.News_analysis.news_stock_analysis import NewsStockAnalyzer
from News_analysis.news_stock_analysis import NewsStockAnalyzer
import threading
import asyncio
import json
import redis
import time
from django.conf import settings
import logging

logger = logging.getLogger(__name__)


class WebInterfaceConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'web_interface'

    def ready(self):
        # 避免在Django重载时多次执行
        if os.environ.get('RUN_MAIN', None) != 'true':
            self.start_analysis_scheduler()
            self.start_news_monitor()  # 启动新闻监控线程

    def start_analysis_scheduler(self):
        """启动定期分析任务"""

        def run_analysis_thread():
            import asyncio
            import time

            print("启动新闻分析线程")

            while True:
                try:
                    # 创建一个新的事件循环
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

                    # 执行分析，每次最多只分析5条最新的新闻
                    analyzer = NewsStockAnalyzer()
                    result = loop.run_until_complete(analyzer.run_analysis(max_news=5, parallel_count=1))
                    print(f"分析完成，结果: {result.get('analysis', '无结果')}")

                    # 关闭循环
                    loop.close()

                    # 等待2小时后再次分析，减少API请求频率
                    time.sleep(7200)
                except Exception as e:
                    print(f"分析出错: {str(e)}")
                    time.sleep(300)  # 出错后等待5分钟再试

        # 启动后台线程
        t = threading.Thread(target=run_analysis_thread, daemon=True)
        t.start()

    def start_news_monitor(self):
        """启动新闻监控线程"""

        def news_monitor():
            import traceback

            logger.info("启动新闻实时分析监控线程")

            try:
                # 加载配置
                config_path = os.path.join(settings.BASE_DIR, 'config', 'config.json')
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)

                redis_config = config.get('redis_config', {})

                # 创建两个Redis连接：一个用于订阅，一个用于查询
                sub_client = redis.Redis(
                    host=redis_config.get('host', '127.0.0.1'),
                    port=redis_config.get('port', 6379),
                    db=redis_config.get('db', 0),
                    password=redis_config.get('password'),
                    decode_responses=True
                )

                redis_client = redis.Redis(
                    host=redis_config.get('host', '127.0.0.1'),
                    port=redis_config.get('port', 6379),
                    db=redis_config.get('db', 0),
                    password=redis_config.get('password'),
                    decode_responses=True
                )

                # 创建订阅对象
                pubsub = sub_client.pubsub()

                # 订阅新闻添加频道
                pubsub.subscribe('stock:hot_news:add')

                logger.info("开始监听新闻添加事件")

                # 创建事件循环
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

                # 初始化分析器
                analyzer = NewsStockAnalyzer(config_path)

                # 异步初始化分析器
                loop.run_until_complete(analyzer.initialize())
                logger.info("分析器初始化完成")

                # 立即分析现有未分析的新闻
                async def analyze_existing_news():
                    try:
                        # 获取未分析的新闻，但限制只获取最新的5条，避免启动时处理过多导致超时
                        news_list, news_hashes = await analyzer.get_news_from_redis(limit=5)
                        if news_list:
                            logger.info(f"启动时发现 {len(news_list)} 条未分析新闻，限制只分析最新的5条")
                            # 使用串行处理，每条新闻处理完后有足够的间隔
                            processed_hashes = await analyzer.process_news_batches(news_list[:5], news_hashes[:5], max_concurrent=1)
                            logger.info(f"启动时分析完成 {len(processed_hashes)} 条新闻")
                    except Exception as e:
                        logger.error(f"启动时分析新闻出错: {str(e)}")
                        traceback.print_exc()

                # 先分析现有未分析的新闻
                loop.run_until_complete(analyze_existing_news())

                # 监听新闻添加事件
                for message in pubsub.listen():
                    if message['type'] == 'message':
                        try:
                            # 解析消息数据
                            data = json.loads(message['data'])
                            news_hash = data.get('hash')

                            if news_hash:
                                logger.info(f"收到新闻更新事件，hash: {news_hash}")

                                # 获取单条新闻内容
                                news_index = data.get('index', 0)
                                news_item = redis_client.lindex('stock:hot_news', news_index)

                                if news_item:
                                    news_data = json.loads(news_item)

                                    # 异步分析新闻
                                    async def analyze_single_news():
                                        try:
                                            logger.info(f"开始分析新闻: {news_hash}")
                                            # 设置较高的优先级，确保最新的新闻得到优先处理
                                            result = await analyzer.analyze_news_with_ai(news_data)

                                            if result:
                                                # 将分析结果保存到Redis
                                                redis_client.hset(
                                                    "stock:news_all_analyses",
                                                    news_hash,
                                                    json.dumps(result, ensure_ascii=False)
                                                )
                                                logger.info(f"新闻分析完成: {news_hash}")

                                                # 标记为已分析
                                                analyzer.mark_news_as_analyzed([news_hash])

                                                # 将这个最新分析的新闻结果添加到汇总
                                                try:
                                                    # 获取这条新闻的分析结果
                                                    latest_results = [result]

                                                    # 合并结果
                                                    combined_result = analyzer.combine_analysis_results(latest_results)
                                                    combined_result['is_new_analysis'] = True
                                                    combined_result['priority_update'] = True  # 标记为优先更新

                                                    # 保存到Redis
                                                    redis_client.set(
                                                        "stock:news_latest_analysis",
                                                        json.dumps(combined_result, ensure_ascii=False)
                                                    )

                                                    # 同时更新总体分析
                                                    redis_client.set(
                                                        "stock:news_analysis_summary",
                                                        json.dumps(combined_result, ensure_ascii=False)
                                                    )

                                                    logger.info(f"已使用最新分析的新闻 {news_hash} 更新分析汇总")
                                                except Exception as e:
                                                    logger.error(f"更新分析汇总时出错: {str(e)}")
                                            else:
                                                logger.warning(f"新闻分析结果为空: {news_hash}")

                                        except Exception as e:
                                            logger.error(f"分析单条新闻时出错: {str(e)}")
                                            traceback.print_exc()

                                    # 执行异步分析
                                    loop.run_until_complete(analyze_single_news())

                        except Exception as e:
                            logger.error(f"处理新闻更新事件时出错: {str(e)}")
                            traceback.print_exc()

            except Exception as e:
                logger.error(f"新闻监控线程出错: {str(e)}")
                traceback.print_exc()
            finally:
                # 关闭Redis连接
                try:
                    pubsub.unsubscribe()
                    sub_client.close()
                    redis_client.close()

                    # 关闭分析器
                    loop.run_until_complete(analyzer.close())
                except:
                    pass

                logger.info("新闻监控线程结束")

        # 启动监控线程
        t = threading.Thread(target=news_monitor, daemon=True)
        t.start()
        logger.info("新闻监控线程已启动")
