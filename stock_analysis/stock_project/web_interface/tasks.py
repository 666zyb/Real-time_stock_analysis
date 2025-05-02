import asyncio
import threading
from .services.stock_service import StockDataService
from .services.news_service import NewsService


class BackgroundTasks:
    def __init__(self):
        self.stock_service = StockDataService()
        self.news_service = NewsService()
        self.running = False

    async def update_stock_data(self):
        """更新股票数据"""
        while self.running:
            try:
                await self.stock_service.update_all_stocks()
                # 每5秒更新一次
                await asyncio.sleep(5)
            except Exception as e:
                print(f"更新股票数据时出错: {str(e)}")
                await asyncio.sleep(10)

    async def update_news_data(self):
        """从Redis更新新闻数据到数据库"""
        while self.running:
            try:
                await self.news_service.update_news_from_redis()
                # 每5分钟更新一次
                await asyncio.sleep(300)
            except Exception as e:
                print(f"更新新闻数据时出错: {str(e)}")
                await asyncio.sleep(60)

    async def run_tasks(self):
        """运行所有后台任务"""
        self.running = True
        tasks = [
            self.update_stock_data(),
            self.update_news_data()
        ]
        await asyncio.gather(*tasks)

    def start(self):
        """在后台线程中启动任务"""

        def _run_event_loop():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.run_tasks())

        # 创建并启动后台线程
        thread = threading.Thread(target=_run_event_loop)
        thread.daemon = True
        thread.start()

    def stop(self):
        """停止任务"""
        self.running = False


# 创建全局实例
background_tasks = BackgroundTasks()