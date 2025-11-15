import asyncio
import aiohttp
from typing import List, Dict
#from stock_analysis.data.hot_News_data import NewsSpider
from  stock_analysis.News_crawler.新浪财经 import NewsSpider

class SinaSpider(NewsSpider):
    def __init__(self):
        """新浪财经的新闻爬虫"""
        self.headers = {
            "accept": "*/*",
            "accept-language": "zh-CN,zh;q=0.9",
            "cache-control": "no-cache",
            "pragma": "no-cache",
            "referer": "https://finance.sina.com.cn/stock/",
            "sec-ch-ua": "\"Chromium\";v=\"9\", \"Not?A_Brand\";v=\"8\"",
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": "\"Windows\"",
            "sec-fetch-dest": "script",
            "sec-fetch-mode": "no-cors",
            "sec-fetch-site": "same-site",
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 SLBrowser/9.0.6.2081 SLBChan/105 SLBVPV/64-bit"
        }
        self.cookies = {
            "UOR": "www.baidu.com,finance.sina.com.cn,",
            "SINAGLOBAL": "223.160.141.105_1741230958.531690",
            "Apache": "223.160.141.143_1741704337.763544",
            "ULV": "1741704387339:6:6:2:223.160.141.143_1741704337.763544:1741704337688"
        }
        self.url = "https://zhibo.sina.com.cn/api/zhibo/feed"
        self.params = {
            "zhibo_id": "152",
            "id": "",
            "tag_id": "0",
            "page": "1",
            "page_size": "20",
            "type": "0",
            # "callback": "t17417483"
        }

    async def fetch_news(self) -> List[Dict]:
        """实现NewsSpider的抽象方法，确保返回List[Dict]格式"""
        try:
            async with aiohttp.ClientSession(headers=self.headers, cookies=self.cookies) as session:
                async with session.get(self.url, params=self.params) as response:
                    res = await response.json()
                    news_list = []  # 创建列表存储新闻

                    # 确保返回的是列表格式
                    if 'result' in res and 'data' in res['result'] and 'feed' in res['result']['data']:
                        for item in res['result']['data']['feed']['list']:
                            news_item = {
                                'content': item['rich_text'],
                                'datetime': item['create_time']  # 确保datetime格式正确
                            }
                            news_list.append(news_item)

                    return news_list  # 返回新闻列表

        except Exception as e:
            print(f"新浪财经新闻获取失败: {str(e)}")
            return []  # 出错时返回空列表


# 测试代码
async def test():
    spider = SinaSpider()
    news = await spider.fetch_news()
    print(f"获取到 {len(news)} 条新闻")
    for item in news[:3]:  # 只打印前3条
        print(item)


if __name__ == "__main__":
    asyncio.run(test())
