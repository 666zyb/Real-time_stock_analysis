import asyncio
import aiohttp
import datetime
from typing import List, Dict
from data.hot_News_data import NewsSpider


class ThsSpider(NewsSpider):
    def __init__(self):
        self.headers = {
            "Accept": "*/*",
            "Accept-Language": "zh-CN,zh;q=0.9",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Pragma": "no-cache",
            "Referer": "https://news.10jqka.com.cn/realtimenews.html",
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-origin",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 SLBrowser/9.0.6.2081 SLBChan/105 SLBVPV/64-bit",
            "X-Requested-With": "XMLHttpRequest",
            "hexin-v": "AzUodaCTviZiVdqaYh5KKMroRLrrsunEs2bNGLda8az7jlsmfwL5lEO23epE",
            "^sec-ch-ua": "^\\^Chromium^^;v=^\\^9^^, ^\\^Not?A_Brand^^;v=^\\^8^^^",
            "sec-ch-ua-mobile": "?0",
            "^sec-ch-ua-platform": "^\\^Windows^^^"
        }
        self.cookies = {
            "log": "",
            "Hm_lvt_722143063e4892925903024537075d0d": "1741833391",
            "Hm_lpvt_722143063e4892925903024537075d0d": "1741833391",
            "HMACCOUNT": "B94D632BC5BD252A",
            "Hm_lvt_929f8b362150b1f77b477230541dbbc2": "1741833391",
            "Hm_lpvt_929f8b362150b1f77b477230541dbbc2": "1741833391",
            "Hm_lvt_78c58f01938e4d85eaf619eae71b4ed1": "1741833391",
            "Hm_lvt_48bd99bf09a11eefce1a981090551bcd": "1741833467",
            "Hm_lpvt_48bd99bf09a11eefce1a981090551bcd": "1741833467",
            "Hm_lpvt_78c58f01938e4d85eaf619eae71b4ed1": "1741833467",
            "Hm_lvt_f79b64788a4e377c608617fba4c736e2": "1741833467",
            "Hm_lpvt_f79b64788a4e377c608617fba4c736e2": "1741833467",
            "v": "AzUodaCTviZiVdqaYh5KKMroRLrrsunEs2bNGLda8az7jlsmfwL5lEO23epE"
        }
        self.url = "https://news.10jqka.com.cn/tapp/news/push/stock/"
        self.params = {
            "page": "1",
            "tag": "",
            "track": "website",
            "pagesize": "400"
        }

    async def fetch_news(self) -> List[Dict]:
        """实现NewsSpider的抽象方法，确保返回List[Dict]格式"""
        try:
            async with aiohttp.ClientSession(headers=self.headers, cookies=self.cookies) as session:
                async with session.get(self.url, params=self.params) as response:
                    res = await response.json()
                    news_list = []  # 创建列表存储新闻

                    # 确保返回的是列表格式
                    if 'data' in res and 'list' in res['data']:
                        for item in res['data']['list']:
                            time=int(item['ctime'])
                            # 确保datetime格式正确
                            ctime=datetime.datetime.fromtimestamp(time).strftime("%Y-%m-%d %H:%M:%S")
                            news_item = {
                                'content': item['digest'],
                                'datetime': ctime
                            }
                            news_list.append(news_item)

                    return news_list  # 返回新闻列表

        except Exception as e:
            print(f"同花顺新闻获取失败: {str(e)}")
            return []  # 出错时返回空列表

# 测试代码
async def test():
    spider = ThsSpider()
    news = await spider.fetch_news()
    print(f"获取到 {len(news)} 条新闻")
    for item in news[:3]:  # 只打印前3条
        print(item)

if __name__=="__main__":
    asyncio.run(test())