import asyncio
import aiohttp
import datetime
from typing import List, Dict
#from stock_analysis.data.hot_News_data import NewsSpider
from stock_analysis.News_crawler.财联社 import NewsSpider
import requests

class ClsSpider(NewsSpider):
    def __init__(self):
        self.headers = {
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "zh-CN,zh;q=0.9",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "application/json;charset=utf-8",
            "Pragma": "no-cache",
            "Referer": "https://www.cls.cn/telegraph",
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-origin",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 SLBrowser/9.0.6.2081 SLBChan/105 SLBVPV/64-bit",
            "^sec-ch-ua": "^\\^Chromium^^;v=^\\^9^^, ^\\^Not?A_Brand^^;v=^\\^8^^^",
            "sec-ch-ua-mobile": "?0",
            "^sec-ch-ua-platform": "^\\^Windows^^^"
        }
        self.cookies = {
            "HWWAFSESID": "5dfe7bee3af39fe71d",
            "HWWAFSESTIME": "1741704965599",
            "Hm_lvt_fa5455bb5e9f0f260c32a1d45603ba3e": "1741704970",
            "HMACCOUNT": "6BDEDF6F6B68A5AF",
            "hasTelegraphNotification": "on",
            "hasTelegraphSound": "on",
            "vipNotificationState": "on",
            "hasTelegraphRemind": "off",
            "Hm_lpvt_fa5455bb5e9f0f260c32a1d45603ba3e": "1741944586",
            "tfstk": "goNoav_V38kWRxE92ol5B9TYAVXAtUGI3kdKvXnF3moX23d88Db38mF-pboptwqTf2QS9WL7hlUww7BS9BcSOXSOX1h3yzGITqdMjPxS04gF94kEYbGVI9dFM1CTPz8-zNUV67n9zlm-YXkrUK8qcV0eYkky0nojR2-r8Q70umiq82or4s5qP2AeYklForoj8XurIzSr4WP8gi5Jn277uZeorYmaz0z8ySWqeLaomBRUildqjz8WTBPmqYVC4kdJGYeU5AN8J6AjwP2qg2VC7pmnIJP-g7SV_xMULRH3H378bk44AbFV8Loaeuh4aRYeTrcomv3oF3jmb7a41jkA_N44GuU7iyLFTqEt0zNrt17sZboriqNdcQo33JP-FXtPxDN0u7yh4Cp2Q0reOldnlp9IUqimXAKLJ4V6IPJGoZviAYujzcQcop9IUqgmXZbDIDMrl4oO."
        }
        self.url = "https://www.cls.cn/nodeapi/updateTelegraphList"
        self.params = {
            "app": "CailianpressWeb",
            "category": "",
            "hasFirstVipArticle": "1",
            "lastTime": "1741944657",
            "os": "web",
            "rn": "20",
            "subscribedColumnIds": "",
            "sv": "8.4.6",
            # "sign": "20f0bed2ebfd1a3fcf588d58989a7c85"
        }

    async def fetch_news(self) -> List[Dict]:
        """实现NewsSpider的抽象方法，确保返回List[Dict]格式"""
        try:
            async with aiohttp.ClientSession(headers=self.headers, cookies=self.cookies) as session:
                async with session.get(self.url, params=self.params) as response:
                    res = await response.json()
                    news_list = []  # 创建列表存储新闻

                    # 确保返回的是列表格式
                    if 'data' in res and 'roll_data' in res['data']:
                        for item in res['data']['roll_data']:
                            time=int(item['ctime'])
                            level=item['level']
                            if level=="A":
                                color="-21101"
                            elif level=="B":
                                color="21111"
                            else:
                                color=""
                            # 确保datetime格式正确
                            ctime=datetime.datetime.fromtimestamp(time).strftime("%Y-%m-%d %H:%M:%S")
                            news_item = {
                                'content': item['content'],
                                'datetime': ctime,
                                'color':color
                            }
                            news_list.append(news_item)

                        for item in res['vipGlobal']:
                            time = int(item['ctime'])
                            # 确保datetime格式正确
                            ctime = datetime.datetime.fromtimestamp(time).strftime("%Y-%m-%d %H:%M:%S")
                            news_item = {
                                'content': item['brief'],
                                'datetime': ctime,
                                'color':'-21101'
                            }
                            news_list.append(news_item)

                    return news_list  # 返回新闻列表

        except Exception as e:
            print(f"财联社新闻获取失败: {str(e)}")
            return []  # 出错时返回空列表

async def test():
    spider = ClsSpider()
    news = await spider.fetch_news()
    print(f"获取到 {len(news)} 条新闻")
    for item in news:  # 只打印前3条
        print(item)


if __name__=="__main__":
    asyncio.run(test())