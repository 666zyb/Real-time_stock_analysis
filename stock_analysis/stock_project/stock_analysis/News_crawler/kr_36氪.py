import requests
import datetime
import re
import asyncio
from lxml import etree
from typing import List, Dict
from data.hot_News_data import NewsSpider # 导入NewsSpider基类



class Kr36Spider(NewsSpider):
    def __init__(self):
        self.data = {
            "partner_id": "web",
            # "timestamp": 1737544523332,
            "param": {
                "type": 0,
                "subnavNick": "web_news",
                "pageSize": 20,
                "pageEvent": 1,
                # "pageCallback": "eyJmaXJzdElkIjo0ODM0MTYwLCJsYXN0SWQiOjQ4MzQwMTcsImZpcnN0Q3JlYXRlVGltZSI6MTczNzYwNTIyODk3MSwibGFzdENyZWF0ZVRpbWUiOjE3Mzc1OTkxNTQ0OTd9",
                "siteId": 1,
                "platformId": 2
            }
        }
        self.url = "https://gateway.36kr.com/api/mis/nav/newsflash/list"
        self.headers = {
            "Accept": "*/*",
            "Accept-Language": "zh-CN,zh;q=0.9",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "application/json",
            "^Cookie": "Hm_lvt_713123c60a0e86982326bae1a51083e1=1737541674; HMACCOUNT=6B3DD5184F6958B7; Hm_lvt_1684191ccae0314c6254306a8333d090=1737541674; sajssdk_2015_cross_new_user=1; sensorsdata2015jssdkcross=^%^7B^%^22distinct_id^%^22^%^3A^%^221948d8de5f286a-05623ef289df0b-27b4036-921600-1948d8de5f36a5^%^22^%^2C^%^22^%^24device_id^%^22^%^3A^%^221948d8de5f286a-05623ef289df0b-27b4036-921600-1948d8de5f36a5^%^22^%^2C^%^22props^%^22^%^3A^%^7B^%^22^%^24latest_traffic_source_type^%^22^%^3A^%^22^%^E7^%^9B^%^B4^%^E6^%^8E^%^A5^%^E6^%^B5^%^81^%^E9^%^87^%^8F^%^22^%^2C^%^22^%^24latest_referrer^%^22^%^3A^%^22^%^22^%^2C^%^22^%^24latest_referrer_host^%^22^%^3A^%^22^%^22^%^2C^%^22^%^24latest_search_keyword^%^22^%^3A^%^22^%^E6^%^9C^%^AA^%^E5^%^8F^%^96^%^E5^%^88^%^B0^%^E5^%^80^%^BC_^%^E7^%^9B^%^B4^%^E6^%^8E^%^A5^%^E6^%^89^%^93^%^E5^%^BC^%^80^%^22^%^7D^%^7D; Hm_lpvt_1684191ccae0314c6254306a8333d090=1737544084; Hm_lpvt_713123c60a0e86982326bae1a51083e1=1737544084^",
            "Origin": "https://36kr.com",
            "Pragma": "no-cache",
            "Referer": "https://36kr.com/",
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-site",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 SLBrowser/9.0.5.12181 SLBChan/105 SLBVPV/64-bit",
            "^sec-ch-ua": "^\\^Chromium^^;v=^\\^9^^, ^\\^Not?A_Brand^^;v=^\\^8^^^",
            "sec-ch-ua-mobile": "?0",
            "^sec-ch-ua-platform": "^\\^Windows^^^"
        }

        self.items = []

    def get_time(self, time):
        """经多少分钟前转换为（%Y-%m-%d %H:%M:%S）格式"""

        # 获取当前时间
        now = datetime.datetime.now()

        pattern = r'(\d+)'

        # 判断是分钟前，小时前，昨天，天前还是周前
        pattern_1 = r'\d+([\u4e00-\u9fa5]+)'
        # 进行匹配
        try:
            match = int(re.search(pattern, time).group(1))
            # print(match)
            match_1 = re.search(pattern_1, time).group(1)
        except:
            match=24
            match_1='小时前'
        if match_1 == '秒前':
            # 计算多少秒前的时间
            time_1 = now - datetime.timedelta(seconds=match)
        elif match_1 == '分钟前':
            # 计算多少分钟前的时间
            time_1 = now - datetime.timedelta(minutes=match)
        elif match_1 == '天前':
            # 计算多少分钟前的时间
            time_1 = now - datetime.timedelta(days=match)
        else:
            # 计算多少小时前的时间
            time_1 = now - datetime.timedelta(hours=match)

        return time_1.strftime("%Y-%m-%d %H:%M:%S")

    def get_pageCallback(self):
        """获取第一个pageCallback"""
        headers = {
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
            "Accept-Language": "zh-CN,zh;q=0.9",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "^Cookie": "Hm_lvt_713123c60a0e86982326bae1a51083e1=1741705140; HMACCOUNT=7DE2F8B0B64FE0FF; Hm_lvt_1684191ccae0314c6254306a8333d090=1741705140; sensorsdata2015jssdkcross=^%^7B^%^22distinct_id^%^22^%^3A^%^2219585b75ac87e-067844194817bf-482a5d03-921600-19585b75ac91678^%^22^%^2C^%^22^%^24device_id^%^22^%^3A^%^2219585b75ac87e-067844194817bf-482a5d03-921600-19585b75ac91678^%^22^%^2C^%^22props^%^22^%^3A^%^7B^%^22^%^24latest_traffic_source_type^%^22^%^3A^%^22^%^E8^%^87^%^AA^%^E7^%^84^%^B6^%^E6^%^90^%^9C^%^E7^%^B4^%^A2^%^E6^%^B5^%^81^%^E9^%^87^%^8F^%^22^%^2C^%^22^%^24latest_referrer^%^22^%^3A^%^22https^%^3A^%^2F^%^2Fwww.baidu.com^%^2Flink^%^22^%^2C^%^22^%^24latest_referrer_host^%^22^%^3A^%^22www.baidu.com^%^22^%^2C^%^22^%^24latest_search_keyword^%^22^%^3A^%^22^%^E6^%^9C^%^AA^%^E5^%^8F^%^96^%^E5^%^88^%^B0^%^E5^%^80^%^BC^%^22^%^7D^%^7D; Hm_lpvt_713123c60a0e86982326bae1a51083e1=1741769932; Hm_lpvt_1684191ccae0314c6254306a8333d090=1741769932; SERVERID=6eb0a1872728d69c244094a636b7db3b^|1741769932^|1741769234^",
            "Pragma": "no-cache",
            "Referer": "https://36kr.com/newsflashes/catalog/3",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "same-origin",
            "Sec-Fetch-User": "?1",
            "Upgrade-Insecure-Requests": "1",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 SLBrowser/9.0.6.2081 SLBChan/105 SLBVPV/64-bit",
            "^sec-ch-ua": "^\\^Chromium^^;v=^\\^9^^, ^\\^Not?A_Brand^^;v=^\\^8^^^",
            "sec-ch-ua-mobile": "?0",
            "^sec-ch-ua-platform": "^\\^Windows^^^"
        }
        url = "https://36kr.com/newsflashes/catalog/0"
        response = requests.get(url, headers=headers)

        html = etree.HTML(response.text)
        item = dict()
        for i in html.xpath('//div[@class="newsflash-catalog-flow-list"]/div[@class="flow-item"]'):
            item = dict()
            item['content'] = i.xpath('./div[@class="item-main"]/div[@class="newsflash-item"]/div[@class="item-desc"]/span/text()')[0]
            time = i.xpath('./div[@class="item-main"]/div[@class="newsflash-item"]/div[@class="item-other"]/div[@class="item-related"]/span/text()')[0]
            # 将多少时间前转换为2025-03-12 11:43:35格式
            item['datetime'] = self.get_time(time)
            self.items.append(item)

        # print(self.items)
        # 获取第一个pageCallback,因为后面每次请求都会用到上一次请求的pageCallback
        # 定义正则表达式模式来匹配pageCallback(只出现了一次)
        pattern = r'"pageCallback":\s*"([^"]+)"'
        # 进行匹配
        match = re.search(pattern, response.text)
        if match:
            return match.group(1)
        else:
            print('未找到pageCallback的值')
            return None

    def get_data(self, pageCallback):
        self.data['param']['pageCallback'] = pageCallback
        # print(self.data)
        response = requests.post(url=self.url, headers=self.headers, json=self.data)
        # print(response)
        return response.json()

    async def fetch_news(self) -> List[Dict]:
        """
        实现NewsSpider的抽象方法
        返回新闻列表
        """
        try:
            self.items = []  # 清空之前的数据
            pageCallback = self.get_pageCallback()

            for page in range(1, 5):
                res = self.get_data(pageCallback)
                pageCallback = res["data"]["pageCallback"]
                self.data["param"]["pageCallback"] = pageCallback

                for i in res['data']['itemList']:
                    item = dict()
                    try:
                        item['content'] = i['templateMaterial']['widgetContent']
                    except:
                        item['content'] = i['templateMaterial']['widgetTitle']

                    time = int(int(i['templateMaterial']['publishTime']) / 1000)
                    item['datetime'] = datetime.datetime.fromtimestamp(time).strftime("%Y-%m-%d %H:%M:%S")
                    self.items.append(item)
            # print(self.items)
            return self.items

        except Exception as e:
            print(f"36氪新闻获取失败: {str(e)}")
            return []


if __name__ == '__main__':
    kr = Kr36Spider()
    asyncio.run(kr.fetch_news())
