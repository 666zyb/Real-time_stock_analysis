import requests
import time
import json
import re

class Sina_stock():
    def __init__(self):
        self.headers = {
            "Accept": "*/*",
            "Accept-Language": "zh-CN,zh;q=0.9",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Pragma": "no-cache",
            "Referer": "https://finance.sina.com.cn/realstock/company/sh688048/nc.shtml",
            "Sec-Fetch-Dest": "script",
            "Sec-Fetch-Mode": "no-cors",
            "Sec-Fetch-Site": "cross-site",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 SLBrowser/9.0.6.2081 SLBChan/105 SLBVPV/64-bit",
            "^sec-ch-ua": "^\\^Chromium^^;v=^\\^9^^, ^\\^Not?A_Brand^^;v=^\\^8^^^",
            "sec-ch-ua-mobile": "?0",
            "^sec-ch-ua-platform": "^\\^Windows^^^"
        }

        self.timestamp = int(time.time()) * 100
        self.config_path = '../config/config.json'
        self.load_config(self.config_path)

        self.url ="https://hq.sinajs.cn/rn={}&list={},{}_i,bk_new_qtxy"
    def load_config(self,config_path: str):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)

    def format_stock_code(self, code: str) -> str:
        """格式化股票代码"""
        if not code.startswith(('sh', 'sz')):
            if code.startswith('6'):
                return f'sh{code}'
            elif code.startswith(('0', '3')):
                return f'sz{code}'
        return code

    def parse_sina_stock_data(self,data_str):
        # 分割字段并去除首尾空值
        fields = data_str.strip().split(',')
        # 构造字典（字段顺序需与新浪财经定义一致）
        stock_info = {
            "名称": fields[0],
            "今日开盘价": float(fields[1]),
            "昨日收盘价": float(fields[2]),
            "当前价格": float(fields[3]),
            "今日最高价": float(fields[4]),
            "今日最低价": float(fields[5]),
            "竞买价": float(fields[6]),
            "竞卖价": float(fields[7]),
            "成交量(手)": int(fields[8]),
            "成交额(元)": float(fields[9]),
            "买一委托量": int(fields[10]),
            "买一报价": float(fields[11]),
            "买二委托量": int(fields[12]),
            "买二报价": float(fields[13]),
            "买三委托量": int(fields[14]),
            "买三报价": float(fields[15]),
            "买四委托量": int(fields[16]),
            "买四报价": float(fields[17]),
            "买五委托量": int(fields[18]),
            "买五报价": float(fields[19]),
            "卖一委托量": int(fields[20]),
            "卖一报价": float(fields[21]),
            "卖二委托量": int(fields[22]),
            "卖二报价": float(fields[23]),
            "卖三委托量": int(fields[24]),
            "卖三报价": float(fields[25]),
            "卖四委托量": int(fields[26]),
            "卖四报价": float(fields[27]),
            "卖五委托量": int(fields[28]),
            "卖五报价": float(fields[29]),
            "日期": fields[30],
            "时间": fields[31],
            "其他保留字段": fields[32]
        }
        return stock_info
    def get_stock_data(self,code):
        try:
            response = requests.get(self.url.format(self.timestamp,code,code), headers=self.headers)
            # 使用正则表达式匹配第一个var开头的数据
            pattern = re.compile(r'var\s+hq_str_\w+="([^"]+)"')
            match = pattern.search(response.text)
            if match:
                first_data = match.group(1)
                result = self.parse_sina_stock_data(first_data)
                # print(result)
            else:
                print("未找到匹配的数据。")
            return result
        except Exception as e:
            print(f"股票数据获取失败: {str(e)}")
            return []  # 出错时返回空列表

if __name__=="__main__":
    sina=Sina_stock()
    sina.get_stock_data('sh600362')


