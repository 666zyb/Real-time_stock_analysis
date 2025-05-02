import tushare as ts

# 设置你的 Tushare token
ts.set_token('da57da51ad72c583b99580b366421827bcb1b088337c683ae95814a2')
# 初始化 pro 接口
pro = ts.pro_api()

df = pro.bak_basic(trade_date='20211012',ts_code="000001.SZ", fields='trade_date,ts_code,name,industry,pe')
print(df)