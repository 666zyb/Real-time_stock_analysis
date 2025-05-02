from django.db import models


class Stock(models.Model):
    """股票基本信息模型"""
    code = models.CharField('股票代码', max_length=10, primary_key=True)
    name = models.CharField('股票名称', max_length=50)
    industry = models.CharField('行业', max_length=50, blank=True)

    class Meta:
        verbose_name = '股票'
        verbose_name_plural = '股票列表'

    def __str__(self):
        return f"{self.name}({self.code})"


class StockRealTimeData(models.Model):
    """股票实时数据模型"""
    stock = models.ForeignKey(Stock, on_delete=models.CASCADE, related_name='realtime_data')
    time = models.DateTimeField('时间')
    open_price = models.FloatField('今日开盘价')
    last_close = models.FloatField('昨日收盘价')
    current_price = models.FloatField('当前价格')
    low_price = models.FloatField('今日最低价')
    volume = models.BigIntegerField('成交量(手)')
    amount = models.FloatField('成交额(元)')

    class Meta:
        verbose_name = '实时数据'
        verbose_name_plural = '实时数据列表'
        unique_together = ('stock', 'time')
        ordering = ['-time']

    def __str__(self):
        return f"{self.stock.name} - {self.time}"


class StockHistoryData(models.Model):
    """股票历史数据模型"""
    id = models.AutoField(primary_key=True)  # 显式定义主键
    stock = models.ForeignKey(Stock, on_delete=models.CASCADE, related_name='history_data')
    date = models.DateField('日期')
    open_price = models.FloatField('开盘价')
    close_price = models.FloatField('收盘价')
    high_price = models.FloatField('最高价')
    low_price = models.FloatField('最低价')
    volume = models.BigIntegerField('成交量(手)')
    amount = models.FloatField('成交额(元)')
    change_percent = models.FloatField('涨跌幅(%)')

    class Meta:
        verbose_name = '历史数据'
        verbose_name_plural = '历史数据列表'
        unique_together = ('stock', 'date')
        ordering = ['-date']

    def __str__(self):
        return f"{self.stock.name} - {self.date}"


class NewsSource(models.Model):
    """新闻来源模型"""
    name = models.CharField('来源名称', max_length=50, primary_key=True)

    class Meta:
        verbose_name = '新闻来源'
        verbose_name_plural = '新闻来源列表'

    def __str__(self):
        return self.name


class News(models.Model):
    """新闻模型"""
    source = models.ForeignKey(NewsSource, on_delete=models.CASCADE, related_name='news')
    content = models.TextField('内容')
    pub_time = models.DateTimeField('发布时间')

    class Meta:
        verbose_name = '新闻'
        verbose_name_plural = '新闻列表'
        ordering = ['-pub_time']

    def __str__(self):
        return f"{self.content[:50]}... ({self.source.name})"


class StockHistory(models.Model):
    stock_code = models.CharField(max_length=20)
    date = models.DateField()
    open_price = models.FloatField()
    close_price = models.FloatField()
    high_price = models.FloatField()
    low_price = models.FloatField()
    volume = models.IntegerField(db_column='成交量(手)')
    amount = models.FloatField(db_column='成交额(元)')

    class Meta:
        db_table = '东阿阿胶_history'  # 替换为实际的表名