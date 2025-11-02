import os
import json
import time
import redis
import mysql.connector
from django.shortcuts import render, get_object_or_404
from django.http import JsonResponse
from django.db import connection
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt
from .models import Stock, StockRealTimeData, News
from .services.chart_service import ChartService
from django.conf import settings
from django.core.paginator import Paginator
from datetime import datetime, timedelta
import pandas as pd
from decimal import Decimal
import logging
from News_analysis.news_stock_analysis import NewsStockAnalyzer
import requests
import akshare as ak
import traceback
import subprocess

# 配置日志
logger = logging.getLogger(__name__)


def get_stocks_from_config():
    """
    从配置文件中获取股票列表，包括主要股票和其他股票
    """
    try:
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config', 'config.json')
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            # 获取主要股票和其他股票
            main_stocks = config.get('stocks', [])
            other_stocks = config.get('other_stocks', [])
            # 返回合并后的列表
            return main_stocks + other_stocks
    except Exception as e:
        logging.error(f"从配置文件中获取股票列表失败: {str(e)}")
        return []


def index(request):
    """主页视图"""
    # 只加载主要股票，不包括other_stocks
    with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config', 'config.json'), 'r', encoding='utf-8') as f:
        config = json.load(f)
        stocks = config.get('stocks', [])
    return render(request, 'index.html', {'stocks': stocks})


def stock_list(request):
    """股票列表视图"""
    # 只加载主要股票，不包括other_stocks
    with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config', 'config.json'), 'r', encoding='utf-8') as f:
        config = json.load(f)
        stocks = config.get('stocks', [])

    # 获取这些股票的实时数据（如果需要）
    from .services.stock_service import StockDataService
    stock_service = StockDataService()

    # 添加实时数据
    stock_data = []
    for stock in stocks:
        stock_info = {
            'code': stock['code'],
            'name': stock['name'],
            'industry': stock.get('industry', '')
        }

        try:
            # 获取实时数据
            formatted_code = stock_service.format_stock_code(stock['code'])
            realtime_data = stock_service.get_realtime_data_sync(formatted_code)
            if realtime_data:
                stock_info.update(realtime_data)
        except Exception as e:
            print(f"获取{stock['code']}实时数据失败: {e}")

        stock_data.append(stock_info)

    return render(request, 'stock_list.html', {'stocks': stock_data})


def load_config():
    config_path = os.path.join(settings.BASE_DIR, 'config', 'config.json')
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config
    except Exception as e:
        print(f"读取配置文件出错: {e}")
        return {}


def save_config(config):
    config_path = os.path.join(settings.BASE_DIR, 'config', 'config.json')
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"保存配置文件出错: {e}")
        return False


def get_stock_name(stock_code):
    config = load_config()
    stock_info = next((s for s in config.get('stocks', []) if s['code'] == stock_code), None)
    return stock_info['name'] if stock_info else None


def get_stock_history(stock_name, limit=500):
    """获取股票历史数据"""
    logger.info(f"尝试获取股票 {stock_name} 的历史数据，限制 {limit} 条")
    table_name = f"{stock_name}_history"  # 生成表名
    query = f"""
        SELECT 
            `日期` AS date, 
            `开盘价` AS open_price, 
            `收盘价` AS close_price, 
            `最高价` AS high_price, 
            `最低价` AS low_price, 
            `成交量(手)` AS volume, 
            `成交额(元)` AS amount,
            `涨跌幅(%)` AS change_percent,
            `涨跌额(元)` AS change_amount,
            `换手率(%)` AS turnover_rate
        FROM {table_name} 
        ORDER BY `日期` DESC 
        LIMIT {limit};  # 获取最近的数据
    """

    try:
        # 首先检查表是否存在
        check_table_query = f"""
        SELECT COUNT(*) 
        FROM information_schema.tables 
        WHERE table_schema = DATABASE() 
        AND table_name = '{table_name}'
        """

        with connection.cursor() as cursor:
            cursor.execute(check_table_query)
            table_exists = cursor.fetchone()[0] > 0

            if not table_exists:
                logger.error(f"表 {table_name} 不存在")
                return None

            # 如果表存在，执行查询
            cursor.execute(query)
            rows = cursor.fetchall()
            logger.info(f"查询到 {len(rows)} 条数据")

            if not rows:
                logger.warning(f"表 {table_name} 中没有数据")
                return None

        # 将查询结果转换为 DataFrame
        df = pd.DataFrame(rows, columns=['date', 'open_price', 'close_price', 'high_price', 'low_price', 'volume', 'amount', 'change_percent', 'change_amount', 'turnover_rate'])
        return df
    except Exception as e:
        logger.error(f"查询失败: {e}, SQL: {query}")
        logger.error(traceback.format_exc())
        return None


def stock_detail(request, stock_code):
    stocks = get_stocks_from_config()
    stock = next((s for s in stocks if s['code'] == stock_code), None)

    if not stock:
        from django.http import Http404
        logger.error(f"股票 {stock_code} 不存在于配置中")
        raise Http404(f"股票 {stock_code} 不存在")

    # 获取最近500条历史数据
    try:
        logger.info(f"尝试获取股票 {stock['name']}({stock_code}) 的历史数据")
        history_data = get_stock_history(stock['name'], limit=500)

        if history_data is None or history_data.empty:
            error_message = f"无法获取股票 {stock['name']}({stock_code}) 的历史数据。请确保已添加历史数据，或尝试重新添加该股票。"
            logger.error(error_message)
            return render(request, 'error.html', {'message': error_message})

        logger.info(f"成功获取股票 {stock['name']}({stock_code}) 的历史数据，共 {len(history_data)} 条记录")
    except Exception as e:
        error_message = f"获取股票 {stock['name']}({stock_code}) 的历史数据时发生错误: {str(e)}"
        logger.error(error_message)
        logger.error(traceback.format_exc())
        return render(request, 'error.html', {'message': error_message})

    # 获取最新实时数据
    from .services.stock_service import StockDataService
    stock_service = StockDataService()
    formatted_code = stock_service.format_stock_code(stock_code)
    logger.info(f"格式化股票代码: {stock_code} -> {formatted_code}")

    # 初始化变量，确保在所有执行路径中都有定义
    latest_data = None
    change = 0
    change_percent = 0

    # 尝试获取最新数据
    try:
        with connection.cursor() as cursor:
            # 查询实时数据表
            table_name = f"stock_{formatted_code}_realtime"

            # 检查表是否存在
            check_table_query = f"""
            SELECT COUNT(*) 
            FROM information_schema.tables 
            WHERE table_schema = DATABASE() 
            AND table_name = '{table_name}'
            """
            cursor.execute(check_table_query)
            table_exists = cursor.fetchone()[0] > 0

            if not table_exists:
                logger.warning(f"实时数据表 {table_name} 不存在")
            elif table_exists:
                logger.info(f"实时数据表 {table_name} 存在，尝试获取最新数据")
                # 获取最新一条记录
                latest_query = f"""
                SELECT * FROM {table_name} 
                ORDER BY `日期` DESC, `时间` DESC 
                LIMIT 1
                """
                cursor.execute(latest_query)
                row = cursor.fetchone()

                if row:
                    # 获取列名
                    columns = [col[0] for col in cursor.description]
                    latest_data_raw = dict(zip(columns, row))
                    logger.info(f"获取到最新实时数据，时间: {latest_data_raw.get('日期')} {latest_data_raw.get('时间')}")

                    # 格式化数据
                    latest_data = {
                        'current_price': float(latest_data_raw['当前价格']),
                        'open_price': float(latest_data_raw['今日开盘价']),
                        'last_close': float(latest_data_raw['昨日收盘价']),
                        'low_price': float(latest_data_raw['今日最低价']),
                        'volume': int(float(latest_data_raw['成交量(手)'])),
                        'amount': float(latest_data_raw['成交额(元)']),
                        'time': latest_data_raw['时间']
                    }

                    # 计算涨跌和涨跌幅
                    change = latest_data['current_price'] - latest_data['last_close']
                    change_percent = (change / latest_data['last_close']) * 100 if latest_data['last_close'] else 0
                else:
                    logger.warning(f"未找到实时数据记录")

    except Exception as e:
        logger.error(f"获取最新数据时出错: {str(e)}")
        logger.error(traceback.format_exc())

    # 确保date列是日期时间类型
    history_data['date'] = pd.to_datetime(history_data['date'])
    logger.info(f"日期范围: {history_data['date'].min()} 至 {history_data['date'].max()}")

    # 将数据转换为JSON格式，使用小数点后两位
    def convert_to_json_safe(value):
        if isinstance(value, Decimal):
            return float(round(value, 2))
        return value

    # 按日期升序排序数据
    history_data_sorted = history_data.sort_values('date')

    chart_data = {
        'dates': history_data_sorted['date'].dt.strftime('%Y-%m-%d').tolist(),
        'open': [convert_to_json_safe(x) for x in history_data_sorted['open_price'].tolist()],
        'close': [convert_to_json_safe(x) for x in history_data_sorted['close_price'].tolist()],
        'high': [convert_to_json_safe(x) for x in history_data_sorted['high_price'].tolist()],
        'low': [convert_to_json_safe(x) for x in history_data_sorted['low_price'].tolist()],
        'volume': [int(x) if isinstance(x, Decimal) else x for x in history_data_sorted['volume'].tolist()]
    }

    context = {
        'stock': stock,
        'history_data': history_data.sort_values('date', ascending=False).head(20),  # 仅显示最新20条
        'latest_data': latest_data,
        'change': change,
        'change_percent': change_percent,
        'chart_data': json.dumps(chart_data)
    }

    logger.info(f"渲染股票 {stock['name']}({stock_code}) 详情页面")
    return render(request, 'stock_detail.html', context)


def news_list(request):
    """新闻列表页面视图"""
    return render(request, 'news_list.html')


@require_http_methods(["GET"])
def                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        api_stock_data(request, stock_code=None):
    """获取股票数据API（从MySQL数据库）"""
    try:
        # 加载配置
        with open(os.path.join(settings.BASE_DIR, 'config', 'config.json'), 'r', encoding='utf-8') as f:
            config = json.load(f)

        # 连接到MySQL
        mysql_config = config.get('mysql_config', {})
        conn = mysql.connector.connect(
            host=mysql_config.get('host', '172.16.0.2'),
            port=mysql_config.get('port', 3306),
            user=mysql_config.get('user', 'root'),
            password=mysql_config.get('password', ''),
            database=mysql_config.get('database', 'stock_analysis')
        )
        cursor = conn.cursor(dictionary=True)

        result = []

        if stock_code:
            # 获取指定股票的数据
            formatted_code = format_stock_code(stock_code)
            table_name = f"stock_{formatted_code}_realtime"

            try:
                query = f"SELECT * FROM {table_name} ORDER BY `时间` DESC LIMIT 1"
                cursor.execute(query)
                data = cursor.fetchone()

                if data:
                    # 获取股票名称和行业
                    # 只从main_stocks中获取股票信息
                    stock_info = next((s for s in config.get('stocks', []) if s['code'] == stock_code), None)
                    name = stock_info['name'] if stock_info else 'Unknown'
                    industry = stock_info.get('industry', '') if stock_info else ''

                    # 计算涨跌和涨跌幅
                    current_price = float(data['当前价格'])
                    last_close = float(data['昨日收盘价'])
                    change = current_price - last_close
                    change_percent = (change / last_close) * 100 if last_close else 0

                    result.append({
                        'code': stock_code,
                        'name': name,
                        'industry': industry,
                        'current_price': current_price,
                        'last_close': last_close,
                        'low_price': float(data['今日最低价']),
                        'change': change,
                        'change_percent': change_percent,
                        'volume': int(data['成交量(手)']),
                        'amount': float(data['成交额(元)']),
                        'time': data['时间']
                    })
            except Exception as e:
                print(f"获取股票 {stock_code} 数据时出错: {str(e)}")
        else:
            # 获取所有股票的数据，但只包括main_stocks中的股票
            # 只遍历main_stocks中的股票
            for stock_info in config.get('stocks', []):
                try:
                    formatted_code = format_stock_code(stock_info['code'])
                    table_name = f"stock_{formatted_code}_realtime"

                    query = f"SELECT * FROM {table_name} ORDER BY `时间` DESC LIMIT 1"
                    cursor.execute(query)
                    data = cursor.fetchone()

                    if data:
                        # 计算涨跌和涨跌幅
                        current_price = float(data['当前价格'])
                        last_close = float(data['昨日收盘价'])
                        change = current_price - last_close
                        change_percent = (change / last_close) * 100 if last_close else 0

                        result.append({
                            'code': stock_info['code'],
                            'name': stock_info['name'],
                            'industry': stock_info.get('industry', ''),
                            'current_price': current_price,
                            'last_close': last_close,
                            'low_price': float(data['今日最低价']),
                            'change': change,
                            'change_percent': change_percent,
                            'volume': int(data['成交量(手)']),
                            'amount': float(data['成交额(元)']),
                            'time': data['时间']
                        })
                except Exception as e:
                    print(f"获取股票 {stock_info['code']} 数据时出错: {str(e)}")

        cursor.close()
        conn.close()

        return JsonResponse({
            'status': 'success',
            'data': result
        })

    except Exception as e:
        print(f"获取股票数据时出错: {str(e)}")
        return JsonResponse({
            'status': 'error',
            'message': str(e),
            'data': []
        })


def format_stock_code(code: str) -> str:
    """格式化股票代码"""
    if not code.startswith(('sh', 'sz')):
        if code.startswith('6'):
            return f'sh{code}'
        elif code.startswith(('0', '3')):
            return f'sz{code}'
    return code


@require_http_methods(["GET"])
def api_news_data(request):
    """
    从Redis获取新闻数据并分页
    参数：
    - page: 页码，默认1
    - page_size: 每页条数，默认10
    - source: 新闻来源，默认all
    """
    try:
        # 获取分页参数
        page = int(request.GET.get('page', 1))
        page_size = int(request.GET.get('page_size', 10))
        source = request.GET.get('source', 'all')

        # 加载Redis配置
        try:
            with open(os.path.join(settings.BASE_DIR, 'config', 'config.json'), 'r', encoding='utf-8') as f:
                config = json.load(f)
                redis_config = config.get('redis_config', {})
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            redis_config = {
                'host': '127.0.0.1',
                'port': 6379,
                'db': 0,
                'password': None
            }

        # 连接Redis
        try:
            redis_client = redis.Redis(
                host=redis_config.get('host', '127.0.0.1'),
                port=redis_config.get('port', 6379),
                db=redis_config.get('db', 0),
                password=redis_config.get('password'),
                decode_responses=True
            )

            # 测试连接
            redis_client.ping()
            logger.info("Redis连接成功")
        except Exception as e:
            logger.error(f"Redis连接失败: {e}")
            return JsonResponse({
                'status': 'error',
                'message': f'Redis连接失败: {str(e)}',
                'data': [],
                'pagination': {
                    'page': page,
                    'page_size': page_size,
                    'total_count': 0,
                    'total_pages': 0
                }
            })

        # 创建分析器实例，用于获取新闻分析结果
        analyzer = NewsStockAnalyzer(os.path.join(settings.BASE_DIR, 'config', 'config.json'))

        # 获取新闻数据
        hot_news_key = "stock:hot_news"
        if not redis_client.exists(hot_news_key):
            logger.error(f"Redis键不存在: {hot_news_key}")
            return JsonResponse({
                'status': 'error',
                'message': f'Redis键不存在: {hot_news_key}',
                'data': [],
                'pagination': {
                    'page': page,
                    'page_size': page_size,
                    'total_count': 0,
                    'total_pages': 0
                }
            })

        # 获取全部数据
        total_count = redis_client.llen(hot_news_key)
        all_news_raw = redis_client.lrange(hot_news_key, 0, -1)
        logger.info(f"从Redis获取到{len(all_news_raw)}条新闻数据")

        # 处理原始数据
        all_news = []
        for idx, news_item in enumerate(all_news_raw):
            try:
                news_data = json.loads(news_item)

                # 过滤新闻来源
                if source != 'all':
                    content = news_data.get('content', '').lower()
                    if source == '新浪财经' and not ('新浪' in content or '微博' in content):
                        continue
                    elif source == '同花顺' and not '同花顺' in content:
                        continue
                    elif source == '36氪' and not ('36氪' in content or '36kr' in content):
                        continue
                    elif source == '财联社' and not '财联社' in content:
                        continue

                # 推断新闻来源
                source_name = '其他'
                content = news_data.get('content', '').lower()

                if '新浪' in content or '微博' in content:
                    source_name = '新浪财经'
                elif '同花顺' in content:
                    source_name = '同花顺'
                elif '36氪' in content or '36kr' in content:
                    source_name = '36氪'
                elif '财联社' in content:
                    source_name = '财联社'

                # 获取新闻分析结果
                news_hash = analyzer.generate_news_hash(news_data)
                analysis_result = redis_client.hget("stock:news_all_analyses", news_hash)
                if analysis_result:
                    try:
                        news_item_obj = {
                            'id': idx,
                            'content': news_data.get('content', ''),
                            'source': source_name,
                            'pub_time': news_data.get('datetime', ''),
                            'datetime': news_data.get('datetime', ''),
                            'color': news_data.get('color', ''),  # 添加color字段
                            'analysis_result': json.loads(analysis_result)
                        }
                    except Exception as e:
                        logger.error(f"解析分析结果出错: {str(e)}")
                        news_item_obj['analysis_result'] = None
                else:
                    news_item_obj = {
                        'id': idx,
                        'content': news_data.get('content', ''),
                        'source': source_name,
                        'pub_time': news_data.get('datetime', ''),
                        'datetime': news_data.get('datetime', ''),
                        'color': news_data.get('color', ''),  # 添加color字段
                        'analysis_result': None
                    }

                all_news.append(news_item_obj)
            except Exception as e:
                logger.error(f"处理第{idx}条新闻数据出错: {e}")
                continue

        # 计算分页
        total_pages = (len(all_news) + page_size - 1) // page_size

        # 确保页码有效
        if page < 1:
            page = 1
        if page > total_pages and total_pages > 0:
            page = total_pages

        # 分页处理
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        page_news = all_news[start_idx:end_idx]

        # 处理结果前记录几条数据进行调试
        if page_news:
            logger.info(f"API返回的前两条新闻数据: {page_news[:2]}")

        return JsonResponse({
            'status': 'success',
            'data': page_news,
            'pagination': {
                'page': page,
                'page_size': page_size,
                'total_count': len(all_news),
                'total_pages': total_pages
            }
        })

    except Exception as e:
        logger.error(f"获取新闻API数据时出错: {str(e)}")
        return JsonResponse({
            'status': 'error',
            'message': f'处理请求时出错: {str(e)}',
            'data': [],
            'pagination': {
                'page': 1,
                'page_size': 10,
                'total_count': 0,
                'total_pages': 0
            }
        })


def generate_mock_news_data():
    """生成模拟新闻数据（当Redis无法访问时使用）"""
    mock_data = []
    sources = ['新浪财经', '同花顺', '36氪', '财联社']

    for i in range(10):
        now = datetime.now() - timedelta(minutes=i * 5)
        mock_data.append({
            'content': f'模拟新闻数据 #{i + 1}: 由于无法连接Redis或获取真实数据，显示此模拟内容。请检查Redis连接和配置。',
            'source': sources[i % len(sources)],
            'pub_time': now.strftime('%Y-%m-%d %H:%M:%S')
        })

    return mock_data


def get_prev_close_from_history(stock_code):
    """从历史数据表获取昨收价 - 修正版本"""
    try:
        # 从配置文件获取股票名称
        config = load_config()
        stocks = config.get('stocks', [])
        stock_info = next((s for s in stocks if s['code'] == stock_code), None)

        if not stock_info:
            print(f"在配置中找不到股票代码: {stock_code}")
            return None

        stock_name = stock_info['name']
        table_name = f"{stock_name}_history"
        print(f"查询历史表获取昨收价: {table_name}")

        # 检查表是否存在
        with connection.cursor() as cursor:
            check_table_query = f"""
            SELECT COUNT(*) 
            FROM information_schema.tables 
            WHERE table_schema = DATABASE() 
            AND table_name = '{table_name}'
            """
            cursor.execute(check_table_query)
            table_exists = cursor.fetchone()[0] > 0

            if not table_exists:
                print(f"历史数据表 {table_name} 不存在")
                return None

            # 查询最近一个交易日的数据
            query = f"""
            SELECT `日期`, `收盘价` 
            FROM {table_name} 
            WHERE `日期` < CURDATE() 
            ORDER BY `日期` DESC 
            LIMIT 1
            """
            cursor.execute(query)
            result = cursor.fetchone()

            if result:
                date, prev_close = result
                prev_close = float(prev_close)
                print(f"从历史表获取到昨收价: {prev_close} (日期: {date})")
                return prev_close
            else:
                print(f"历史表 {table_name} 中没有找到昨收价数据")
                return None

    except Exception as e:
        print(f"从历史表获取昨收价失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def get_prev_close_from_api(stock_code):
    """从API获取昨收价"""
    try:
        import akshare as ak

        # 使用akshare获取股票基本信息
        stock_info = ak.stock_individual_info_em(symbol=stock_code)
        if not stock_info.empty:
            # 查找昨收价
            for _, row in stock_info.iterrows():
                if row['item'] == '昨收':
                    return float(row['value'])

        return None
    except Exception as e:
        print(f"从API获取昨收价失败: {e}")
        return None


def is_trading_time():
    """判断当前是否为A股交易时间"""
    from datetime import datetime

    now = datetime.now()
    weekday = now.weekday()  # 0-4为工作日，5-6为周末

    # 周末非交易
    if weekday >= 5:
        return False

    # 法定节假日判断（这里需要更复杂的逻辑，暂时简单处理）
    # 可以接入节假日API或者维护一个节假日列表

    current_time = now.time()

    # 上午交易时段: 9:30-11:30
    morning_start = datetime.strptime('09:30', '%H:%M').time()
    morning_end = datetime.strptime('11:30', '%H:%M').time()

    # 下午交易时段: 13:00-15:00
    afternoon_start = datetime.strptime('13:00', '%H:%M').time()
    afternoon_end = datetime.strptime('15:00', '%H:%M').time()

    return ((morning_start <= current_time <= morning_end) or
            (afternoon_start <= current_time <= afternoon_end))


def get_realtime_data(request, stock_code):
    """获取分时数据 - 修复时间格式问题"""
    try:
        # 1. 检查akshare是否可用
        try:
            import akshare as ak
        except ImportError:
            return JsonResponse({
                'status': 'error',
                'message': '请先安装akshare: pip install akshare'
            })

        # 2. 检查股票代码格式
        if not stock_code or len(stock_code) != 6:
            return JsonResponse({
                'status': 'error',
                'message': f'股票代码格式错误: {stock_code}，应为6位数字'
            })

        # 3. 获取昨收价
        prev_close = get_prev_close_from_history(stock_code)
        if prev_close is None:
            prev_close = get_prev_close_from_api(stock_code)

        if prev_close is None:
            return JsonResponse({
                'status': 'error',
                'message': '无法获取基准价格数据，请检查股票代码是否正确'
            })

        print(f"获取到昨收价: {prev_close}")

        # 4. 获取当日分时数据
        today = datetime.now().strftime('%Y%m%d')
        print(f"尝试获取 {stock_code} 在 {today} 的分时数据")

        try:
            stock_data = ak.stock_zh_a_hist_min_em(
                symbol=stock_code,
                period='1',
                start_date=today,
                end_date=today,
                adjust=''
            )
        except Exception as e:
            print(f"akshare获取数据失败: {e}")
            return JsonResponse({
                'status': 'error',
                'message': f'获取分时数据失败: {str(e)}'
            })

        if stock_data.empty:
            print(f"未获取到 {stock_code} 在 {today} 的分时数据")
            # 尝试获取最近交易日的分时数据
            yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y%m%d')
            print(f"尝试获取昨日 {yesterday} 的数据")

            try:
                stock_data = ak.stock_zh_a_hist_min_em(
                    symbol=stock_code,
                    period='1',
                    start_date=yesterday,
                    end_date=yesterday,
                    adjust=''
                )
            except Exception as e:
                print(f"获取昨日数据也失败: {e}")
                return JsonResponse({
                    'status': 'error',
                    'message': f'获取历史分时数据失败: {str(e)}'
                })

            if stock_data.empty:
                return JsonResponse({
                    'status': 'error',
                    'message': '当前无交易数据，请确认股票代码是否正确且市场在交易中'
                })

        # 5. 处理分时数据 - 修复时间格式问题
        times = []
        prices = []
        volumes = []

        print(f"获取到的数据列: {stock_data.columns.tolist()}")
        print(f"数据样例:\n{stock_data.head()}")

        for _, row in stock_data.iterrows():
            # 处理时间字段 - 直接使用字符串，不需要strftime
            time_str = str(row['时间'])

            # 如果时间包含日期部分，只取时间部分
            if ' ' in time_str:
                time_str = time_str.split(' ')[1]

            # 如果时间包含秒，只取到分钟
            if ':' in time_str and time_str.count(':') == 2:
                time_str = ':'.join(time_str.split(':')[:2])

            times.append(time_str)

            # 处理价格和成交量
            try:
                prices.append(float(row['收盘']))
                volumes.append(int(row['成交量']))
            except Exception as e:
                print(f"处理数据行时出错: {e}, 行数据: {row}")
                continue

        print(f"成功获取 {len(times)} 条分时数据")
        print(f"时间范围: {times[0]} 到 {times[-1]}")
        print(f"价格范围: {min(prices) if prices else 'N/A'} 到 {max(prices) if prices else 'N/A'}")

        return JsonResponse({
            'status': 'success',
            'data': {
                'times': times,
                'prices': prices,
                'volumes': volumes,
                'prev_close': prev_close
            }
        })

    except Exception as e:
        print(f"获取分时数据失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return JsonResponse({
            'status': 'error',
            'message': f'获取数据失败: {str(e)}'
        })


def get_stock_industry(stock_code):
    """
    使用akshare和东方财富API获取股票行业信息
    返回行业信息或者空字符串（如果获取失败）
    """
    logger.info(f"尝试获取股票 {stock_code} 的行业信息")

    # 尝试使用akshare获取
    industry_info = get_stock_industry_from_akshare(stock_code)
    if industry_info:
        logger.info(f"通过akshare成功获取到股票 {stock_code} 的行业信息: {industry_info}")
        return industry_info

    # 如果akshare获取失败，尝试使用东方财富API获取
    logger.warning(f"无法通过akshare获取股票 {stock_code} 的行业信息，尝试使用东方财富API")
    industry_info = get_stock_industry_from_eastmoney(stock_code)
    if industry_info:
        logger.info(f"通过东方财富API成功获取到股票 {stock_code} 的行业信息: {industry_info}")
        return industry_info

    logger.error(f"所有方法都无法获取股票 {stock_code} 的行业信息")
    return ""


def get_stock_industry_from_akshare(stock_code):
    """
    使用akshare获取股票行业信息
    """
    try:
        # 根据股票代码前缀判断交易所
        if stock_code.startswith('6'):
            stock_with_suffix = f"{stock_code}.SH"  # 上海证券交易所
        elif stock_code.startswith(('0', '3')):
            stock_with_suffix = f"{stock_code}.SZ"  # 深圳证券交易所
        else:
            logger.warning(f"股票代码 {stock_code} 格式不正确，无法判断交易所")
            return ""

        logger.info(f"使用akshare查询股票 {stock_with_suffix} 的行业信息")

        # 获取沪深个股行业数据
        stock_industry_df = ak.stock_individual_info_em(symbol=stock_with_suffix)

        # 输出完整的DataFrame内容，用于调试
        logger.info(f"akshare返回的DataFrame:\n{stock_industry_df.to_string()}")

        # 查找行业信息
        if not stock_industry_df.empty:
            for index, row in stock_industry_df.iterrows():
                logger.info(f"检查行 {index}: {row[0]} = {row[1]}")
                if row[0] == "行业":
                    if row[1] and str(row[1]) != "nan":
                        return str(row[1])
        else:
            logger.warning(f"akshare返回的DataFrame为空")

        return ""
    except Exception as e:
        logger.error(f"通过akshare获取股票 {stock_code} 行业信息失败: {str(e)}")
        logger.error(traceback.format_exc())
        return ""


def get_stock_industry_from_eastmoney(stock_code):
    """
    使用东方财富API获取股票行业信息
    """
    try:
        # 构建请求URL
        url = "http://push2.eastmoney.com/api/qt/stock/get"
        params = {
            "secid": f"{'1' if stock_code.startswith('6') else '0'}.{stock_code}",
            "ut": "fa5fd1943c7b386f172d6893dbfba10b",
            "fields": "f127,f128,f129,f130",  # 行业相关字段
            "invt": "2",
            "fltt": "2",
            "cb": "jQuery1124006726408459518854_1591696165952",
            "_": "1591696165954"
        }

        logger.info(f"使用东方财富API查询股票 {stock_code} 的行业信息")

        response = requests.get(url, params=params)
        # 提取JSON部分
        json_str = response.text
        start = json_str.find('{')
        end = json_str.rfind('}') + 1
        json_data = json.loads(json_str[start:end])

        logger.info(f"东方财富API返回数据: {json_data}")

        # 提取行业信息
        data = json_data.get('data', {})
        industry = data.get('f127', '')  # 行业

        return industry
    except Exception as e:
        logger.error(f"通过东方财富API获取股票 {stock_code} 行业信息失败: {str(e)}")
        logger.error(traceback.format_exc())
        return ""


@csrf_exempt
@require_http_methods(["GET"])
def search_stock(request):
    """搜索股票并返回结果"""
    keyword = request.GET.get('keyword', '')
    if not keyword:
        return JsonResponse({'status': 'error', 'message': '请输入搜索关键词'})

    try:
        # 使用东方财富的搜索API
        url = f"http://searchapi.eastmoney.com/api/suggest/get"
        params = {
            'input': keyword,
            'type': '14',
            'token': 'D43BF722C8E33BDC906FB84D85E326E8',
            'count': '10'
        }

        response = requests.get(url, params=params)
        data = response.json()

        if data.get('QuotationCodeTable', {}).get('Data'):
            results = data['QuotationCodeTable']['Data']
            stocks = []

            for item in results:
                # 只处理A股股票 (上海和深圳交易所)
                if item.get('SecurityType') in ['1', '2']:  # 1为上海，2为深圳
                    stock_code = item.get('Code', '')
                    stock_name = item.get('Name', '')
                    exchange = '上海' if item.get('SecurityType') == '1' else '深圳'

                    # 转换为6位股票代码
                    if stock_code and len(stock_code) == 6:
                        # 尝试获取行业信息
                        industry = ""
                        try:
                            industry = get_stock_industry(stock_code)
                        except Exception as e:
                            logger.error(f"获取 {stock_code} 行业信息失败: {str(e)}")
                            industry = item.get('MktName', '')  # 如果获取失败，使用原有的市场名称

                        # 如果行业信息为空，使用市场名称作为备用
                        if not industry:
                            industry = item.get('MktName', '')

                        stocks.append({
                            'code': stock_code,
                            'name': stock_name,
                            'industry': industry,
                            'exchange': exchange
                        })

            return JsonResponse({
                'status': 'success',
                'data': stocks
            })
        else:
            return JsonResponse({'status': 'error', 'message': '未找到相关股票'})

    except Exception as e:
        logger.error(f"搜索股票出错: {str(e)}")
        logger.error(traceback.format_exc())
        return JsonResponse({'status': 'error', 'message': f'搜索出错: {str(e)}'})


def delete_stock_data_from_database(stock_info):
    """删除指定股票的历史数据表和实时数据表，技术指标数据表，实时技术指标数据表"""
    try:
        # 获取MySQL配置
        config = load_config()
        mysql_config = config.get('mysql_config', {})

        # 连接到MySQL数据库
        conn = mysql.connector.connect(
            host=mysql_config.get('host', '127.0.0.1'),
            port=mysql_config.get('port', 3306),
            user=mysql_config.get('user', 'root'),
            password=mysql_config.get('password', ''),
            database=mysql_config.get('database', 'stock_analysis')
        )
        cursor = conn.cursor()

        success = True
        # 删除历史数据表
        table_name = f"{stock_info['name']}_history"
        try:
            # 检查表是否存在
            check_table_query = f"""
            SELECT COUNT(*) 
            FROM information_schema.tables 
            WHERE table_schema = DATABASE() 
            AND table_name = '{table_name}'
            """
            cursor.execute(check_table_query)
            table_exists = cursor.fetchone()[0] > 0

            if table_exists:
                # 删除表
                drop_table_query = f"DROP TABLE `{table_name}`"
                cursor.execute(drop_table_query)
                conn.commit()
                logger.info(f"成功删除历史数据表 {table_name}")
            else:
                logger.info(f"历史数据表 {table_name} 不存在，无需删除")
        except Exception as e:
            logger.error(f"删除历史数据表 {table_name} 失败: {str(e)}")
            success = False

        # 删除实时数据表
        formatted_code = format_stock_code(stock_info['code'])
        realtime_table_name = f"stock_{formatted_code}_realtime"
        try:
            # 检查表是否存在
            check_table_query = f"""
            SELECT COUNT(*) 
            FROM information_schema.tables 
            WHERE table_schema = DATABASE() 
            AND table_name = '{realtime_table_name}'
            """
            cursor.execute(check_table_query)
            table_exists = cursor.fetchone()[0] > 0

            if table_exists:
                # 删除表
                drop_table_query = f"DROP TABLE IF EXISTS `{realtime_table_name}`"
                logger.error(f"执行SQL: {drop_table_query}")
                cursor.execute(drop_table_query)
                # 立即提交，不等待后续操作
                conn.commit()
                logger.info(f"成功删除实时数据表 {realtime_table_name}")

                # 二次验证表已被删除
                cursor.execute(check_table_query)
                table_still_exists = cursor.fetchone()[0] > 0
                if table_still_exists:
                    logger.error(f"实时数据表 {realtime_table_name} 似乎没有被删除，尽管没有报错")

                    # 尝试更强力的方式删除 - 直接执行SQL，不使用参数化
                    try:
                        cursor.execute(f"SET FOREIGN_KEY_CHECKS=0;")
                        cursor.execute(f"DROP TABLE IF EXISTS `{realtime_table_name}`;")
                        cursor.execute(f"SET FOREIGN_KEY_CHECKS=1;")
                        conn.commit()
                        logger.error(f"已尝试强制删除表 {realtime_table_name}")

                        # 三次验证
                        cursor.execute(check_table_query)
                        table_still_exists_again = cursor.fetchone()[0] > 0
                        if table_still_exists_again:
                            logger.error(f"即使强制删除，表 {realtime_table_name} 仍然存在")
                        else:
                            logger.info(f"强制删除成功，表 {realtime_table_name} 已删除")
                    except Exception as e2:
                        logger.error(f"尝试强制删除表时出错: {str(e2)}")
                else:
                    logger.info(f"已确认实时数据表 {realtime_table_name} 已成功删除")
            else:
                logger.info(f"实时数据表 {realtime_table_name} 不存在，无需删除")
        except Exception as e:
            logger.error(f"删除实时数据表 {realtime_table_name} 失败: {str(e)}")
            success = False

        # 删除Redis中的实时数据键
        try:
            # 获取Redis配置
            redis_config = config.get('redis_config', {})

            # 连接到Redis
            redis_client = redis.Redis(
                host=redis_config.get('host', '127.0.0.1'),
                port=redis_config.get('port', 6379),
                db=redis_config.get('db', 0),
                password=redis_config.get('password'),
                decode_responses=True
            )

            # 尝试删除两种可能的键格式
            redis_key1 = f"stock:realtime:{formatted_code}"

            # 删除键
            if redis_client.exists(redis_key1):
                redis_client.delete(redis_key1)
                logger.info(f"成功删除Redis键 {redis_key1}")
            else:
                logger.info(f"Redis键 {redis_key1} 不存在，无需删除")

            # 关闭Redis连接
            redis_client.close()
        except Exception as e:
            logger.error(f"删除Redis实时数据键失败: {str(e)}")
            logger.error(traceback.format_exc())
            # Redis键删除失败不影响整体成功状态

        # 删除技术指标数据表
        technical_indicators_table_name = f"technical_indicators_{stock_info['name']}"
        try:
            # 检查表是否存在
            check_table_query = f"""
                SELECT COUNT(*)
                FROM information_schema.tables 
                WHERE table_schema = DATABASE() 
                AND table_name = '{technical_indicators_table_name}'
                """
            cursor.execute(check_table_query)
            table_exists = cursor.fetchone()[0] > 0

            if table_exists:
                # 删除表
                drop_table_query = f"DROP TABLE IF EXISTS `{technical_indicators_table_name}`"
                logger.error(f"执行SQL: {drop_table_query}")
                cursor.execute(drop_table_query)
                # 立即提交，不等待后续操作
                conn.commit()
                logger.info(f"成功删除技术指标数据表 {technical_indicators_table_name}")

                # 二次验证表已被删除
                cursor.execute(check_table_query)
                table_still_exists = cursor.fetchone()[0] > 0
                if table_still_exists:
                    logger.error(f"技术指标数据表 {technical_indicators_table_name} 似乎没有被删除，尽管没有报错")

                    # 尝试更强力的方式删除 - 直接执行SQL，不使用参数化
                    try:
                        cursor.execute(f"SET FOREIGN_KEY_CHECKS=0;")
                        cursor.execute(f"DROP TABLE IF EXISTS `{technical_indicators_table_name}`;")
                        cursor.execute(f"SET FOREIGN_KEY_CHECKS=1;")
                        conn.commit()
                        logger.error(f"已尝试强制删除表 {technical_indicators_table_name}")

                        # 三次验证
                        cursor.execute(check_table_query)
                        table_still_exists_again = cursor.fetchone()[0] > 0
                        if table_still_exists_again:
                            logger.error(f"即使强制删除，表 {technical_indicators_table_name} 仍然存在")
                        else:
                            logger.info(f"强制删除成功，表 {technical_indicators_table_name} 已删除")
                    except Exception as e2:
                        logger.error(f"尝试强制删除表时出错: {str(e2)}")
                else:
                    logger.info(f"已确认技术指标数据表 {technical_indicators_table_name} 已成功删除")
            else:
                logger.info(f"技术指标数据表 {technical_indicators_table_name} 不存在，无需删除")
        except Exception as e:
            logger.error(f"删除技术指标数据表 {technical_indicators_table_name} 失败: {str(e)}")
            success = False

        # 删除实时技术指标数据表
        realtime_technical_table_name = f"realtime_technical_{stock_info['name']}"
        try:
            # 检查表是否存在
            check_table_query = f"""
                    SELECT COUNT(*)
                    FROM information_schema.tables 
                    WHERE table_schema = DATABASE() 
                    AND table_name = '{realtime_technical_table_name}'
                    """
            cursor.execute(check_table_query)
            table_exists = cursor.fetchone()[0] > 0

            if table_exists:
                # 删除表
                drop_table_query = f"DROP TABLE IF EXISTS `{realtime_technical_table_name}`"
                logger.error(f"执行SQL: {drop_table_query}")
                cursor.execute(drop_table_query)
                # 立即提交，不等待后续操作
                conn.commit()
                logger.info(f"成功删除实时技术指标数据表 {realtime_technical_table_name}")

                # 二次验证表已被删除
                cursor.execute(check_table_query)
                table_still_exists = cursor.fetchone()[0] > 0
                if table_still_exists:
                    logger.error(f"实时技术指标数据表 {realtime_technical_table_name} 似乎没有被删除，尽管没有报错")

                    # 尝试更强力的方式删除 - 直接执行SQL，不使用参数化
                    try:
                        cursor.execute(f"SET FOREIGN_KEY_CHECKS=0;")
                        cursor.execute(f"DROP TABLE IF EXISTS `{realtime_technical_table_name}`;")
                        cursor.execute(f"SET FOREIGN_KEY_CHECKS=1;")
                        conn.commit()
                        logger.error(f"已尝试强制删除表 {realtime_technical_table_name}")

                        # 三次验证
                        cursor.execute(check_table_query)
                        table_still_exists_again = cursor.fetchone()[0] > 0
                        if table_still_exists_again:
                            logger.error(f"即使强制删除，表 {realtime_technical_table_name} 仍然存在")
                        else:
                            logger.info(f"强制删除成功，表 {realtime_technical_table_name} 已删除")
                    except Exception as e2:
                        logger.error(f"尝试强制删除表时出错: {str(e2)}")
                else:
                    logger.info(f"已确认实时技术指标数据表 {realtime_technical_table_name} 已成功删除")
            else:
                logger.info(f"实时技术指标数据表 {realtime_technical_table_name} 不存在，无需删除")
        except Exception as e:
            logger.error(f"删除实时技术指标数据表 {realtime_technical_table_name} 失败: {str(e)}")
            success = False

        # 关闭连接
        cursor.close()
        conn.close()

        return success
    except Exception as e:
        logger.error(f"删除股票 {stock_info['name']}({stock_info['code']}) 数据失败: {str(e)}")
        logger.error(traceback.format_exc())
        return False


def create_realtime_data_table(stock_code, stock_name):
    """创建实时数据表"""
    try:
        # 获取MySQL配置
        config = load_config()
        mysql_config = config.get('mysql_config', {})

        # 连接到MySQL数据库
        conn = mysql.connector.connect(
            host=mysql_config.get('host', '127.0.0.1'),
            port=mysql_config.get('port', 3306),
            user=mysql_config.get('user', 'root'),
            password=mysql_config.get('password', ''),
            database=mysql_config.get('database', 'stock_analysis')
        )
        cursor = conn.cursor()

        # 格式化股票代码
        formatted_code = format_stock_code(stock_code)

        # 表名
        table_name = f"stock_{formatted_code}_realtime"

        # 检查表是否存在
        check_table_query = f"""
        SELECT COUNT(*) 
        FROM information_schema.tables 
        WHERE table_schema = DATABASE() 
        AND table_name = '{table_name}'
        """
        cursor.execute(check_table_query)
        table_exists = cursor.fetchone()[0] > 0

        if table_exists:
            logger.info(f"实时数据表 {table_name} 已存在，无需创建")
            cursor.close()
            conn.close()
            return True

        # 创建表
        create_table_query = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
                `时间` VARCHAR(255) PRIMARY KEY,
                `今日开盘价` VARCHAR(255),
                `昨日收盘价` VARCHAR(255),
                `当前价格` VARCHAR(255),
                `今日最低价` VARCHAR(255),
                `竞买价` VARCHAR(255),
                `竞卖价` VARCHAR(255),
                `成交量(手)` VARCHAR(255),
                `成交额(元)` VARCHAR(255),
                `买一委托量` VARCHAR(255),
                `买一报价` VARCHAR(255),
                `买二委托量` VARCHAR(255),
                `买二报价` VARCHAR(255),
                `买三委托量` VARCHAR(255),
                `买四委托量` VARCHAR(255),
                `买四报价` VARCHAR(255),
                `买五委托量` VARCHAR(255),
                `买五报价` VARCHAR(255),
                `卖一委托量` VARCHAR(255),
                `卖一报价` VARCHAR(255),
                `卖二报价` VARCHAR(255),
                `卖三委托量` VARCHAR(255),
                `卖三报价` VARCHAR(255),
                `卖四委托量` VARCHAR(255),
                `卖五委托量` VARCHAR(255),
                `卖五报价` VARCHAR(255),
                `日期` VARCHAR(255),
                `其他保留字段` VARCHAR(255)
            )
        """

        cursor.execute(create_table_query)
        conn.commit()

        logger.info(f"成功创建实时数据表 {table_name}")

        # 关闭连接
        cursor.close()
        conn.close()

        return True
    except Exception as e:
        logger.error(f"创建实时数据表失败: {str(e)}")
        logger.error(traceback.format_exc())
        return False


def create_history_data_table(stock_code, stock_name):
    """创建股票历史数据表"""
    try:
        # 获取MySQL配置
        config = load_config()
        mysql_config = config.get('mysql_config', {})

        # 连接到MySQL数据库
        conn = mysql.connector.connect(
            host=mysql_config.get('host', '127.0.0.1'),
            port=mysql_config.get('port', 3306),
            user=mysql_config.get('user', 'root'),
            password=mysql_config.get('password', ''),
            database=mysql_config.get('database', 'stock_analysis')
        )
        cursor = conn.cursor()

        # 表名
        table_name = f"{stock_name}_history"

        # 检查表是否存在
        check_table_query = f"""
        SELECT COUNT(*) 
        FROM information_schema.tables 
        WHERE table_schema = DATABASE() 
        AND table_name = '{table_name}'
        """
        cursor.execute(check_table_query)
        table_exists = cursor.fetchone()[0] > 0

        if table_exists:
            logger.info(f"历史数据表 {table_name} 已存在，无需创建")
            cursor.close()
            conn.close()
            return True

        # 创建表
        create_table_query = f"""
        CREATE TABLE IF NOT EXISTS `{table_name}` (
            `id` INT AUTO_INCREMENT PRIMARY KEY,  -- 添加主键字段
            `日期` DATE,
            `开盘价` DECIMAL(10, 2),
            `收盘价` DECIMAL(10, 2),
            `最高价` DECIMAL(10, 2),
            `最低价` DECIMAL(10, 2),
            `成交量(手)` INT,
            `成交额(元)` DECIMAL(20, 2),
            `振幅(%)` DECIMAL(10, 2),
            `涨跌幅(%)` DECIMAL(10, 2),
            `涨跌额(元)` DECIMAL(10, 2),
            `换手率(%)` DECIMAL(10, 2),
            `市盈率` DECIMAL(10, 4),
            `市盈率TTM` DECIMAL(10, 4),
            `市净率` DECIMAL(10, 4),
            `股息率` DECIMAL(10, 4),
            `股息率TTM` DECIMAL(10, 4),
            `市销率` DECIMAL(10, 4),
            `市销率TTM` DECIMAL(10, 4),
            `总市值` DECIMAL(20, 4),
            `总市值(元)` DECIMAL(20, 2),
            `流通市值(元)` DECIMAL(20, 2),
            `总股本(股)` BIGINT,
            `流通股本` BIGINT,
            UNIQUE KEY unique_date (`日期`)  -- 添加唯一约束
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """

        cursor.execute(create_table_query)
        conn.commit()

        logger.info(f"成功创建历史数据表 {table_name}")

        # 关闭连接
        cursor.close()
        conn.close()

        return True
    except Exception as e:
        logger.error(f"创建历史数据表失败: {str(e)}")
        logger.error(traceback.format_exc())
        return False


def create_technical_indicators_table(stock_code, stock_name):
    """创建技术指标数据表"""
    try:
        # 获取MySQL配置
        config = load_config()
        mysql_config = config.get('mysql_config', {})

        # 连接到MySQL数据库
        conn = mysql.connector.connect(
            host=mysql_config.get('host', '127.0.0.1'),
            port=mysql_config.get('port', 3306),
            user=mysql_config.get('user', 'root'),
            password=mysql_config.get('password', ''),
            database=mysql_config.get('database', 'stock_analysis')
        )
        cursor = conn.cursor()

        # 表名
        table_name = f"technical_indicators_{stock_name}"

        # 检查表是否存在
        check_table_query = f"""
        SELECT COUNT(*) 
        FROM information_schema.tables 
        WHERE table_schema = DATABASE() 
        AND table_name = '{table_name}'
        """
        cursor.execute(check_table_query)
        table_exists = cursor.fetchone()[0] > 0

        if table_exists:
            logger.info(f"技术指标数据表 {table_name} 已存在，无需创建")
            cursor.close()
            conn.close()
            return True

        # 创建表
        create_table_query = f"""
        CREATE TABLE IF NOT EXISTS `{table_name}` (
            `日期` DATE,
            `MACD` DECIMAL(10, 4),
            `Signal` DECIMAL(10, 4),
            `MACD_Hist` DECIMAL(10, 4),
            `RSI` DECIMAL(10, 4),
            `Upper_Band` DECIMAL(10, 2),
            `Lower_Band` DECIMAL(10, 2),
            `MA5` DECIMAL(10, 2),
            `MA10` DECIMAL(10, 2),
            UNIQUE KEY unique_date (`日期`)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """

        cursor.execute(create_table_query)
        conn.commit()

        logger.info(f"成功创建技术指标数据表 {table_name}")

        # 关闭连接
        cursor.close()
        conn.close()

        return True
    except Exception as e:
        logger.error(f"创建技术指标数据表失败: {str(e)}")
        logger.error(traceback.format_exc())
        return False


@csrf_exempt
@require_http_methods(["POST"])
def add_stock(request):
    """添加股票到配置文件"""
    try:
        import json
        data = json.loads(request.body)
        stock_code = data.get('code')
        stock_name = data.get('name')
        industry = data.get('industry')

        if not stock_code or not stock_name:
            return JsonResponse({'status': 'error', 'message': '股票代码和名称不能为空'})

        # 如果前端没有提供行业信息，尝试获取
        if not industry:
            logger.info(f"前端未提供行业信息，尝试获取股票 {stock_code} 的行业信息")
            try:
                industry = get_stock_industry(stock_code)
            except Exception as e:
                logger.error(f"获取 {stock_code} 行业信息失败: {str(e)}")

        # 加载当前配置
        config = load_config()
        if not config:
            return JsonResponse({'status': 'error', 'message': '无法读取配置文件'})

        # 检查股票是否已存在
        stocks = config.get('stocks', [])
        for stock in stocks:
            if stock.get('code') == stock_code:
                return JsonResponse({'status': 'error', 'message': '该股票已经在列表中'})

        # 添加新股票
        new_stock = {
            'code': stock_code,
            'name': stock_name,
            'industry': industry or ''
        }

        logger.info(f"添加新股票: {new_stock}")
        stocks.append(new_stock)

        removed_stock = None
        # 如果超过10个股票，删除最早加入的
        if len(stocks) > 10:
            # 删除第一个（最早的）元素
            removed_stock = stocks.pop(0)
            message = f'股票添加成功，已自动删除最早添加的股票：{removed_stock["name"]}({removed_stock["code"]})'
            logger.info(f"删除最早的股票: {removed_stock}")

            # 删除已移除股票的历史数据和实时数据
            try:
                logger.info(f"尝试删除股票 {removed_stock['name']}({removed_stock['code']}) 的所有数据表")
                delete_result = delete_stock_data_from_database(removed_stock)
                if delete_result:
                    logger.info(f"成功删除股票 {removed_stock['name']}({removed_stock['code']}) 的所有数据表")
                else:
                    logger.warning(f"删除股票 {removed_stock['name']}({removed_stock['code']}) 的所有数据表失败")
            except Exception as e:
                logger.error(f"删除股票数据表时出错: {str(e)}")
        else:
            message = '股票添加成功'

        # 更新配置
        config['stocks'] = stocks

        # 保存配置
        if save_config(config):
            logger.info(f"配置文件保存成功，添加股票: {new_stock}")

            # 创建历史数据表
            try:
                logger.info(f"尝试创建股票 {stock_name}({stock_code}) 的历史数据表")
                history_table_result = create_history_data_table(stock_code, stock_name)
                if history_table_result:
                    logger.info(f"成功创建股票 {stock_name}({stock_code}) 的历史数据表")
                else:
                    logger.warning(f"创建股票 {stock_name}({stock_code}) 的历史数据表失败")
            except Exception as e:
                logger.error(f"创建历史数据表时出错: {str(e)}")
                logger.error(traceback.format_exc())

            # 创建实时数据表
            try:
                logger.info(f"尝试创建股票 {stock_name}({stock_code}) 的实时数据表")
                realtime_table_result = create_realtime_data_table(stock_code, stock_name)
                if realtime_table_result:
                    logger.info(f"成功创建股票 {stock_name}({stock_code}) 的实时数据表")
                else:
                    logger.warning(f"创建股票 {stock_name}({stock_code}) 的实时数据表失败")
            except Exception as e:
                logger.error(f"创建实时数据表时出错: {str(e)}")
                logger.error(traceback.format_exc())

            # 创建技术指标数据表
            try:
                logger.info(f"尝试创建股票 {stock_name}({stock_code}) 的技术指标数据表")
                technical_indicators_table_result = create_technical_indicators_table(stock_code, stock_name)
                if technical_indicators_table_result:
                    logger.info(f"成功创建股票 {stock_name}({stock_code}) 的技术指标数据表")
                else:
                    logger.warning(f"创建股票 {stock_name}({stock_code}) 的技术指标数据表失败")
            except Exception as e:
                logger.error(f"创建技术指标数据表时出错: {str(e)}")
                logger.error(traceback.format_exc())

            # 在后台执行搜狐证券爬虫获取历史数据
            try:
                # 获取日期范围（获取过去一年的数据）
                end_date = datetime.now().strftime('%Y%m%d')
                start_date = (datetime.now() - timedelta(days=365 * 10)).strftime('%Y%m%d')

                # 构建搜狐证券的股票代码
                sohu_stock_code = f"cn_{stock_code}"

                # 获取脚本路径
                script_path = os.path.join(settings.BASE_DIR, '../data/搜狐证券.py')

                # 检查文件是否存在
                if os.path.exists(script_path):
                    # 使用子进程执行搜狐证券脚本
                    logger.info(f"开始执行搜狐证券爬虫获取 {stock_name}({stock_code}) 的历史数据")

                    # 导入搜狐证券模块并直接调用函数
                    try:
                        import sys
                        sys.path.append(settings.BASE_DIR)
                        from stock_analysis.stock_project.data.搜狐证券 import process_stock_history

                        # 执行数据获取和保存
                        result = process_stock_history(sohu_stock_code, start_date, end_date, stock_name)

                        if result:
                            logger.info(f"成功获取并保存股票 {stock_name}({stock_code}) 的历史数据")
                        else:
                            logger.warning(f"获取股票 {stock_name}({stock_code}) 的历史数据失败")
                    except Exception as e:
                        logger.error(f"执行搜狐证券爬虫时出错: {str(e)}")
                        logger.error(traceback.format_exc())
                else:
                    logger.warning(f"搜狐证券脚本文件不存在: {script_path}")
            except Exception as e:
                logger.error(f"尝试获取股票历史数据时出错: {str(e)}")

            return JsonResponse({
                'status': 'success',
                'message': message,
                'data': new_stock
            })
        else:
            logger.error("保存配置文件失败")
            return JsonResponse({'status': 'error', 'message': '保存配置文件失败'})

    except Exception as e:
        logger.error(f"添加股票失败: {str(e)}")
        logger.error(traceback.format_exc())
        return JsonResponse({'status': 'error', 'message': f'添加股票失败: {str(e)}'})


@csrf_exempt
@require_http_methods(["GET", "POST"])
def settings_page(request):
    """设置页面：管理股票列表和AI配置"""
    # 加载当前配置
    config = load_config()
    message = ''
    error = ''

    if request.method == 'POST':
        action = request.POST.get('action')

        # 处理删除股票操作
        if action == 'delete_stock':
            stock_code = request.POST.get('stock_code')
            try:
                # 在删除配置前保存股票信息
                stock_info = next((s for s in config.get('stocks', []) if s['code'] == stock_code), None)

                if not stock_info:
                    error = f'找不到股票代码 {stock_code}'
                else:
                    # 从配置中删除股票
                    config['stocks'] = [s for s in config.get('stocks', []) if s['code'] != stock_code]

                    stocks_other = config.get('other_stocks', [])
                    stocks_list = stocks_other
                    stocks = {stock['code']: stock for stock in stocks_list}

                    # 保存更新后的配置
                    if save_config(config):
                        if stock_code not in stocks:
                            # 删除数据库中的相关表
                            if delete_stock_data_from_database(stock_info):
                                message = f'成功删除股票 {stock_info["name"]}({stock_code}) 及其数据表'
                            else:
                                message = f'股票 {stock_info["name"]}({stock_code}) 已从配置中删除，但删除数据表时出现问题'
                    else:
                        error = '保存配置失败'

            except Exception as e:
                error = f'删除股票时出错: {str(e)}'

        # 处理添加股票操作
        elif action == 'add_stock':
            stock_code = request.POST.get('stock_code')
            stock_name = request.POST.get('stock_name')
            stock_industry = request.POST.get('stock_industry', '')

            # 验证输入
            if not stock_code or not stock_name:
                error = '股票代码和名称不能为空'
            elif any(s['code'] == stock_code for s in config.get('stocks', [])):
                error = f'股票代码 {stock_code} 已存在'
            else:
                try:
                    # 准备股票数据
                    stock_data = {
                        'code': stock_code,
                        'name': stock_name,
                        'industry': stock_industry
                    }

                    # 模拟请求对象传递给add_stock函数
                    import json
                    from django.http import HttpRequest

                    # 创建一个模拟的请求对象
                    mock_request = HttpRequest()
                    mock_request.method = 'POST'
                    mock_request._body = json.dumps(stock_data).encode('utf-8')

                    # 调用add_stock函数添加股票
                    response = add_stock(mock_request)
                    response_data = json.loads(response.content)

                    if response_data['status'] == 'success':
                        message = response_data['message']
                    else:
                        error = response_data['message']

                except Exception as e:
                    error = f'添加股票时出错: {str(e)}'
                    logger.error(f"在设置页面添加股票时出错: {str(e)}")
                    logger.error(traceback.format_exc())

        # 处理更新AI配置操作
        elif action == 'update_ai_config':
            try:
                provider = request.POST.get('provider')
                api_key = request.POST.get('api_key')
                api_base = request.POST.get('api_base')
                model = request.POST.get('model')
                temperature = float(request.POST.get('temperature', 0.2))

                # 更新AI配置
                if 'ai_config' not in config:
                    config['ai_config'] = {}

                config['ai_config'].update({
                    'provider': provider,
                    'api_key': api_key,
                    'api_base': api_base,
                    'api_version': 'v1',
                    'model': model,
                    'temperature': temperature,
                    'max_retries': int(request.POST.get('max_retries', 3)),
                    'retry_delay': int(request.POST.get('retry_delay', 5))
                })

                if save_config(config):
                    message = 'AI配置已成功更新'
                else:
                    error = '保存AI配置失败'
            except Exception as e:
                error = f'更新AI配置时出错: {str(e)}'

        # 处理更新数据采集设置
        elif action == 'update_settings':
            try:
                realtime_interval = int(request.POST.get('realtime_interval', 1))

                # 更新设置
                if 'settings' not in config:
                    config['settings'] = {}

                config['settings'].update({
                    'realtime_interval': realtime_interval
                })

                if save_config(config):
                    message = '数据采集设置已成功更新'
                else:
                    error = '保存数据采集设置失败'
            except Exception as e:
                error = f'更新数据采集设置时出错: {str(e)}'

        # 处理更新同花顺配置
        elif action == 'update_ths_config':
            try:
                # 加载ths_config.json
                import json
                ths_config_path = os.path.join(settings.BASE_DIR, 'auto_trader', 'ths_config.json')
                with open(ths_config_path, 'r', encoding='utf-8') as f:
                    ths_config = json.load(f)

                # 更新同花顺配置
                ths_config.update({
                    'ths_path': request.POST.get('ths_path', 'E:\\同花顺\\同花顺\\xiadan.exe'),
                    'max_retry': int(request.POST.get('max_retry', 3))
                })

                # 更新交易配置
                if 'trade_config' not in ths_config:
                    ths_config['trade_config'] = {}

                ths_config['trade_config'].update({
                    'max_trades_per_day': int(request.POST.get('max_trades_per_day', 5)),
                    'max_amount_per_trade': float(request.POST.get('max_amount_per_trade', 10000)),
                    'min_interval': int(request.POST.get('min_interval', 30)),
                    'confirm_timeout': int(request.POST.get('confirm_timeout', 5)),
                    'price_adjust_pct': float(request.POST.get('price_adjust_pct', 0.002))
                })

                # 保存更新后的配置
                with open(ths_config_path, 'w', encoding='utf-8') as f:
                    json.dump(ths_config, f, ensure_ascii=False, indent=4)

                message = '同花顺配置已成功更新'
            except Exception as e:
                error = f'更新同花顺配置时出错: {str(e)}'
                logger.error(f"更新同花顺配置时出错: {str(e)}")
                logger.error(traceback.format_exc())

        # 处理更新凯利公式配置
        elif action == 'update_kelly_config':
            try:
                import json
                # 加载kelly_config.json
                kelly_config_path = os.path.join(settings.BASE_DIR, 'config', 'kelly_config.json')
                with open(kelly_config_path, 'r', encoding='utf-8') as f:
                    kelly_config = json.load(f)

                # 更新凯利公式配置
                if 'kelly_config' not in kelly_config:
                    kelly_config['kelly_config'] = {}

                kelly_config['kelly_config'].update({
                    'default_win_rate': float(request.POST.get('default_win_rate', 0.55)),
                    'max_position_ratio': float(request.POST.get('max_position_ratio', 0.3)),
                    'half_kelly': request.POST.get('half_kelly', 'true').lower() == 'true',
                    'stop_loss_ratio': float(request.POST.get('stop_loss_ratio', 0.05)),
                    'take_profit_ratio': float(request.POST.get('take_profit_ratio', 0.1)),
                    'max_kelly_score': float(request.POST.get('max_kelly_score', 0.5))
                })

                # 更新交易设置
                if 'trade_settings' not in kelly_config:
                    kelly_config['trade_settings'] = {}

                kelly_config['trade_settings'].update({
                    'total_capital': float(request.POST.get('total_capital', 100000)),
                    'available_capital': float(request.POST.get('available_capital', 80000)),
                    'max_stocks': int(request.POST.get('max_stocks', 5)),
                    'min_score_to_buy': int(request.POST.get('min_score_to_buy', 80)),
                    'trading_fee_rate': float(request.POST.get('trading_fee_rate', 0.0005))
                })

                # 保存更新后的配置
                with open(kelly_config_path, 'w', encoding='utf-8') as f:
                    json.dump(kelly_config, f, ensure_ascii=False, indent=4)

                message = '凯利公式配置已成功更新'
            except Exception as e:
                error = f'更新凯利公式配置时出错: {str(e)}'
                logger.error(f"更新凯利公式配置时出错: {str(e)}")
                logger.error(traceback.format_exc())

    # 获取最新配置
    config = load_config()

    # 加载同花顺配置
    ths_config = {}
    try:
        ths_config_path = os.path.join(settings.BASE_DIR, 'auto_trader', 'ths_config.json')
        with open(ths_config_path, 'r', encoding='utf-8') as f:
            ths_config = json.load(f)
    except Exception as e:
        logger.error(f"加载同花顺配置失败: {str(e)}")

    # 加载凯利公式配置
    kelly_config = {}
    try:
        kelly_config_path = os.path.join(settings.BASE_DIR, 'auto_trader', 'kelly_config.json')
        with open(kelly_config_path, 'r', encoding='utf-8') as f:
            kelly_config = json.load(f)
    except Exception as e:
        logger.error(f"加载凯利公式配置失败: {str(e)}")

    # 准备模板上下文
    context = {
        'stocks': config.get('stocks', []),
        'ai_config': config.get('ai_config', {}),
        'settings': config.get('settings', {}),
        'ths_config': ths_config,
        'kelly_config': kelly_config,
        'message': message,
        'error': error
    }

    return render(request, 'settings.html', context)

def trade_history_page(request):
    """交易记录页面视图"""
    trade_history = []
    try:
        # 加载配置
        with open(os.path.join(settings.BASE_DIR, 'config', 'config.json'), 'r', encoding='utf-8') as f:
            config = json.load(f)

        # 连接到MySQL
        mysql_config = config.get('mysql_config', {})
        conn = mysql.connector.connect(
            host=mysql_config.get('host', '127.0.0.1'),
            port=mysql_config.get('port', 3306),
            user=mysql_config.get('user', 'root'),
            password=mysql_config.get('password', ''),
            database=mysql_config.get('database', 'stock_analysis')
        )
        cursor = conn.cursor(dictionary=True)

        # 获取交易记录
        query = """
        SELECT * FROM trade_history 
        ORDER BY buy_time DESC 
        LIMIT 100
        """
        cursor.execute(query)
        trade_history = cursor.fetchall()

        # 处理百分比显示
        for trade in trade_history:
            if trade['profit_rate'] is not None:
                trade['profit_rate'] = round(trade['profit_rate'] * 100, 1)
            if trade['actual_position'] is not None:
                trade['actual_position'] = round(trade['actual_position'] * 100, 1)

            # 计算盈亏金额
            if trade['sell_price'] is not None:
                trade['profit_amount'] = round((trade['sell_price'] - trade['buy_price']) * trade['quantity'], 2)
            else:
                trade['profit_amount'] = None

        cursor.close()
        conn.close()
    except Exception as e:
        logger.error(f"获取交易记录失败: {str(e)}")
        logger.error(traceback.format_exc())

    return render(request, 'trade_history.html', {
        'trade_history': trade_history
    })