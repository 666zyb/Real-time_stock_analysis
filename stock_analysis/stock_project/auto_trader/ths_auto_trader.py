import os
import time
import json
import logging
import mysql.connector
import pyautogui
import pyperclip
import random
from datetime import datetime, timedelta
import traceback
from PIL import ImageGrab, Image, ImageEnhance, ImageFilter, ImageDraw, ImageFont
import pytesseract
import threading
import asyncio
import concurrent.futures

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('ths_auto_trader.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# 设置pytesseract路径 - 根据实际安装路径修改
pytesseract.pytesseract.tesseract_cmd = r'..\tesseract.exe'


class THSAutoTrader:
    """同花顺自动交易类"""

    def __init__(self, config_path=None):
        """初始化交易对象"""
        # 设置logging
        self._setup_logging()

        logger.info("初始化同花顺自动交易程序...")

        # 设置默认设置
        default_config = {
            "ths_main_title": "网上股票交易系统5.0",
            "ths_trade_title": "同花顺版",
            "verify_code_region": [710, 340, 780, 370],
            "verify_code_color": (0, 0, 0),
            "max_retry": 3,
            "simulate_human": True,
            "use_tesseract": True,
            "tesseract_cmd": r"..\tesseract.exe"
        }

        # 加载配置文件
        self.config = default_config

        # 初始化交易相关参数 - 在加载配置前就进行初始化
        self.trade_config = {
            'max_trades_per_day': 5,  # 每日最大交易次数
            'max_amount_per_trade': 10000,  # 每次交易最大金额（元）
            'min_interval': 30,  # 两次交易之间的最小间隔（秒）
            'confirm_timeout': 5,  # 确认对话框超时时间（秒）
            'price_adjust_pct': 0.002  # 价格调整比例（0.2%）
        }

        # 初始化数据库连接相关属性 - 在加载配置前就进行初始化
        self.db_config = {}
        self.db_connection = None
        self.db_cursor = None

        # 读取config.json中的股票信息 - 为实时价格监控准备
        self.stock_code_map = {}
        self.config_json_path = "../config/config.json"

        # 加载配置文件 - 现在可以安全地加载配置文件了
        if config_path:
            self._load_config(config_path)

        # 加载config.json中的股票信息
        self._load_stock_codes()

        # 设置图像识别相关参数
        self.images_dir = os.path.join(os.path.dirname(__file__), 'images')

        # 确保images目录存在
        if not os.path.exists(self.images_dir):
            os.makedirs(self.images_dir)
            logger.info(f"已创建图像目录: {self.images_dir}")
            self._create_default_template_images()

        # 设置Tesseract OCR
        if self.config.get("use_tesseract", True):
            tesseract_path = self.config.get("tesseract_cmd")
            if tesseract_path and os.path.exists(tesseract_path):
                pytesseract.pytesseract.tesseract_cmd = tesseract_path
                logger.info(f"已设置Tesseract OCR路径: {tesseract_path}")
            else:
                logger.warning("未找到Tesseract OCR路径，OCR功能可能无法正常工作")

        # 从配置中获取窗口标题
        self.ths_main_title = self.config.get("ths_main_title", "网上股票交易系统5.0")
        self.ths_trade_title = self.config.get("ths_trade_title", "同花顺版")

        # 初始化其他属性
        self.running = False
        self.stop_event = threading.Event()

        # 为实时价格监控添加控制变量
        self.realtime_monitor_running = False
        self.realtime_monitor_thread = None

        # 屏幕分辨率
        self.screen_width, self.screen_height = pyautogui.size()
        logger.info(f"屏幕分辨率: {self.screen_width}x{self.screen_height}")

        # 同花顺界面元素位置（需要根据实际分辨率调整）
        self.ui_elements = self._init_ui_elements()

        # 检查是否有pygetwindow库
        try:
            import pygetwindow
            self.has_pygetwindow = True
            logger.info("将使用pygetwindow库进行窗口操作")
        except ImportError:
            self.has_pygetwindow = False
            logger.info("未检测到pygetwindow库，将使用图像识别方法查找窗口")

        # 记录交易状态
        self.trade_records = []
        self.last_trade_time = None
        self.trading_enabled = True

        # 添加软件启动标志
        self.software_started = False

        # 添加交易锁，防止多个股票操作同时进行
        self.is_trading = False  # 当前是否有交易正在执行
        self.current_trading_stock = None  # 当前正在交易的股票

        # 在初始化时启动软件
        logger.info("程序启动时尝试启动同花顺软件...")
        self.start_trading_software()

        logger.info("同花顺自动交易程序初始化完成")

    def _load_stock_codes(self):
        """从config.json加载股票代码和名称映射"""
        try:
            if os.path.exists(self.config_json_path):
                with open(self.config_json_path, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)

                # 加载主要股票列表
                stocks = config_data.get("stocks", [])
                for stock in stocks:
                    code = stock.get("code", "")
                    name = stock.get("name", "")
                    if code and name:
                        self.stock_code_map[name] = code

                # 加载其他股票列表
                other_stocks = config_data.get("other_stocks", [])
                for stock in other_stocks:
                    code = stock.get("code", "")
                    name = stock.get("name", "")
                    if code and name:
                        self.stock_code_map[name] = code

                logger.info(f"从config.json加载了{len(self.stock_code_map)}只股票的代码信息")
            else:
                logger.warning(f"找不到配置文件: {self.config_json_path}")
        except Exception as e:
            logger.error(f"加载股票代码信息出错: {str(e)}")

    async def check_realtime_price(self, stock_name, stock_code, target_price, signal_id, stop_loss_price):
        """异步检查实时价格与目标价格"""
        try:
            # 如果监控已停止，则直接返回
            if not self.realtime_monitor_running or self.stop_event.is_set():
                return

            # 确定股票代码前缀（沪市/深市）
            prefix = "sh" if stock_code.startswith("6") else "sz"
            full_code = f"{prefix}{stock_code}"

            # 构建表名
            table_name = f"stock_{full_code}_realtime"

            # 查询实时价格 - 使用中文字段名
            query = f"""
            SELECT `当前价格` 
            FROM {table_name} 
            ORDER BY `时间` DESC 
            LIMIT 1
            """

            if not self.reconnect_db_if_needed():
                logger.error(f"数据库连接失败，无法检查股票 {stock_name}({stock_code}) 的实时价格")
                return

            # 使用新的游标执行查询
            cursor = self.db_connection.cursor(dictionary=True)
            cursor.execute(query)
            result = cursor.fetchone()
            cursor.close()

            if not result:
                logger.debug(f"未找到股票 {stock_name}({stock_code}) 的实时价格数据")
                return

            # 转换当前价格字符串为浮点数
            current_price_str = result['当前价格']
            try:
                current_price = float(current_price_str)
            except ValueError:
                logger.error(f"无法将当前价格 '{current_price_str}' 转换为数字")
                return

            # 确保target_price和stop_loss_price是float类型，处理decimal.Decimal类型
            try:
                # 将target_price和stop_loss_price转换为float
                target_price_float = float(target_price)  # 止盈价格
                stop_loss_price_float = float(stop_loss_price)  # 止损价格

                # 当前价格大于等于止盈价格（止盈卖出）或者当前价格小于等于止损价格（止损卖出）
                if current_price >= target_price_float:
                    # 止盈卖出
                    logger.info(f"止盈卖出: 股票 {stock_name}({stock_code}) 当前价格 {current_price} >= 止盈价格 {target_price_float}")
                    # 暂停监控，然后在单独线程中执行卖出
                    self.pause_monitoring_and_sell(signal_id, stock_code, current_price)
                elif current_price <= stop_loss_price_float:
                    # 止损卖出
                    logger.info(f"止损卖出: 股票 {stock_name}({stock_code}) 当前价格 {current_price} <= 止损价格 {stop_loss_price_float}")
                    # 暂停监控，然后在单独线程中执行卖出
                    self.pause_monitoring_and_sell(signal_id, stock_code, current_price)

            except (ValueError, TypeError) as e:
                logger.error(f"类型转换错误: {str(e)}，当前价格类型: {type(current_price)}，止盈价格类型: {type(target_price)}，止损价格类型: {type(stop_loss_price)}")
                logger.error(f"当前价格值: {current_price}，止盈价格值: {target_price}，止损价格值: {stop_loss_price}")

        except Exception as e:
            logger.error(f"检查股票 {stock_name}({stock_code}) 实时价格时出错: {str(e)}")
            logger.error(traceback.format_exc())

    def pause_monitoring_and_sell(self, signal_id, stock_code, current_price):
        """暂停监控并在线程中执行卖出操作"""
        # 获取股票名称以便日志记录
        stock_name = "未知股票"
        try:
            if self.reconnect_db_if_needed():
                query = """
                SELECT stock_name FROM trading_signals WHERE id = %s
                """
                self.db_cursor.execute(query, (signal_id,))
                result = self.db_cursor.fetchone()
                if result:
                    stock_name = result['stock_name']
        except Exception as e:
            logger.error(f"获取股票名称出错: {str(e)}")

        # 检查是否有交易正在进行
        if self.is_trading:
            trading_stock = self.current_trading_stock or "未知股票"
            logger.warning(f"无法卖出 {stock_name}({stock_code})，当前正在执行 {trading_stock} 的交易操作")
            # 将在可以交易时再次检查价格，无需立即重试
            return

        # 检查是否是交易时间
        if not self.is_trading_time():
            logger.warning(f"当前非交易时段，无法卖出 {stock_name}({stock_code})")
            return

        # 检查是否允许交易
        if not self.trading_enabled:
            logger.warning(f"交易功能已禁用，无法卖出 {stock_name}({stock_code})")
            return

        # 停止实时监控线程
        was_monitoring = self.realtime_monitor_running
        if was_monitoring:
            logger.info("暂停实时数据监控以执行卖出操作...")
            self.pause_realtime_monitor()  # 使用新方法暂停监控而不是停止

        # 执行卖出操作
        try:
            self.execute_sell_for_signal(signal_id, stock_code, current_price)
        finally:
            # 操作完成后，恢复监控
            if was_monitoring:
                logger.info("卖出操作完成，恢复实时数据监控...")
                self.resume_realtime_monitor()  # 使用新方法恢复监控

    def pause_realtime_monitor(self):
        """暂停实时价格监控但保持数据库连接"""
        if not self.realtime_monitor_running:
            logger.warning("实时价格监控未运行，无需暂停")
            return

        logger.info("暂停实时价格监控...")
        # 设置停止标志，但不关闭线程
        self.stop_event.set()

        # 等待监控循环退出
        if self.realtime_monitor_thread:
            # 最多等待3秒
            for _ in range(30):
                if not self.realtime_monitor_running:
                    break
                time.sleep(0.1)

        logger.info("实时价格监控已暂停")

    def resume_realtime_monitor(self):
        """恢复实时价格监控"""
        # 清除停止标志
        self.stop_event.clear()

        # 如果实时监控线程已经结束，重新启动
        if not self.realtime_monitor_thread or not self.realtime_monitor_thread.is_alive():
            logger.info("重新启动实时价格监控线程...")
            # 启动一个新的监控线程
            self.start_realtime_monitor()
        else:
            # 恢复运行标志
            self.realtime_monitor_running = True
            logger.info("实时价格监控已恢复")

    def stop_realtime_monitor(self):
        """完全停止实时价格监控（包括关闭线程）"""
        if not self.realtime_monitor_running:
            logger.warning("实时价格监控未运行")
            return

        self.realtime_monitor_running = False
        self.stop_event.set()  # 设置停止事件

        if self.realtime_monitor_thread:
            self.realtime_monitor_thread.join(timeout=5)
            if self.realtime_monitor_thread.is_alive():
                logger.warning("实时价格监控线程未能正常停止")
            else:
                logger.info("实时价格监控线程已停止")

        self.realtime_monitor_thread = None
        self.stop_event.clear()  # 重置停止事件

    def force_db_sync_after_sell(self, signal_id):
        """卖出后强制刷新数据库状态，防止重复卖出"""
        try:
            if not self.reconnect_db_if_needed():
                logger.error(f"数据库连接失败，无法更新信号 #{signal_id} 状态")
                return False

            # 确保数据库事务已提交
            self.db_connection.commit()

            # 再次执行一个更新操作，设置为已卖出状态
            query = """
            UPDATE trading_signals
            SET is_sold = 1, 
                sell_time = %s,
                trade_status = 'COMPLETED',
                last_update_time = %s
            WHERE id = %s AND is_bought = 1 AND is_sold = 0
            """

            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            self.db_cursor.execute(query, (current_time, current_time, signal_id))
            self.db_connection.commit()

            affected_rows = self.db_cursor.rowcount
            if affected_rows > 0:
                logger.info(f"已强制更新信号 #{signal_id} 为已卖出状态")
                return True
            else:
                logger.warning(f"未能更新信号 #{signal_id} 状态，可能已经是已卖出状态")
                return False

        except Exception as e:
            logger.error(f"强制更新数据库状态出错: {str(e)}")
            logger.error(traceback.format_exc())
            return False

    def execute_sell_for_signal(self, signal_id, stock_code, current_price):
        """为指定信号执行卖出操作"""
        try:
            # 查询该信号的详细信息
            if not self.reconnect_db_if_needed():
                logger.error(f"数据库连接失败，无法执行信号 #{signal_id} 的卖出操作")
                return

            query = """
            SELECT stock_name, buy_quantity 
            FROM trading_signals 
            WHERE id = %s
            """
            self.db_cursor.execute(query, (signal_id,))
            signal_info = self.db_cursor.fetchone()

            if not signal_info:
                logger.error(f"找不到信号 #{signal_id} 的信息")
                return

            stock_name = signal_info['stock_name']
            amount = signal_info['buy_quantity']

            # 执行卖出操作
            success, message = self.execute_sell_order(stock_code, current_price, amount)

            if success:
                logger.info(f"成功执行信号 #{signal_id} 的卖出操作: {stock_name}({stock_code})")

                # 记录交易结果
                trade_info = {
                    'stock_code': stock_code,
                    'stock_name': stock_name,
                    'signal_type': 'sell',
                    'price': current_price,
                    'amount': amount
                }
                self.log_trade_result(signal_id, True, trade_info, None)

                # 强制更新数据库状态为已卖出
                self.force_db_sync_after_sell(signal_id)

                # 保存交易记录到trade_history表
                trade_history_saved = self.save_trade_history(signal_id, is_buy=False, override_check=True, price=current_price)

                # 添加小延时，确保数据库事务完成
                time.sleep(0.5)

                # 只有在成功保存交易历史后才删除交易记录
                if trade_history_saved:
                    # 确认交易历史已保存，再删除交易记录
                    if self.delete_trade_record(signal_id):
                        logger.info(f"卖出成功，交易历史已保存，已删除交易记录: {stock_name}({stock_code})")
                    else:
                        logger.warning(f"卖出成功，交易历史已保存，但删除交易记录失败: {stock_name}({stock_code})")
                else:
                    logger.warning(f"卖出成功，但保存交易历史失败，不删除交易记录: {stock_name}({stock_code})")

                # 添加延迟，等待数据库更新
                logger.info("等待2秒，确保数据库有足够时间更新...")
                time.sleep(2)
            else:
                logger.error(f"执行信号 #{signal_id} 的卖出操作失败: {message}")

        except Exception as e:
            logger.error(f"为信号 #{signal_id} 执行卖出操作时出错: {str(e)}")
            logger.error(traceback.format_exc())

    async def monitor_realtime_prices(self):
        """监控所有待卖出股票的实时价格"""
        logger.info("开始监控实时股票价格")

        while self.realtime_monitor_running and not self.stop_event.is_set():
            try:
                # 获取所有待卖出的股票信号
                if not self.reconnect_db_if_needed():
                    logger.error("数据库连接失败，暂停实时价格监控")
                    await asyncio.sleep(1)
                    continue

                query = """
                SELECT id, stock_code, stock_name, take_profit_price, stop_loss_price
                FROM trading_signals 
                WHERE is_bought = 1 AND is_sold = 0
                """
                self.db_cursor.execute(query)
                sell_signals = self.db_cursor.fetchall()

                if sell_signals:
                    logger.debug(f"找到 {len(sell_signals)} 条待卖出信号")

                    # 并发检查每支股票的实时价格
                    tasks = []
                    for signal in sell_signals:
                        stock_name = signal['stock_name']
                        stock_code = None

                        # 从stock_code_map中查找股票代码
                        if stock_name in self.stock_code_map:
                            stock_code = self.stock_code_map[stock_name]
                        else:
                            # 如果没找到，可能已经在信号中有股票代码
                            stock_code = signal['stock_code']

                        if not stock_code:
                            logger.warning(f"无法确定股票 {stock_name} 的代码，跳过该股票")
                            continue

                        signal_id = signal['id']
                        target_price = signal['take_profit_price']  # 止盈价格
                        stop_loss_price = signal['stop_loss_price']  # 止损价格

                        # 创建检查任务
                        task = self.check_realtime_price(stock_name, stock_code, target_price, signal_id, stop_loss_price)
                        tasks.append(task)

                    # 并发执行所有检查任务
                    if tasks:
                        await asyncio.gather(*tasks)

                # 每秒检查一次
                await asyncio.sleep(1)

            except Exception as e:
                logger.error(f"监控实时价格时出错: {str(e)}")
                logger.error(traceback.format_exc())
                await asyncio.sleep(1)

    def start_realtime_monitor(self):
        """启动实时价格监控线程"""
        if self.realtime_monitor_running:
            logger.warning("实时价格监控已经在运行中")
            return

        self.realtime_monitor_running = True

        # 创建异步任务运行器
        async def run_monitor():
            await self.monitor_realtime_prices()

        # 在新线程中运行异步循环
        def run_async_loop():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(run_monitor())
            loop.close()

        self.realtime_monitor_thread = threading.Thread(target=run_async_loop)
        self.realtime_monitor_thread.daemon = True
        self.realtime_monitor_thread.start()

        logger.info("实时价格监控线程已启动")

    def run(self, interval=30):
        """启动自动交易监控"""
        logger.info("启动同花顺自动交易监控")
        logger.info(f"检查间隔: {interval} 秒")

        # 连接数据库
        try:
            self.connect_to_db()
        except Exception as e:
            logger.error(f"数据库连接失败: {str(e)}")
            logger.warning("将使用无数据库模式运行，但一些功能可能受限")

        # 创建日志表
        self.create_trading_logs_table_if_not_exists()

        # 在程序启动时激活一次窗口
        logger.info("程序启动时激活同花顺交易窗口...")
        if not self.find_and_activate_window(no_start=False):
            logger.error("无法激活同花顺交易窗口，请手动启动同花顺并重试")
            return
        logger.info("同花顺交易窗口已激活，开始监控")

        # 启动实时价格监控
        self.start_realtime_monitor()

        try:
            while True:
                start_time = time.time()
                current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                logger.info(f"===== {current_time} - 开始检查待执行交易 =====")

                # 处理待执行的交易
                processed_count = self.process_pending_trades()

                if processed_count > 0:
                    logger.info(f"本次共处理 {processed_count} 条交易信号")
                else:
                    logger.info("当前没有待执行的交易信号")

                # 计算本次执行耗时
                elapsed = time.time() - start_time
                logger.info(f"本次检查耗时: {elapsed:.2f} 秒")

                # 计算等待时间
                wait_time = max(0, interval - elapsed)
                if wait_time > 0:
                    logger.info(f"等待 {wait_time:.2f} 秒后进行下一次检查...")
                    time.sleep(wait_time)

        except KeyboardInterrupt:
            logger.info("\n检测到 Ctrl+C，程序正在退出...")
        except Exception as e:
            logger.error(f"程序运行出错: {str(e)}")
            logger.error(traceback.format_exc())
        finally:
            # 停止实时价格监控
            self.stop_realtime_monitor()

            logger.info("关闭资源...")
            self.close_connections()
            logger.info("程序已退出")

    def _setup_logging(self):
        """设置日志配置"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('ths_auto_trader.log', encoding='utf-8')
            ]
        )
        logger.info("日志配置完成")

    def _load_config(self, config_path):
        """加载配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)

            # 获取数据库配置
            if 'mysql_config' in self.config:
                self.db_config = self.config.get('mysql_config', {})
                logger.info("数据库配置加载成功")

            # 获取交易参数配置（如果有）
            if 'trade_config' in self.config:
                for key, value in self.config['trade_config'].items():
                    if key in self.trade_config:
                        self.trade_config[key] = value

            logger.info(f"交易参数: {self.trade_config}")
            logger.info("配置文件加载成功")

        except Exception as e:
            logger.error(f"加载配置文件出错: {str(e)}")
            raise

    def check_database_settings(self):
        """检查数据库配置是否完整"""
        required_fields = ['host', 'user', 'password', 'database']

        if not self.db_config:
            logger.warning("数据库配置为空")
            return False

        for field in required_fields:
            if field not in self.db_config:
                logger.warning(f"数据库配置缺少必要字段: {field}")
                return False

        return True

    def connect_to_db(self):
        """连接数据库"""
        # 检查数据库配置是否正确
        if not self.check_database_settings():
            logger.warning("数据库配置不完整，将无法连接到数据库")
            self.db_connection = None
            self.db_cursor = None
            return False

        try:
            logger.info("正在连接数据库...")
            self.db_connection = mysql.connector.connect(
                host=self.db_config.get('host', 'localhost'),
                port=self.db_config.get('port', 3306),
                user=self.db_config.get('user', 'root'),
                password=self.db_config.get('password', ''),
                database=self.db_config.get('database', 'stock_analysis')
            )
            self.db_cursor = self.db_connection.cursor(dictionary=True)
            logger.info("数据库连接成功")
            return True
        except Exception as e:
            logger.error(f"数据库连接失败: {str(e)}")
            self.db_connection = None
            self.db_cursor = None
            return False

    def reconnect_db_if_needed(self):
        """如果数据库连接断开，则重新连接"""
        if not hasattr(self, 'db_connection') or self.db_connection is None:
            logger.warning("数据库连接未初始化")
            return False

        try:
            if not self.db_connection.is_connected():
                logger.info("数据库连接已断开，尝试重新连接...")
                self.connect_to_db()
            return True
        except Exception as e:
            logger.error(f"重新连接数据库失败: {str(e)}")
            return False

    def close_connections(self):
        """关闭数据库连接"""
        try:
            if self.db_cursor:
                self.db_cursor.close()
            if self.db_connection:
                self.db_connection.close()
            logger.info("数据库连接已关闭")
        except Exception as e:
            logger.error(f"关闭数据库连接出错: {str(e)}")

    def get_pending_trades(self):
        """从数据库获取待执行的交易信号"""
        if not hasattr(self, 'db_cursor') or self.db_cursor is None:
            logger.warning("数据库游标未初始化，无法获取交易信号")
            return []

        try:
            if not self.reconnect_db_if_needed():
                return []

            # 查询交易信号表中状态为'PENDING'的信号
            query = """
            SELECT id, stock_code, stock_name, analysis_time, 
                   current_price as price, buy_quantity as amount, 
                   CASE 
                       WHEN is_bought = 0 AND is_sold = 0 THEN 'buy'
                       WHEN is_bought = 1 AND is_sold = 0 THEN 'sell'
                       ELSE 'unknown'
                   END as signal_type,
                   trade_status, notes as extra_info
            FROM trading_signals
            WHERE trade_status = 'PENDING'
            ORDER BY analysis_time ASC
            """
            self.db_cursor.execute(query)
            signals = self.db_cursor.fetchall()

            if signals:
                logger.info(f"获取到 {len(signals)} 条待执行交易信号")

            return signals
        except Exception as e:
            logger.error(f"获取待执行交易信号出错: {str(e)}")
            return []

    def update_signal_status(self, signal_id, status, remarks=None):
        """更新交易信号状态"""
        if not hasattr(self, 'db_cursor') or self.db_cursor is None:
            logger.warning(f"数据库游标未初始化，无法更新信号 #{signal_id} 状态")
            return False

        try:
            if not self.reconnect_db_if_needed():
                return False

            # 将状态映射到表中的状态字段
            status_map = {
                'completed': 'COMPLETED',
                'failed': 'FAILED',
                'postponed': 'PENDING',
                'pending': 'PENDING'  # 确保pending也映射到PENDING状态
            }

            trade_status = status_map.get(status, status.upper())

            # 构建更新SQL
            update_values = {
                'trade_status': trade_status,
                'last_update_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

            # 查询当前记录以确定信号类型
            self.db_cursor.execute(
                "SELECT CASE WHEN is_bought = 0 AND is_sold = 0 THEN 'buy' WHEN is_bought = 1 AND is_sold = 0 THEN 'sell' ELSE 'unknown' END as signal_type FROM trading_signals WHERE id = %s",
                (signal_id,)
            )
            result = self.db_cursor.fetchone()

            if result:
                signal_type = result['signal_type']

                # 如果是买入信号且成功完成或保持pending状态用于后续卖出
                if signal_type == 'buy' and (status == 'completed' or status == 'pending'):
                    # 买入成功，更新is_bought为1
                    update_values['is_bought'] = True
                    update_values['buy_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    logger.info(f"买入成功，更新is_bought=1，信号ID: {signal_id}")

                # 如果是卖出信号且成功完成
                elif signal_type == 'sell' and status == 'completed':
                    update_values['is_sold'] = True
                    update_values['sell_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            # 添加备注信息
            if remarks:
                update_values['notes'] = remarks
                update_values['error_message'] = remarks if status == 'failed' else None

            # 构建更新SQL
            set_clause = ", ".join([f"{k} = %s" for k in update_values.keys()])
            values = list(update_values.values())

            query = f"""
            UPDATE trading_signals
            SET {set_clause}
            WHERE id = %s
            """
            values.append(signal_id)

            self.db_cursor.execute(query, values)
            self.db_connection.commit()

            logger.info(f"交易信号 #{signal_id} 状态已更新为: {trade_status}")
            return True
        except Exception as e:
            logger.error(f"更新交易信号状态出错: {str(e)}")
            return False

    def log_trade_result(self, signal_id, success, trade_info, error_msg=None):
        """记录交易执行结果"""
        if not hasattr(self, 'db_cursor') or self.db_cursor is None:
            logger.warning(f"数据库游标未初始化，无法记录信号 #{signal_id} 的交易结果")
            return False

        try:
            if not self.reconnect_db_if_needed():
                return False

            # 获取信号类型
            signal_type = trade_info.get('signal_type', '')

            # 记录到交易日志表
            try:
                # 确保使用当前正确的价格
                # 对于卖出操作，确保使用当前的实际卖出价格，而不是买入价格
                if signal_type == 'sell':
                    logger.info(f"记录卖出交易结果 - 信号ID: {signal_id}, 价格: {trade_info.get('price')}")

                insert_query = """
                INSERT INTO trade_execution_logs (
                    signal_id, trade_time, success, trade_info, error_message
                ) VALUES (%s, %s, %s, %s, %s)
                """

                # 将trade_info转换为JSON字符串
                trade_info_json = json.dumps(trade_info)

                self.db_cursor.execute(insert_query, (
                    signal_id,
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    1 if success else 0,
                    trade_info_json,
                    error_msg or ''
                ))
                self.db_connection.commit()
                logger.info(f"已记录交易日志，信号ID: {signal_id}, 类型: {signal_type}, 成功: {success}")
            except Exception as e:
                logger.error(f"记录交易日志失败: {e}")
                logger.error(traceback.format_exc())

            # 构建更新数据
            update_data = {}

            # 根据交易类型和结果更新相应字段
            if success:
                if signal_type == 'buy':
                    # 买入成功，更新买入价格
                    if 'price' in trade_info:
                        update_data['buy_price'] = trade_info['price']
                    if 'amount' in trade_info:
                        update_data['buy_quantity'] = trade_info['amount']

                elif signal_type == 'sell':
                    # 卖出成功，更新卖出价格
                    if 'price' in trade_info:
                        update_data['sell_price'] = trade_info['price']
                        logger.info(f"更新卖出价格: {trade_info['price']}")

            # 如果有错误信息，更新错误信息字段
            if error_msg:
                update_data['error_message'] = error_msg

            # 只有在有更新数据时才执行更新
            if update_data:
                set_clause = ", ".join([f"{k} = %s" for k in update_data.keys()])
                values = list(update_data.values())
                values.append(signal_id)  # 添加WHERE条件的参数

                query = f"""
                UPDATE trading_signals
                SET {set_clause}
                WHERE id = %s
                """

                self.db_cursor.execute(query, values)
                self.db_connection.commit()

                logger.info(f"更新交易信号成功: #{signal_id}")
                return True
            else:
                logger.info(f"没有需要更新的数据: #{signal_id}")
                return True

        except Exception as e:
            logger.error(f"记录交易结果失败: {e}")
            logger.error(traceback.format_exc())
            return False

    def find_and_activate_window(self, no_start=False):
        """
        自动查找并激活同花顺交易窗口，无需用户干预

        参数:
            no_start (bool): 如果为True，即使找不到窗口也不会尝试启动软件

        Returns:
            bool: 窗口是否成功激活
        """
        logger.info("开始查找并激活同花顺交易窗口...")
        window_found = False

        # 1. 首先尝试使用pygetwindow查找窗口
        try:
            import pygetwindow as gw
            logger.info("使用pygetwindow查找窗口...")

            # 查找主交易窗口
            windows = gw.getAllTitles()
            logger.info(f"当前所有窗口: {windows}")

            ths_windows = []
            for title in windows:
                if self.ths_main_title in title or self.ths_trade_title in title or "同花顺" in title:
                    ths_windows.append(title)

            if ths_windows:
                logger.info(f"找到同花顺相关窗口: {ths_windows}")
                # 尝试激活第一个找到的窗口
                window = gw.getWindowsWithTitle(ths_windows[0])[0]
                window.activate()

                # 等待窗口激活
                time.sleep(1)

                # 检查窗口是否激活
                active_window = gw.getActiveWindow()
                if active_window and active_window.title in ths_windows:
                    logger.info(f"成功激活窗口: {active_window.title}")
                    window_found = True
                else:
                    logger.warning(f"窗口未成功激活，尝试其他方法")
            else:
                logger.warning("未找到同花顺交易窗口，尝试使用图像识别方法...")
        except Exception as e:
            logger.warning(f"使用pygetwindow查找窗口失败: {str(e)}")

        # 2. 如果pygetwindow失败，使用图像识别方法查找窗口
        if not window_found:
            logger.info("使用图像识别方法查找窗口...")

            # 检查是否安装了OpenCV (使用try except检测是否支持confidence参数)
            has_opencv = True
            try:
                import cv2
                logger.info("检测到OpenCV库，可以使用高级图像识别功能")
            except ImportError:
                has_opencv = False
                logger.warning("未检测到OpenCV库，将使用基本图像匹配功能，准确度可能降低")
                logger.warning("建议安装OpenCV以提高图像识别准确度: pip install opencv-python")

            # 使用PyAutoGUI查找窗口或图标
            # 尝试查找图标
            icon_path = os.path.join(self.images_dir, 'ths_icon.png')
            header_path = os.path.join(self.images_dir, 'ths_header.png')

            if os.path.exists(icon_path):
                logger.info(f"尝试查找同花顺图标: {icon_path}")
                try:
                    # 根据是否有OpenCV使用不同的参数
                    if has_opencv:
                        icon_pos = pyautogui.locateOnScreen(icon_path, confidence=0.7)
                    else:
                        icon_pos = pyautogui.locateOnScreen(icon_path)

                    if icon_pos:
                        logger.info(f"找到同花顺图标位置: {icon_pos}")
                        # 点击图标位置，激活或打开窗口
                        pyautogui.click(icon_pos.left + icon_pos.width / 2,
                                        icon_pos.top + icon_pos.height / 2)
                        time.sleep(2)  # 等待窗口打开
                        window_found = True
                except Exception as e:
                    logger.warning(f"查找图标失败: {str(e)}")

            # 尝试查找标题栏
            if not window_found and os.path.exists(header_path):
                logger.info(f"尝试查找同花顺标题栏: {header_path}")
                try:
                    # 根据是否有OpenCV使用不同的参数
                    if has_opencv:
                        header_pos = pyautogui.locateOnScreen(header_path, confidence=0.7)
                    else:
                        header_pos = pyautogui.locateOnScreen(header_path)

                    if header_pos:
                        logger.info(f"找到同花顺标题栏位置: {header_pos}")
                        # 点击标题栏位置，激活窗口
                        pyautogui.click(header_pos.left + header_pos.width / 2,
                                        header_pos.top + header_pos.height / 2)
                        time.sleep(1)
                        window_found = True
                except Exception as e:
                    logger.warning(f"查找标题栏失败: {str(e)}")

        # 3. 如果窗口仍未找到且不禁止启动，尝试启动程序
        if not window_found and not no_start and not self.software_started:
            logger.info("未找到同花顺窗口，尝试启动程序...")
            self.start_trading_software()
            window_found = self.find_and_activate_window(no_start=True)

        if window_found:
            logger.info("同花顺交易窗口已成功激活")
        else:
            logger.error("无法找到或激活同花顺交易窗口")

        return window_found

    def simulate_mouse_click(self, x, y, click_type='left', add_offset=True):
        """模拟鼠标点击"""
        # 添加随机偏移，使操作更像人类
        if add_offset:
            offset_x = random.randint(-3, 3)
            offset_y = random.randint(-3, 3)
            x += offset_x
            y += offset_y

        # 移动鼠标到目标位置
        pyautogui.moveTo(x, y, duration=random.uniform(0.1, 0.3))

        # 等待一小段随机时间
        time.sleep(random.uniform(0.05, 0.15))

        # 执行点击
        if click_type == 'left':
            pyautogui.click()
        elif click_type == 'right':
            pyautogui.rightClick()
        elif click_type == 'double':
            pyautogui.doubleClick()

        # 等待随机时间
        time.sleep(random.uniform(0.1, 0.3))

    def input_text(self, text):
        """输入文本（不使用Ctrl+V，每次自动先清空再输入，极速模式）"""
        logger.info(f"准备输入文本: {text}")

        # 先清空输入框
        logger.info("清空当前输入框")
        pyautogui.hotkey('ctrl', 'a')
        pyautogui.press('delete')

        # 方法1: 使用数字小键盘（对数字最有效）
        if text.isdigit():
            try:
                # 使用小键盘输入数字（确保NumLock已打开）
                pyautogui.press('numlock')

                # 逐个输入数字，极速模式
                for digit in text:
                    pyautogui.press(f'num{digit}')

                logger.info("使用数字小键盘输入成功")
                return
            except Exception as e:
                logger.warning(f"使用数字小键盘输入失败: {str(e)}")

        # 方法2: 直接模拟按键（所有字符均适用）
        try:
            # 直接输入文本，使用极速模式（无间隔）
            pyautogui.write(text, interval=0)

            logger.info("使用直接键盘输入成功")
            return
        except Exception as e:
            logger.warning(f"使用直接键盘输入失败: {str(e)}")

        # 方法3: 用鼠标右键菜单粘贴
        try:
            # 复制到剪贴板
            pyperclip.copy(text)

            # 右键点击输入框
            pyautogui.rightClick()
            time.sleep(0.01)  # 保留极小延迟以确保菜单显示

            # 通常粘贴选项在右键菜单的第一项或第二项
            # 尝试按下箭头和回车键选择粘贴
            pyautogui.press('down')  # 移动到可能的"粘贴"选项
            pyautogui.press('enter')

            logger.info("使用右键菜单粘贴成功")
            return
        except Exception as e:
            logger.warning(f"使用右键菜单粘贴失败: {str(e)}")

        # 记录错误
        logger.error(f"所有输入方法都失败，无法输入文本: {text}")

    def clear_input(self):
        """清除输入框内容"""
        pyautogui.hotkey('ctrl', 'a')
        time.sleep(0.5)
        pyautogui.press('delete')
        time.sleep(0.5)

    def capture_screen_region(self, region):
        """截取屏幕区域"""
        screenshot = ImageGrab.grab(bbox=region)
        return screenshot

    def process_image_for_ocr(self, image):
        """使用PIL处理图像以准备OCR

        替代原来使用OpenCV的图像处理
        """
        try:
            # 转换为灰度图像
            gray_image = image.convert('L')

            # 增强对比度
            enhancer = ImageEnhance.Contrast(gray_image)
            enhanced_image = enhancer.enhance(2.0)

            # 应用一些滤镜来帮助文本识别
            sharpened = enhanced_image.filter(ImageFilter.SHARPEN)

            # 阈值处理（二值化）
            threshold = 150
            binary_image = sharpened.point(lambda p: p > threshold and 255)

            return binary_image
        except Exception as e:
            logger.error(f"图像处理出错: {str(e)}")
            return image

    def recognize_text(self, image):
        """识别图像中的文本"""
        try:
            # 使用PIL处理图像
            processed_image = self.process_image_for_ocr(image)

            # 使用Tesseract识别文本
            text = pytesseract.image_to_string(processed_image, lang='chi_sim+eng')
            return text.strip()
        except Exception as e:
            logger.error(f"文本识别出错: {str(e)}")
            return ""

    def verify_dialog_content(self, expected_stock_code=None):
        """验证确认对话框内容"""
        try:
            # 等待确认对话框出现
            time.sleep(1)

            # 截取确认对话框区域
            dialog_region = (self.screen_width // 2 - 250, self.screen_height // 2 - 150,
                             self.screen_width // 2 + 250, self.screen_height // 2 + 150)
            dialog_img = self.capture_screen_region(dialog_region)

            # 识别对话框文本
            dialog_text = self.recognize_text(dialog_img)
            logger.info(f"确认对话框文本: {dialog_text}")

            # 验证对话框内容
            if expected_stock_code and expected_stock_code not in dialog_text:
                logger.warning(f"确认对话框中未找到预期的股票代码: {expected_stock_code}")
                return False

            # 检查是否存在错误提示（如资金不足、交易时段限制等）
            error_keywords = ['资金不足', '可用资金不足', '非交易时段', '禁止交易', '错误']
            for keyword in error_keywords:
                if keyword in dialog_text:
                    logger.error(f"确认对话框中存在错误信息: {keyword}")
                    return False

            return True
        except Exception as e:
            logger.error(f"验证对话框内容出错: {str(e)}")
            return False

    def clear_price_input(self, price_x, price_y):
        """使用多种方法尝试清空价格输入框"""
        logger.info("尝试清空价格输入框...")

        # 方法1: 双击选中
        try:
            # 移动到价格输入框
            pyautogui.moveTo(price_x, price_y, duration=0.2)
            # 使用双击（一次doubleClick而不是两次click）
            pyautogui.doubleClick()
            time.sleep(0.1)
            # 尝试输入空字符
            pyautogui.write("", interval=0)
            # 按删除键
            pyautogui.press('delete')
            logger.info("使用双击方式尝试清空")
            return
        except Exception as e:
            logger.warning(f"双击清空失败: {str(e)}")

        # 方法2: 三连击选中全部文本
        try:
            # 移动到价格输入框
            pyautogui.moveTo(price_x, price_y, duration=0.2)
            # 连续点击三次（通常可以选中整行）
            pyautogui.click()
            time.sleep(0.1)
            pyautogui.click()
            time.sleep(0.1)
            pyautogui.click()
            # 尝试输入空字符
            pyautogui.write("", interval=0)
            # 按删除键
            pyautogui.press('delete')
            logger.info("使用三连击方式尝试清空")
            return
        except Exception as e:
            logger.warning(f"三连击清空失败: {str(e)}")

        # 方法3: 拖动选择
        try:
            # 移动到价格输入框
            pyautogui.moveTo(price_x, price_y, duration=0.2)
            # 点击并按住
            pyautogui.mouseDown()
            # 拖动30像素以选择文本
            pyautogui.moveTo(price_x + 50, price_y, duration=0.3)
            # 释放鼠标
            pyautogui.mouseUp()
            # 尝试输入空字符
            pyautogui.write("", interval=0)
            # 按删除键
            pyautogui.press('delete')
            logger.info("使用拖动选择方式尝试清空")
            return
        except Exception as e:
            logger.warning(f"拖动选择清空失败: {str(e)}")

        # 方法4: 使用组合键(END+SHIFT+HOME)选择全部文本
        try:
            # 移动到价格输入框并点击
            pyautogui.moveTo(price_x, price_y, duration=0.2)
            pyautogui.click()
            time.sleep(0.1)
            # 使用END键移到末尾
            pyautogui.press('end')
            time.sleep(0.1)
            # 按住SHIFT+HOME选择到行首
            pyautogui.keyDown('shift')
            pyautogui.press('home')
            pyautogui.keyUp('shift')
            # 按删除键
            pyautogui.press('delete')
            logger.info("使用组合键选择方式尝试清空")
            return
        except Exception as e:
            logger.warning(f"组合键选择清空失败: {str(e)}")

        # 最后方法: 直接多按几次删除键和退格键
        try:
            # 移动到价格输入框并点击
            pyautogui.moveTo(price_x, price_y, duration=0.2)
            pyautogui.click()
            # 多按几次删除键和退格键
            for _ in range(10):
                pyautogui.press('delete')
                pyautogui.press('backspace')
            logger.info("使用多次删除键方式尝试清空")
        except Exception as e:
            logger.warning(f"多次删除键清空失败: {str(e)}")

        logger.warning("尝试了多种方法，可能都无法清空价格输入框")

    def execute_buy_order(self, stock_code, price, amount=None, max_amount=None):
        """执行买入订单"""
        # 检查是否有其他交易正在进行
        if self.is_trading:
            trading_stock = self.current_trading_stock or "未知股票"
            logger.warning(f"当前正在进行 {trading_stock} 的交易操作，无法买入 {stock_code}")
            return False, f"交易系统忙，正在操作 {trading_stock}"

        # 设置交易锁
        self.is_trading = True
        self.current_trading_stock = f"买入 {stock_code}"
        logger.info(f"开始执行交易: {self.current_trading_stock}")

        try:
            # 不再在每次交易时激活窗口，假设窗口已在程序启动时激活
            # 点击买入标签页
            self.simulate_mouse_click(*self.ui_elements['buy_tab'])
            logger.info("已点击买入标签页")

            # 输入股票代码
            self.simulate_mouse_click(*self.ui_elements['stock_code_input'])
            self.input_text(stock_code)  # input_text已包含清空操作
            logger.info(f"已输入股票代码: {stock_code}")

            # 清空价格输入框（因为输入股票代码后可能会自动填入当前价格）
            price_x, price_y = self.ui_elements['stock_price_input']
            self.clear_price_input(price_x, price_y)

            # 输入价格（略高于市价0.2%，提高成交概率）
            adjusted_price = price * (1 + self.trade_config['price_adjust_pct'])
            adjusted_price = round(adjusted_price, 2)  # 保留两位小数
            adjusted_price_str = str(adjusted_price)

            # 点击后直接输入价格，不再调用input_text
            self.simulate_mouse_click(price_x, price_y)
            pyautogui.write(adjusted_price_str, interval=0)
            logger.info(f"已输入买入价格: {adjusted_price} (原价: {price})")

            # 输入数量
            final_amount = amount
            if not final_amount and max_amount:
                # 计算可买入的股数（必须是100的整数倍）
                shares = int(max_amount / adjusted_price // 100 * 100)
                if shares < 100:
                    self.is_trading = False
                    self.current_trading_stock = None
                    return False, f"买入金额 {max_amount} 元不足以购买最小100股（需要 {adjusted_price * 100} 元）"
                final_amount = shares

            if not final_amount:
                self.is_trading = False
                self.current_trading_stock = None
                return False, "未指定买入数量或金额"

            final_amount_str = str(final_amount)
            self.simulate_mouse_click(*self.ui_elements['stock_amount_input'])
            self.input_text(final_amount_str)  # input_text已包含清空操作
            logger.info(f"已输入买入数量: {final_amount} 股")

            # 点击买入按钮
            self.simulate_mouse_click(*self.ui_elements['buy_button'])
            logger.info("已点击买入按钮")

            # # 验证确认对话框
            # if not self.verify_dialog_content(stock_code):
            #     # 点击取消按钮
            #     self.simulate_mouse_click(*self.ui_elements['cancel_button'])
            #     return False, "确认对话框内容验证失败"

            # 点击第一个确认按钮
            self.simulate_mouse_click(*self.ui_elements['confirm_button'])
            logger.info("已点击第一个确认按钮")

            # 点击第二个确认按钮（如果有）
            if 'second_confirm' in self.ui_elements:
                self.simulate_mouse_click(*self.ui_elements['second_confirm'])
                logger.info("已点击第二个确认按钮")

            # 等待操作完成 - 保留极短等待
            time.sleep(0.5)

            # 记录本次交易
            self.last_trade_time = datetime.now()
            self.trade_records.append({
                'type': 'buy',
                'stock_code': stock_code,
                'price': adjusted_price,
                'amount': final_amount,
                'time': self.last_trade_time.strftime('%Y-%m-%d %H:%M:%S')
            })

            return True, f"买入 {stock_code} 成功，数量: {final_amount} 股，价格: {adjusted_price} 元"

        except Exception as e:
            logger.error(f"执行买入订单出错: {str(e)}")
            logger.error(traceback.format_exc())
            return False, f"执行买入订单出错: {str(e)}"
        finally:
            # 解除交易锁
            self.is_trading = False
            self.current_trading_stock = None
            logger.info("交易完成，解除交易锁")

    def execute_sell_order(self, stock_code, price=None, amount=None):
        """执行卖出订单"""
        # 检查是否有其他交易正在进行
        if self.is_trading:
            trading_stock = self.current_trading_stock or "未知股票"
            logger.warning(f"当前正在进行 {trading_stock} 的交易操作，无法卖出 {stock_code}")
            return False, f"交易系统忙，正在操作 {trading_stock}"

        # 设置交易锁
        self.is_trading = True
        self.current_trading_stock = f"卖出 {stock_code}"
        logger.info(f"开始执行交易: {self.current_trading_stock}")

        try:
            # 不再在每次交易时激活窗口，假设窗口已在程序启动时激活
            # 点击卖出标签页
            self.simulate_mouse_click(*self.ui_elements['sell_tab'])
            logger.info("已点击卖出标签页")

            # 输入股票代码
            self.simulate_mouse_click(*self.ui_elements['stock_code_input'])
            self.input_text(stock_code)  # input_text已包含清空操作
            logger.info(f"已输入股票代码: {stock_code}")

            # 清空价格输入框（因为输入股票代码后可能会自动填入当前价格）
            price_x, price_y = self.ui_elements['stock_price_input']
            self.clear_price_input(price_x, price_y)

            # 输入价格（如果提供）
            if price:
                # 略低于市价0.2%，提高成交概率
                adjusted_price = price * (1 - self.trade_config['price_adjust_pct'])
                adjusted_price = round(adjusted_price, 2)  # 保留两位小数
                adjusted_price_str = str(adjusted_price)

                # 点击后直接输入价格，不再调用input_text
                self.simulate_mouse_click(price_x, price_y)
                pyautogui.write(adjusted_price_str, interval=0)
                logger.info(f"已输入卖出价格: {adjusted_price} (原价: {price})")

            # 输入数量（如果提供）
            if amount:
                self.simulate_mouse_click(*self.ui_elements['stock_amount_input'])
                self.input_text(str(amount))  # input_text已包含清空操作
                logger.info(f"已输入卖出数量: {amount} 股")

            # 点击卖出按钮
            self.simulate_mouse_click(*self.ui_elements['sell_button'])
            logger.info("已点击卖出按钮")

            # # 验证确认对话框
            # if not self.verify_dialog_content(stock_code):
            #     # 点击取消按钮
            #     self.simulate_mouse_click(*self.ui_elements['cancel_button'])
            #     return False, "确认对话框内容验证失败"

            # 点击第一个确认按钮
            self.simulate_mouse_click(*self.ui_elements['confirm_button'])
            logger.info("已点击第一个确认按钮")

            # 点击第二个确认按钮（如果有）
            if 'second_confirm' in self.ui_elements:
                self.simulate_mouse_click(*self.ui_elements['second_confirm'])
                logger.info("已点击第二个确认按钮")

            # 等待操作完成 - 添加0.5秒延迟，确保交易完成
            logger.info("等待0.5秒，确保交易完成和数据库更新...")
            time.sleep(0.5)

            # 记录本次交易
            self.last_trade_time = datetime.now()
            trade_info = {
                'type': 'sell',
                'stock_code': stock_code,
                'time': self.last_trade_time.strftime('%Y-%m-%d %H:%M:%S')
            }

            if price:
                trade_info['price'] = adjusted_price
            if amount:
                trade_info['amount'] = amount

            self.trade_records.append(trade_info)

            return True, f"卖出 {stock_code} 成功"

        except Exception as e:
            logger.error(f"执行卖出订单出错: {str(e)}")
            logger.error(traceback.format_exc())
            return False, f"执行卖出订单出错: {str(e)}"
        finally:
            # 解除交易锁
            self.is_trading = False
            self.current_trading_stock = None
            logger.info("交易完成，解除交易锁")

    def can_execute_trade(self):
        """检查是否可以执行交易"""
        # 检查是否启用交易
        if not self.trading_enabled:
            logger.warning("交易功能当前已禁用")
            return False

        # 检查交易次数是否达到每日上限
        today_trades = [r for r in self.trade_records
                        if r['time'].startswith(datetime.now().strftime('%Y-%m-%d'))]
        if len(today_trades) >= self.trade_config['max_trades_per_day']:
            logger.warning(f"今日交易次数已达上限: {self.trade_config['max_trades_per_day']}")
            return False

        # 不再检查交易间隔
        # 如果上次交易时间存在，记录间隔但不限制交易
        if self.last_trade_time:
            elapsed = (datetime.now() - self.last_trade_time).total_seconds()
            if elapsed < self.trade_config['min_interval']:
                # 只记录日志，不阻止交易
                logger.info(f"距上次交易间隔 {elapsed:.2f} 秒，允许连续交易")

        return True

    def is_trading_time(self):
        """检查当前是否为交易时段"""
        now = datetime.now()
        weekday = now.weekday()

        # 周末不交易
        # if weekday >= 5:  # 5是周六，6是周日
        #     return False

        # 获取当前时间的小时和分钟
        current_time = now.time()

        # 上午交易时段：9:30 - 11:30
        morning_start = datetime.strptime("0:30", "%H:%M").time()
        morning_end = datetime.strptime("11:30", "%H:%M").time()

        # 下午交易时段：13:00 - 15:00
        afternoon_start = datetime.strptime("13:00", "%H:%M").time()
        afternoon_end = datetime.strptime("23:00", "%H:%M").time()

        # 判断当前时间是否在交易时段内
        is_trading = ((current_time >= morning_start and current_time <= morning_end) or
                      (current_time >= afternoon_start and current_time <= afternoon_end))

        return is_trading

    def process_pending_trades(self):
        """处理所有待执行的交易"""
        # 获取所有待执行的交易信号
        signals = self.get_pending_trades()
        if not signals:
            return 0

        processed_count = 0

        # 添加强制释放交易锁的检查
        if self.is_trading:
            # 检查是否为上次交易时间添加了时间戳属性
            if hasattr(self, 'lock_time') and self.lock_time:
                elapsed = (datetime.now() - self.lock_time).total_seconds()
                if elapsed > 300:  # 5分钟
                    logger.warning(f"检测到交易锁被持有超过5分钟，强制释放")
                    self.force_release_trading_lock()
            else:
                # 如果没有lock_time属性，添加一个
                self.lock_time = datetime.now()

        # 不再在每次处理时尝试激活窗口，假设窗口已在程序启动时激活
        for signal in signals:
            signal_id = signal['id']
            stock_code = signal['stock_code']
            stock_name = signal['stock_name']
            signal_type = signal['signal_type']

            logger.info(f"处理交易信号 #{signal_id}: {signal_type} {stock_name}({stock_code})")

            # 检查是否为交易时段
            if not self.is_trading_time():
                logger.warning("当前非交易时段，跳过执行")
                self.update_signal_status(signal_id, 'postponed', '非交易时段')
                continue

            # 检查是否可以执行交易
            if not self.can_execute_trade():
                logger.warning("当前不满足交易条件，跳过执行")
                self.update_signal_status(signal_id, 'postponed', '不满足交易条件')
                continue

            # 检查是否有其他交易正在进行
            if self.is_trading:
                current_op = self.current_trading_stock or "未知操作"
                logger.warning(f"当前正在执行操作: {current_op}，将推迟交易 {stock_name}({stock_code})")
                self.update_signal_status(signal_id, 'postponed', f'系统忙：{current_op}')
                continue

            # 根据信号类型执行交易
            success = False
            message = ""
            trade_info = {
                'stock_code': stock_code,
                'stock_name': stock_name,
                'signal_type': signal_type
            }

            try:
                if signal_type == 'buy':
                    # 获取当前价格
                    price = signal.get('price')
                    if not price:
                        # 这里简化处理，实际应从行情接口获取最新价格
                        price = 10.0  # 示例价格

                    # 获取买入数量或金额
                    amount = signal.get('amount')
                    max_amount = self.trade_config['max_amount_per_trade']

                    # 执行买入
                    success, message = self.execute_buy_order(
                        stock_code, price, amount, max_amount)

                    trade_info['price'] = price
                    if amount:
                        trade_info['amount'] = amount
                    else:
                        trade_info['max_amount'] = max_amount

                    # 买入成功后，更新状态但仍保持PENDING，以便后续卖出
                    if success:
                        self.update_signal_status(signal_id, 'pending', message)
                        # 记录交易结果
                        self.log_trade_result(signal_id, success, trade_info, None)
                        # 保存买入交易记录到trade_history表
                        self.save_trade_history(signal_id, is_buy=True, price=price)
                        logger.info(f"买入成功，保持PENDING状态以便后续卖出: {stock_name}({stock_code})")
                        processed_count += 1
                        continue  # 跳过后续的状态更新

                elif signal_type == 'sell':
                    # 实现卖出逻辑而不是直接跳过
                    # 获取卖出参数
                    price = signal.get('price')
                    amount = signal.get('amount')

                    # 执行卖出
                    success, message = self.execute_sell_order(
                        stock_code, price, amount)

                    # 确保卖出价格是实际的卖出价格（可能略低于请求价格）
                    # 由于execute_sell_order内部会调整卖出价格，我们需要获取实际使用的价格
                    actual_sell_price = price
                    if price:
                        # 略低于市价0.2%，与execute_sell_order中的调整一致
                        actual_sell_price = round(price * (1 - self.trade_config['price_adjust_pct']), 2)
                        logger.info(f"调整后的卖出价格: {actual_sell_price} (原价: {price})")
                        trade_info['price'] = actual_sell_price  # 使用调整后的实际卖出价格

                    if amount:
                        trade_info['amount'] = amount

                    # 卖出成功后，更新状态为completed并处理相关记录
                    if success:
                        # 先记录交易结果
                        self.log_trade_result(signal_id, success, trade_info, None)

                        # 更新信号状态为completed
                        self.update_signal_status(signal_id, 'completed', message)

                        # 保存卖出交易记录到trade_history表
                        trade_history_saved = self.save_trade_history(signal_id, is_buy=False, override_check=True, price=price)

                        # 添加小延时，确保数据库事务完成
                        time.sleep(0.5)

                        # 只有在成功保存交易历史后才删除交易记录
                        if trade_history_saved:
                            # 确认交易历史已保存，再删除交易记录
                            if self.delete_trade_record(signal_id):
                                logger.info(f"卖出成功，交易历史已保存，已删除交易记录: {stock_name}({stock_code})")
                            else:
                                logger.warning(f"卖出成功，交易历史已保存，但删除交易记录失败: {stock_name}({stock_code})")
                        else:
                            logger.warning(f"卖出成功，但保存交易历史失败，不删除交易记录: {stock_name}({stock_code})")

                        processed_count += 1
                        continue  # 跳过后续的状态更新
                else:
                    message = f"未知的信号类型: {signal_type}"

            except Exception as e:
                message = f"执行交易出错: {str(e)}"
                logger.error(message)
                logger.error(traceback.format_exc())

            # 更新信号状态（仅在上面没有特殊处理的情况下执行）
            new_status = 'completed' if success else 'failed'
            self.update_signal_status(signal_id, new_status, message)

            # 记录交易结果（仅在上面没有特殊处理的情况下执行）
            self.log_trade_result(signal_id, success, trade_info,
                                  None if success else message)

            processed_count += 1

        return processed_count

    def force_release_trading_lock(self):
        """强制释放交易锁"""
        if self.is_trading:
            previous_stock = self.current_trading_stock or "未知股票"
            logger.warning(f"强制释放交易锁，之前的操作: {previous_stock}")
            self.is_trading = False
            self.current_trading_stock = None
            # 重置锁定时间
            self.lock_time = None
            return True
        return False

    def delete_trade_record(self, signal_id):
        """删除交易记录"""
        if not hasattr(self, 'db_cursor') or self.db_cursor is None:
            logger.warning(f"数据库游标未初始化，无法删除交易记录 #{signal_id}")
            return False

        try:
            if not self.reconnect_db_if_needed():
                return False

            # 删除交易记录
            query = "DELETE FROM trading_signals WHERE id = %s"
            self.db_cursor.execute(query, (signal_id,))
            self.db_connection.commit()

            logger.info(f"成功删除交易记录 #{signal_id}")
            return True
        except Exception as e:
            logger.error(f"删除交易记录出错: {str(e)}")
            logger.error(traceback.format_exc())
            return False

    def create_trading_logs_table_if_not_exists(self):
        """创建交易执行日志表（如果不存在）"""
        if not hasattr(self, 'db_cursor') or self.db_cursor is None:
            logger.warning("数据库游标未初始化，无法创建交易日志表")
            return False

        try:
            if not self.reconnect_db_if_needed():
                return False

            query = """
            CREATE TABLE IF NOT EXISTS trade_execution_logs (
                id INT AUTO_INCREMENT PRIMARY KEY,
                signal_id INT NOT NULL,
                trade_time DATETIME NOT NULL,
                success TINYINT(1) NOT NULL DEFAULT 0,
                trade_info JSON NOT NULL,
                error_message TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
            """

            self.db_cursor.execute(query)
            self.db_connection.commit()
            logger.info("交易执行日志表创建成功（如果不存在）")
            return True
        except Exception as e:
            logger.error(f"创建交易执行日志表出错: {str(e)}")
            return False

    def _create_default_template_images(self):
        """创建默认的模板图像，用于识别同花顺窗口"""
        try:
            logger.info("正在创建默认模板图像...")

            # 创建同花顺图标模板
            icon_path = os.path.join(self.images_dir, 'ths_icon.png')
            icon_img = Image.new('RGB', (40, 40), color=(255, 150, 50))  # 橙色背景
            draw = ImageDraw.Draw(icon_img)
            try:
                # 尝试加载字体
                font = ImageFont.truetype("simhei.ttf", 12)
            except:
                # 如果无法加载字体，使用默认字体
                font = ImageFont.load_default()

            draw.text((5, 15), "同花顺", fill=(255, 255, 255), font=font)
            icon_img.save(icon_path)
            logger.info(f"已创建同花顺图标模板: {icon_path}")

            # 创建同花顺标题栏模板
            header_path = os.path.join(self.images_dir, 'ths_header.png')
            header_img = Image.new('RGB', (200, 30), color=(50, 100, 240))  # 蓝色背景
            draw = ImageDraw.Draw(header_img)
            draw.text((20, 5), "网上股票交易系统5.0", fill=(255, 255, 255), font=font)
            header_img.save(header_path)
            logger.info(f"已创建同花顺标题栏模板: {header_path}")

        except Exception as e:
            logger.error(f"创建默认模板图像失败: {str(e)}")
            logger.warning("请手动提供同花顺窗口相关的截图，放置在images目录中")

    def _init_ui_elements(self):
        """初始化同花顺界面UI元素位置"""
        # 根据屏幕分辨率获取相对位置
        # 这些坐标需要根据实际的同花顺交易软件界面进行调整
        elements = {
            # 标签页位置
            'buy_tab': (190, 210),
            'sell_tab': (164, 230),

            # 输入框位置
            'stock_code_input': (555, 270),
            'stock_price_input': (555, 330),
            'stock_amount_input': (555, 420),

            # 按钮位置
            'buy_button': (610, 455),
            'sell_button': (610, 455),
            'confirm_button': (875, 763),
            'second_confirm': (950, 700),
            'cancel_button': (self.screen_width // 2 + 50, self.screen_height // 2 + 50),
        }

        logger.info("UI元素位置初始化完成")
        return elements

    def start_trading_software(self):
        """启动同花顺交易软件"""
        if self.software_started:
            logger.info("同花顺软件已经启动，无需重复启动")
            return True

        logger.info("尝试启动同花顺交易软件...")
        try:
            # 常见的同花顺安装路径
            possible_paths = [
                r"E:\同花顺\同花顺\xiadan.exe",
            ]

            # 添加配置文件中的路径（如果有）
            ths_path = self.config.get("ths_path", "")
            if ths_path:
                possible_paths.insert(0, ths_path)

            # 尝试每一个可能的路径
            for path in possible_paths:
                if os.path.exists(path):
                    logger.info(f"找到同花顺程序路径: {path}")
                    os.startfile(path)
                    logger.info("已启动同花顺程序，等待其运行...")
                    time.sleep(10)  # 等待程序启动

                    # 尝试激活窗口
                    if self.find_and_activate_window(no_start=True):
                        self.software_started = True
                        logger.info("同花顺软件已成功启动并激活")
                        return True
                    break

            if not self.software_started:
                logger.error("无法找到或启动同花顺程序，请在配置文件中指定正确的 ths_path")
                return False

        except Exception as e:
            logger.error(f"启动同花顺程序失败: {str(e)}")
            return False

    def save_trade_history(self, signal_id, is_buy=True, override_check=True, price=None):
        """保存交易记录到trade_history表

        Args:
            signal_id: 交易信号ID
            is_buy: 是否为买入操作，True为买入，False为卖出
            override_check: 是否忽略售出状态检查
            price: 价格（如果为None则使用数据库中的值）

        Returns:
            bool: 是否成功保存
        """
        try:
            # 导入KellyPositionManager
            from kelly_position_manager import KellyPositionManager

            # 初始化KellyPositionManager
            config_path = "kelly_config.json"
            kelly_manager = KellyPositionManager(config_path)

            # 在卖出前，确保交易信号已更新
            if not is_buy and override_check:
                # 首先获取交易信号数据
                if not self.reconnect_db_if_needed():
                    logger.error("数据库连接失败，无法获取交易信号数据")
                    return False

                # 如果价格为None，则获取当前价格
                if price is None:
                    # 查询交易信号
                    query = "SELECT sell_price FROM trading_signals WHERE id = %s"
                    self.db_cursor.execute(query, (signal_id,))
                    signal_data = self.db_cursor.fetchone()

                    if signal_data and signal_data['sell_price']:
                        price = signal_data['sell_price']

                # 强制更新交易信号为已售出状态
                update_query = """
                UPDATE trading_signals 
                SET is_sold = TRUE, 
                    sell_time = NOW()
                WHERE id = %s
                """
                self.db_cursor.execute(update_query, (signal_id,))
                self.db_connection.commit()

                logger.info(f"更新信号 #{signal_id} 为已售出状态以确保交易记录能够保存")

            # 调用save_trade方法保存交易记录
            result = kelly_manager.save_trade(signal_id, is_buy, override_check, price)

            if result:
                logger.info(f"成功保存交易记录到trade_history表，信号ID: {signal_id}, 类型: {'买入' if is_buy else '卖出'}")
            else:
                logger.warning(f"保存交易记录到trade_history表失败，信号ID: {signal_id}")

            return result
        except Exception as e:
            logger.error(f"保存交易记录到trade_history表时出错: {str(e)}")
            logger.error(traceback.format_exc())
            return False

    def update_signal_after_buy(self, signal_id, actual_buy_price, actual_quantity=None):
        """买入交易执行后更新交易信号

        Args:
            signal_id: 交易信号ID
            actual_buy_price: 实际买入价格
            actual_quantity: 实际买入数量，如果为None则使用信号中的数量

        Returns:
            bool: 是否成功更新
        """
        try:
            if not self.reconnect_db_if_needed():
                logger.error("数据库连接失败，无法更新交易信号")
                return False

            # 首先获取交易信号信息
            self.cursor.execute("SELECT buy_quantity FROM trading_signals WHERE id = %s", (signal_id,))
            signal = self.cursor.fetchone()

            if not signal:
                logger.warning(f"未找到ID为{signal_id}的交易信号")
                return False

            # 如果未提供实际数量，使用信号中的数量
            if actual_quantity is None:
                actual_quantity = signal['buy_quantity']

            # 更新交易信号
            update_query = """
            UPDATE trading_signals SET
                is_bought = TRUE,
                buy_price = %s,
                buy_quantity = %s,
                buy_time = NOW(),
                trade_status = 'BOUGHT',
                last_update_time = NOW()
            WHERE id = %s
            """

            self.cursor.execute(update_query, (actual_buy_price, actual_quantity, signal_id))
            self.conn.commit()

            # 保存交易记录并更新可用资金
            total_amount = actual_buy_price * actual_quantity
            self.update_available_capital(total_amount, is_buy=True)
            self.save_trade(signal_id, is_buy=True)

            # 保存交易记录到trade_history表
            self.save_trade_history(signal_id, is_buy=True)

            logger.info(f"成功更新买入信号 #{signal_id}: 价格: {actual_buy_price}, 数量: {actual_quantity}, 金额: {total_amount:.2f}元")
            return True

        except Exception as e:
            logger.error(f"更新买入信号失败: {e}")
            logger.error(traceback.format_exc())
            return False

    def update_signal_after_sell(self, signal_id, actual_sell_price):
        """卖出交易执行后更新交易信号

        Args:
            signal_id: 交易信号ID
            actual_sell_price: 实际卖出价格

        Returns:
            bool: 是否成功更新
        """
        try:
            if not self.reconnect_db_if_needed():
                logger.error("数据库连接失败，无法更新交易信号")
                return False

            # 更新交易信号
            update_query = """
            UPDATE trading_signals SET
                is_sold = TRUE,
                sell_price = %s,
                sell_time = NOW(),
                trade_status = 'COMPLETED',
                last_update_time = NOW()
            WHERE id = %s
            """

            self.cursor.execute(update_query, (actual_sell_price, signal_id))
            self.conn.commit()

            # 保存交易记录
            self.save_trade(signal_id, is_buy=False)

            # 保存交易记录到trade_history表
            self.save_trade_history(signal_id, is_buy=False, override_check=True, price=actual_sell_price)

            logger.info(f"成功更新卖出信号 #{signal_id}: 价格: {actual_sell_price}")
            return True

        except Exception as e:
            logger.error(f"更新卖出信号失败: {e}")
            logger.error(traceback.format_exc())
            return False


def main():
    """主函数"""
    print("初始化同花顺自动交易系统...")

    # 检查必要的依赖库
    try:
        print("检查依赖库...")
        import pyautogui
        import pyperclip
        from PIL import ImageGrab, Image, ImageEnhance, ImageFilter
        import pytesseract

        # 检查是否安装了OpenCV（可选但推荐）
        try:
            import cv2
            has_opencv = True
            print("检测到OpenCV库，可以使用高级图像识别功能")
        except ImportError:
            has_opencv = False
            print("\n警告: 未检测到OpenCV库，图像识别功能将受限")
            print("建议安装OpenCV以获得更好的图像识别效果:")
            print("pip install opencv-python\n")

        # 尝试导入pygetwindow库，用于窗口操作
        try:
            import pygetwindow as gw
            has_pygetwindow = True
            print("检测到pygetwindow库，将使用它进行窗口操作")
        except ImportError:
            has_pygetwindow = False
            print("未检测到pygetwindow库，将使用图像识别方法查找窗口")
            print("建议安装pygetwindow以提高窗口查找的准确性：pip install pygetwindow")

        print("依赖库检查通过")
    except ImportError as e:
        print(f"缺少必要的依赖库: {str(e)}")
        print("请运行以下命令安装依赖库:")
        print("pip install pyautogui pyperclip pillow pytesseract mysql-connector-python pygetwindow opencv-python")
        return

    # 检查配置文件是否存在
    config_file = "ths_config.json"
    if not os.path.exists(config_file):
        print(f"配置文件 {config_file} 不存在，创建示例配置...")
        # 创建示例配置
        example_config = {
            "ths_main_title": "网上股票交易系统5.0",
            "ths_trade_title": "同花顺版",
            "ths_path": "E:\\同花顺\\同花顺\\xiadan.exe",
            "verify_code_region": [710, 340, 780, 370],
            "verify_code_color": [0, 0, 0],
            "max_retry": 3,
            "simulate_human": True,
            "use_tesseract": True,
            "tesseract_cmd": "..\\tesseract.exe",
            "mysql_config": {
                "host": "localhost",
                "port": 3306,
                "user": "root",
                "password": "your_password",
                "database": "stock_analysis"
            },
            "trade_config": {
                "max_trades_per_day": 5,
                "max_amount_per_trade": 10000,
                "min_interval": 30,
                "confirm_timeout": 5,
                "price_adjust_pct": 0.002
            }
        }

        try:
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(example_config, f, ensure_ascii=False, indent=4)
            print(f"已创建示例配置文件：{config_file}")
            print("请修改配置文件中的参数，特别是数据库连接信息，然后重新运行程序")

            # 输出UI元素位置配置建议
            print("\n注意：您需要根据实际的同花顺交易软件界面调整UI元素位置")
            print("以下是UI元素位置的示例（需要根据实际情况调整）：")
            print("'buy_tab': (300, 200) - 买入标签页位置")
            print("'sell_tab': (400, 200) - 卖出标签页位置")
            print("'stock_code_input': (350, 250) - 股票代码输入框位置")
            print("'stock_price_input': (350, 300) - 股票价格输入框位置")
            print("'stock_amount_input': (350, 350) - 股票数量输入框位置")
            print("'buy_button': (300, 400) - 买入按钮位置")
            print("'sell_button': (300, 400) - 卖出按钮位置")
            print("'confirm_button': (屏幕宽度/2-50, 屏幕高度/2+50) - 确认按钮位置")
            print("'cancel_button': (屏幕宽度/2+50, 屏幕高度/2+50) - 取消按钮位置")

            return
        except Exception as e:
            print(f"创建示例配置文件失败: {str(e)}")

    trader = THSAutoTrader(config_file)

    # 设置检查间隔为10秒
    check_interval = 10

    try:
        # 启动交易监控
        print(f"启动自动交易监控，检查间隔: {check_interval} 秒")
        trader.run(interval=check_interval)
    except Exception as e:
        print(f"运行出错: {str(e)}")
        traceback.print_exc()
    finally:
        print("程序已退出")


if __name__ == "__main__":
    main()