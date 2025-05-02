#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import time
import math
import logging
import mysql.connector
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import traceback

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('kelly_position.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


class KellyPositionManager:
    """使用凯利公式管理仓位和设置止盈止损"""

    def __init__(self, config_path=None):
        """初始化仓位管理器

        Args:
            config_path: 配置文件路径，默认为None
        """
        logger.info("初始化凯利公式仓位管理器...")

        # 默认配置
        self.default_config = {
            "mysql_config": {
                "host": "localhost",
                "port": 3306,
                "user": "root",
                "password": "zyb123456668866",
                "database": "stock_analysis"
            },
            "kelly_config": {
                "default_win_rate": 0.55,  # 默认胜率
                "max_position_ratio": 0.3,  # 最大仓位比例上限
                "min_position_ratio": 0.05,  # 最小仓位比例下限
                "half_kelly": True,  # 是否使用半凯利公式（更保守）
                "stop_loss_ratio": 0.05,  # 默认止损比例 (5%)
                "take_profit_ratio": 0.02,  # 默认止盈比例 (10%)
                "max_kelly_score": 0.5,  # 凯利公式计算结果最大值上限
                "win_loss_ratio": 1.5  # 默认盈亏比
            },
            "trade_settings": {
                "total_capital": 100000,  # 总资金
                "available_capital": 80000,  # 可用资金
                "max_stocks": 5,  # 最多持有股票数
                "min_score_to_buy": 80,  # 最低买入得分
                "process_interval": 1,  # 处理间隔（秒）
                "trading_fee_rate": 0.0005  # 交易费率（买卖双向）
            }
        }

        # 初始化配置
        self.config = self.default_config

        # 加载配置文件
        if config_path:
            self._load_config(config_path)

        # 初始化总资金和可用资金
        self.total_capital = self.config['trade_settings']['total_capital']
        self.available_capital = self.config['trade_settings']['available_capital']
        logger.info(f"总资金: {self.total_capital}元，可用资金: {self.available_capital}元")

        # 初始化数据库连接
        self.conn = None
        self.cursor = None
        self.connect_to_db()

        # 历史胜率数据缓存
        self.win_rate_cache = {}

        logger.info("凯利公式仓位管理器初始化完成")

    def _load_config(self, config_path):
        """加载配置文件

        Args:
            config_path: 配置文件路径
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                user_config = json.load(f)

            # 合并配置
            if 'mysql_config' in user_config:
                self.config['mysql_config'].update(user_config['mysql_config'])

            if 'kelly_config' in user_config:
                self.config['kelly_config'].update(user_config['kelly_config'])

            if 'trade_settings' in user_config:
                self.config['trade_settings'].update(user_config['trade_settings'])

            logger.info(f"配置文件 {config_path} 加载成功")
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            logger.error(traceback.format_exc())

    def connect_to_db(self):
        """连接到数据库"""
        try:
            if self.conn and self.conn.is_connected():
                logger.info("数据库已连接")
                return True

            self.conn = mysql.connector.connect(
                host=self.config['mysql_config']['host'],
                port=self.config['mysql_config'].get('port', 3306),
                user=self.config['mysql_config']['user'],
                password=self.config['mysql_config']['password'],
                database=self.config['mysql_config']['database']
            )
            self.cursor = self.conn.cursor(dictionary=True)
            logger.info("数据库连接成功")

            # 确保必要的表已创建
            self.create_tables()

            return True
        except Exception as e:
            logger.error(f"数据库连接失败: {e}")
            logger.error(traceback.format_exc())
            return False

    def reconnect_if_needed(self):
        """检查数据库连接，必要时重连"""
        try:
            if not self.conn or not self.conn.is_connected():
                logger.info("数据库连接已断开，尝试重连...")
                return self.connect_to_db()
            return True
        except Exception as e:
            logger.error(f"检查数据库连接失败: {e}")
            return self.connect_to_db()

    def create_tables(self):
        """创建必要的数据库表"""
        try:
            # 创建交易历史表，用于跟踪交易胜率
            self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS trade_history (
                id INT AUTO_INCREMENT PRIMARY KEY,
                stock_code VARCHAR(10) NOT NULL COMMENT '股票代码',
                stock_name VARCHAR(50) NOT NULL COMMENT '股票名称',
                buy_price DECIMAL(10,2) NOT NULL COMMENT '买入价格',
                sell_price DECIMAL(10,2) COMMENT '卖出价格',
                quantity INT NOT NULL COMMENT '买入数量',
                profit_rate DECIMAL(10,4) COMMENT '盈亏率(%)',
                profit_amount DECIMAL(10,2) COMMENT '盈亏金额',
                buy_capital DECIMAL(12,2) COMMENT '买入后剩余资金',
                sell_capital DECIMAL(12,2) COMMENT '卖出后剩余资金',
                trade_date DATE COMMENT '交易日期',
                buy_time DATETIME COMMENT '买入时间',
                sell_time DATETIME COMMENT '卖出时间',
                is_win BOOLEAN COMMENT '是否盈利',
                actual_position FLOAT COMMENT '实际仓位比例',
                trading_signal_id INT COMMENT '关联的交易信号ID',
                INDEX(stock_code),
                INDEX(trade_date)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='交易历史记录表';
            """)
            self.conn.commit()
            logger.info("交易历史表创建或检查完成")

            # 检查表是否创建成功
            self.cursor.execute("SHOW TABLES LIKE 'trade_history'")
            if self.cursor.fetchone():
                logger.info("确认trade_history表已存在")
            else:
                logger.error("表创建操作执行成功，但trade_history表仍不存在，可能存在权限问题")

            # 创建止盈止损配置表
            self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS stop_limit_settings (
                id INT AUTO_INCREMENT PRIMARY KEY,
                stock_code VARCHAR(10) NOT NULL,
                stock_name VARCHAR(50) NOT NULL,
                stop_loss_ratio FLOAT NOT NULL,
                take_profit_ratio FLOAT NOT NULL,
                dynamic_adjustment BOOLEAN DEFAULT FALSE,
                trailing_stop BOOLEAN DEFAULT FALSE,
                trailing_percent FLOAT,
                signal_id INT,
                is_active BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                UNIQUE KEY (stock_code, signal_id)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """)
            self.conn.commit()

            # 修改trading_signals表以增加必要字段
            # 注意：这将在stock_analysis_decision.py的create_trading_signals_table方法中进行

            logger.info("数据库表创建或检查完成")
            return True
        except Exception as e:
            logger.error(f"创建表失败: {e}")
            logger.error(traceback.format_exc())
            return False

    def get_historical_win_rate(self, stock_code):
        """获取股票的历史交易胜率 - 简化版本，直接返回默认值

        Args:
            stock_code: 股票代码

        Returns:
            默认胜率和盈亏比
        """
        # 直接使用配置中的默认值
        default_win_rate = self.config['kelly_config']['default_win_rate']
        default_win_loss_ratio = self.config['kelly_config']['win_loss_ratio']

        logger.info(f"使用默认胜率: {default_win_rate:.2f}, 盈亏比: {default_win_loss_ratio:.2f}")
        return default_win_rate, default_win_loss_ratio

    def calculate_kelly_position(self, win_rate, win_loss_ratio):
        """使用凯利公式计算最佳仓位比例

        凯利公式: f* = (p*b - q)/b
        其中:
        - f*: 最佳仓位比例
        - p: 获胜概率
        - q: 失败概率 (1-p)
        - b: 获胜时的盈亏比

        Args:
            win_rate: 胜率 (0-1)
            win_loss_ratio: 盈亏比

        Returns:
            最佳仓位比例 (0-1)
        """
        try:
            # 验证输入
            if win_rate <= 0 or win_rate >= 1:
                logger.warning(f"无效的胜率值: {win_rate}，使用默认值: 0.55")
                win_rate = 0.55

            if win_loss_ratio <= 0:
                logger.warning(f"无效的盈亏比: {win_loss_ratio}，使用默认值: 1.5")
                win_loss_ratio = 1.5

            # 计算凯利值
            kelly = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio

            # 处理负值
            if kelly < 0:
                logger.warning(f"凯利计算结果为负值: {kelly}，调整为0")
                kelly = 0

            # 限制最大值
            max_kelly = self.config['kelly_config']['max_kelly_score']
            if kelly > max_kelly:
                logger.info(f"凯利计算结果 {kelly:.4f} 超过上限 {max_kelly}，已调整")
                kelly = max_kelly

            # 如果启用半凯利，则使用更保守的策略
            if self.config['kelly_config']['half_kelly']:
                kelly = kelly / 2
                logger.info(f"使用半凯利公式，仓位比例: {kelly:.4f}")

            # 应用配置的仓位限制
            min_ratio = self.config['kelly_config']['min_position_ratio']
            max_ratio = self.config['kelly_config']['max_position_ratio']

            if kelly < min_ratio:
                kelly = min_ratio
                logger.info(f"凯利计算结果低于最小仓位比例，调整为: {min_ratio}")
            elif kelly > max_ratio:
                kelly = max_ratio
                logger.info(f"凯利计算结果高于最大仓位比例，调整为: {max_ratio}")

            return kelly

        except Exception as e:
            logger.error(f"计算凯利仓位失败: {e}")
            logger.error(traceback.format_exc())
            # 返回默认的保守仓位
            return self.config['kelly_config']['min_position_ratio']

    def calculate_stop_loss_take_profit(self, current_price, stock_code, win_rate=None, win_loss_ratio=None):
        """计算止损和止盈点位

        Args:
            current_price: 当前股价
            stock_code: 股票代码
            win_rate: 胜率，默认为None（将使用默认值）
            win_loss_ratio: 盈亏比，默认为None（将使用默认值）

        Returns:
            (止损价, 止盈价)元组
        """
        try:
            # 获取股票特定的止损止盈设置
            query = """
            SELECT stop_loss_ratio, take_profit_ratio 
            FROM stop_limit_settings
            WHERE stock_code = %s AND is_active = TRUE
            ORDER BY updated_at DESC
            LIMIT 1
            """
            self.cursor.execute(query, (stock_code,))
            custom_setting = self.cursor.fetchone()

            # 使用自定义设置或默认值
            if custom_setting:
                stop_loss_ratio = custom_setting['stop_loss_ratio']
                take_profit_ratio = custom_setting['take_profit_ratio']
                logger.info(f"使用股票 {stock_code} 的自定义止损止盈设置: 止损 {stop_loss_ratio:.2%}, 止盈 {take_profit_ratio:.2%}")
            else:
                # 使用默认值
                stop_loss_ratio = self.config['kelly_config']['stop_loss_ratio']
                take_profit_ratio = self.config['kelly_config']['take_profit_ratio']
                logger.info(f"使用默认止损止盈比例: 止损 {stop_loss_ratio:.2%}, 止盈 {take_profit_ratio:.2%}")

            # 计算实际价格
            stop_loss_price = round(current_price * (1 - stop_loss_ratio), 2)
            take_profit_price = round(current_price * (1 + take_profit_ratio), 2)

            return stop_loss_price, take_profit_price

        except Exception as e:
            logger.error(f"计算止损止盈失败: {e}")
            logger.error(traceback.format_exc())

            # 使用默认值
            default_stop_loss = round(current_price * (1 - self.config['kelly_config']['stop_loss_ratio']), 2)
            default_take_profit = round(current_price * (1 + self.config['kelly_config']['take_profit_ratio']), 2)

            return default_stop_loss, default_take_profit

    def get_score_factor(self, score):
        """根据股票分析得分计算仓位调整因子

        Args:
            score: 股票分析得分 (0-100)

        Returns:
            仓位调整因子 (0.5-1.5)
        """
        try:
            # 验证输入
            if score < 0 or score > 100:
                logger.warning(f"无效的得分: {score}，使用默认值")
                return 1.0

            # 将得分映射到0.5-1.5的范围
            # 得分80以上开始增加仓位，低于80减少仓位
            min_score = self.config['trade_settings']['min_score_to_buy']

            if score < min_score:
                return 0.5  # 得分不够，减半仓位
            else:
                # 线性映射: 80->1.0, 100->1.5
                factor = 1.0 + (score - min_score) / (100 - min_score) * 0.5
                return factor

        except Exception as e:
            logger.error(f"计算得分因子失败: {e}")
            return 1.0

    def adjust_position_by_risk(self, base_position, realtime_score, daily_score, fundamental_score):
        """根据不同维度的风险评分调整仓位

        Args:
            base_position: 基础仓位比例 (0-1)
            realtime_score: 实时分析得分
            daily_score: 日线分析得分
            fundamental_score: 基本面分析得分

        Returns:
            调整后的仓位比例 (0-1)
        """
        try:
            # 检查输入
            if base_position <= 0 or base_position > 1:
                logger.warning(f"无效的基础仓位: {base_position}，使用默认值")
                base_position = 0.1

            # 各项评分的权重
            realtime_weight = 0.3
            daily_weight = 0.4
            fundamental_weight = 0.3

            # 计算总评分（加权平均）
            weights_sum = 0
            weighted_score = 0

            if realtime_score is not None:
                weighted_score += realtime_score * realtime_weight
                weights_sum += realtime_weight

            if daily_score is not None:
                weighted_score += daily_score * daily_weight
                weights_sum += daily_weight

            if fundamental_score is not None:
                weighted_score += fundamental_score * fundamental_weight
                weights_sum += fundamental_weight

            # 如果没有足够的评分数据，返回原始仓位
            if weights_sum == 0:
                return base_position

            # 归一化得分
            avg_score = weighted_score / weights_sum

            # 计算调整因子 (0.8-1.2)
            # 得分越高，仓位越大
            adjustment_factor = 0.8 + (avg_score / 100) * 0.4

            # 应用调整
            adjusted_position = base_position * adjustment_factor

            # 确保不超过最大仓位限制
            max_position = self.config['kelly_config']['max_position_ratio']
            if adjusted_position > max_position:
                adjusted_position = max_position

            logger.info(f"仓位调整: 基础 {base_position:.4f} -> 调整后 {adjusted_position:.4f} (调整因子: {adjustment_factor:.2f})")
            return adjusted_position

        except Exception as e:
            logger.error(f"调整仓位失败: {e}")
            logger.error(traceback.format_exc())
            return base_position

    def calculate_quantity(self, price, position_ratio):
        """计算买入数量

        Args:
            price: 股票价格
            position_ratio: 仓位比例 (0-1)

        Returns:
            买入数量（股）
        """
        try:
            # 计算买入金额
            buy_amount = self.available_capital * position_ratio

            # 计算可买股数（必须是100的整数倍）
            quantity = int(buy_amount / price / 100) * 100

            # 确保至少买入100股
            if quantity < 100:
                logger.warning(f"计算得到的数量 {quantity} 小于100股，调整为100股")
                quantity = 100

            # 验证资金是否足够
            actual_amount = quantity * price
            fee = actual_amount * self.config['trade_settings']['trading_fee_rate']
            total_cost = actual_amount + fee

            if total_cost > self.available_capital:
                logger.warning(f"资金不足: 需要 {total_cost:.2f}元, 可用 {self.available_capital:.2f}元")
                # 调整为可承担的最大数量
                max_affordable = int((self.available_capital / (price * (1 + self.config['trade_settings']['trading_fee_rate']))) / 100) * 100
                if max_affordable >= 100:
                    quantity = max_affordable
                    logger.info(f"已调整买入数量为 {quantity} 股")
                else:
                    logger.warning(f"可用资金不足以购买最小单位(100股)，设置为0")
                    quantity = 0

            return quantity

        except Exception as e:
            logger.error(f"计算买入数量失败: {e}")
            logger.error(traceback.format_exc())
            # 返回默认的100股
            return 100

    def process_buy_signals(self):
        """处理所有待买入信号

        计算凯利公式仓位、止盈止损，并更新交易信号表
        """
        try:
            if not self.reconnect_if_needed():
                logger.error("数据库连接失败，无法处理买入信号")
                return 0

            # 查询未处理的买入信号 - 排除已计算的信号
            query = """
            SELECT * FROM trading_signals 
            WHERE is_bought = FALSE 
              AND trade_status = 'PENDING'
              AND (target_position IS NULL OR position_calculated = FALSE)
              AND score >= %s
            ORDER BY score DESC, analysis_time DESC
            """
            min_score = self.config['trade_settings']['min_score_to_buy']
            self.cursor.execute(query, (min_score,))
            buy_signals = self.cursor.fetchall()

            logger.info(f"找到 {len(buy_signals)} 条待处理的买入信号")

            processed_count = 0
            for signal in buy_signals:
                signal_id = signal['id']
                stock_code = signal['stock_code']
                stock_name = signal['stock_name']
                score = signal['score']
                current_price = signal['current_price']

                logger.info(f"处理买入信号 #{signal_id}: {stock_name}({stock_code}), 得分: {score:.2f}")

                # 1. 使用默认的胜率和盈亏比
                win_rate = self.config['kelly_config']['default_win_rate']
                win_loss_ratio = self.config['kelly_config']['win_loss_ratio']

                # 2. 计算凯利公式仓位
                kelly_position = self.calculate_kelly_position(win_rate, win_loss_ratio)

                # 3. 基于得分调整仓位
                score_factor = self.get_score_factor(score)
                base_position = kelly_position * score_factor

                # 4. 根据多维度风险调整仓位
                realtime_score = signal.get('realtime_score')
                daily_score = signal.get('daily_score')
                fundamental_score = signal.get('fundamental_score')

                final_position = self.adjust_position_by_risk(
                    base_position, realtime_score, daily_score, fundamental_score)

                # 5. 计算止损止盈点位
                stop_loss_price, take_profit_price = self.calculate_stop_loss_take_profit(
                    current_price, stock_code)

                # 6. 计算买入数量
                buy_quantity = self.calculate_quantity(current_price, final_position)

                # 如果计算出的数量为0，跳过该信号
                if buy_quantity <= 0:
                    logger.warning(f"信号 #{signal_id} 计算的买入数量为0，可能是资金不足，跳过处理")
                    continue

                # 7. 更新交易信号表 - 添加position_calculated标记
                update_query = """
                UPDATE trading_signals SET
                    target_position = %s,
                    stop_loss_price = %s,
                    take_profit_price = %s,
                    buy_quantity = %s,
                    portfolio_ratio = %s,
                    notes = %s,
                    position_calculated = TRUE,
                    last_update_time = NOW()
                WHERE id = %s
                """

                notes = (f"凯利仓位: {kelly_position:.4f}, 得分因子: {score_factor:.2f}, "
                         f"最终仓位: {final_position:.4f}")

                self.cursor.execute(update_query, (
                    final_position,
                    stop_loss_price,
                    take_profit_price,
                    buy_quantity,
                    final_position,
                    notes,
                    signal_id
                ))

                logger.info(f"更新信号 #{signal_id}: 目标仓位 {final_position:.4f}, "
                            f"止损价 {stop_loss_price:.2f}, 止盈价 {take_profit_price:.2f}, "
                            f"买入数量 {buy_quantity}")

                # 8. 计算并更新可用资金
                total_amount = current_price * buy_quantity
                self.update_available_capital(total_amount, is_buy=True)
                logger.info(f"信号 #{signal_id} 预计交易金额: {total_amount:.2f}元, 更新后可用资金: {self.available_capital:.2f}元")

                processed_count += 1

            # 提交事务
            self.conn.commit()
            logger.info(f"共处理 {processed_count} 条买入信号")

            return processed_count

        except Exception as e:
            logger.error(f"处理买入信号失败: {e}")
            logger.error(traceback.format_exc())
            return 0

    def run(self, interval=1):
        """启动仓位管理器主循环

        Args:
            interval: 检查间隔（秒）
        """
        logger.info(f"凯利公式仓位管理器启动，检查间隔: {interval}秒")

        try:
            while True:
                start_time = time.time()

                logger.info("=== 开始处理买入信号 ===")
                processed_count = self.process_buy_signals()

                logger.info(f"本次处理 {processed_count} 条买入信号")

                # 计算等待时间
                elapsed = time.time() - start_time
                wait_time = max(0, interval - elapsed)

                if wait_time > 0:
                    logger.info(f"等待 {wait_time:.2f} 秒后继续...")
                    time.sleep(wait_time)

        except KeyboardInterrupt:
            logger.info("检测到Ctrl+C，程序退出...")
        except Exception as e:
            logger.error(f"程序运行异常: {e}")
            logger.error(traceback.format_exc())
        finally:
            if self.cursor:
                self.cursor.close()
            if self.conn:
                self.conn.close()
            logger.info("程序已退出")

    def update_trading_signals_table(self):
        """更新trading_signals表结构，添加必要的字段"""
        try:
            if not self.reconnect_if_needed():
                return False

            # 检查表是否存在
            self.cursor.execute("SHOW TABLES LIKE 'trading_signals'")
            if not self.cursor.fetchone():
                logger.error("trading_signals表不存在，请先运行stock_analysis_decision.py创建该表")
                return False

            # 要添加的字段列表
            fields_to_add = [
                ("portfolio_ratio", "FLOAT DEFAULT NULL"),
                ("target_position", "FLOAT DEFAULT NULL"),
                ("stop_loss_price", "DECIMAL(10,2) DEFAULT NULL"),
                ("take_profit_price", "DECIMAL(10,2) DEFAULT NULL"),
                ("buy_quantity", "INT DEFAULT NULL"),
                ("position_calculated", "BOOLEAN DEFAULT FALSE"),
                ("last_update_time", "DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP")
            ]

            # 获取现有字段
            self.cursor.execute("DESCRIBE trading_signals")
            existing_fields = [row['Field'].lower() for row in self.cursor.fetchall()]

            # 添加缺失的字段
            for field_name, field_def in fields_to_add:
                if field_name.lower() not in existing_fields:
                    try:
                        alter_query = f"ALTER TABLE trading_signals ADD COLUMN {field_name} {field_def}"
                        self.cursor.execute(alter_query)
                        logger.info(f"已添加字段: {field_name}")
                    except Exception as e:
                        logger.error(f"添加字段 {field_name} 失败: {e}")

            self.conn.commit()
            logger.info("trading_signals表更新完成")
            return True

        except Exception as e:
            logger.error(f"更新表结构失败: {e}")
            logger.error(traceback.format_exc())
            return False

    def update_available_capital(self, amount, is_buy=True):
        """更新可用资金

        Args:
            amount: 交易金额
            is_buy: 是否为买入交易，True为买入（减少资金），False为卖出（增加资金）

        Returns:
            float: 更新后的可用资金
        """
        try:
            if not self.reconnect_if_needed():
                logger.error("数据库连接失败，无法更新可用资金")
                return self.available_capital

            # 计算交易费用
            fee_rate = self.config['trade_settings']['trading_fee_rate']
            fee = amount * fee_rate

            # 根据交易类型更新资金
            if is_buy:
                # 买入：减少可用资金（交易金额+费用）
                self.available_capital -= (amount + fee)
                logger.info(f"买入交易: 金额 {amount:.2f}元, 费用 {fee:.2f}元, 剩余可用资金 {self.available_capital:.2f}元")
            else:
                # 卖出：增加可用资金（交易金额-费用）
                self.available_capital += (amount - fee)
                logger.info(f"卖出交易: 金额 {amount:.2f}元, 费用 {fee:.2f}元, 剩余可用资金 {self.available_capital:.2f}元")

            # 确保可用资金不为负
            if self.available_capital < 0:
                logger.warning(f"可用资金计算为负值 {self.available_capital:.2f}元，调整为0")
                self.available_capital = 0

            # 更新配置文件中的可用资金数据
            self.config['trade_settings']['available_capital'] = self.available_capital
            self._save_config()

            return self.available_capital

        except Exception as e:
            logger.error(f"更新可用资金失败: {e}")
            logger.error(traceback.format_exc())
            return self.available_capital

    def _save_config(self):
        """保存当前配置到配置文件"""
        try:
            config_path = "kelly_config.json"
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, ensure_ascii=False, indent=4)
            logger.info(f"配置已保存到 {config_path}")
            return True
        except Exception as e:
            logger.error(f"保存配置失败: {e}")
            logger.error(traceback.format_exc())
            return False

    def save_trade(self, signal_id, is_buy=True, override_check=False, price=None):
        """保存交易记录到trade_history表

        Args:
            signal_id: 交易信号ID
            is_buy: 是否为买入操作，True为买入，False为卖出（更新）
            override_check: 是否忽略状态检查
            price: 可选的价格参数，如果提供则使用此价格

        Returns:
            bool: 是否成功保存
        """
        try:
            if not self.reconnect_if_needed():
                logger.error("数据库连接失败，无法保存交易记录")
                return False

            # 从trading_signals表获取交易信号数据
            query = """
            SELECT id, stock_code, stock_name, buy_price, sell_price, buy_quantity, 
                   buy_time, sell_time, target_position, stop_loss_price, take_profit_price,
                   is_bought, is_sold
            FROM trading_signals
            WHERE id = %s
            """
            self.cursor.execute(query, (signal_id,))
            signal = self.cursor.fetchone()

            if not signal:
                logger.warning(f"未找到ID为{signal_id}的交易信号")
                return False

            stock_code = signal['stock_code']
            stock_name = signal['stock_name']

            # 如果是买入操作
            if is_buy:
                if not signal['is_bought'] and not override_check:
                    logger.warning(f"信号 #{signal_id} 标记为未买入，无法记录买入交易")
                    return False

                buy_price = price if price is not None else signal['buy_price']
                buy_time = signal['buy_time'] or datetime.now()
                quantity = signal['buy_quantity']

                if not buy_price or not quantity:
                    logger.warning(f"信号 #{signal_id} 缺少买入价格或数量")
                    return False

                # 计算买入金额和费用
                buy_amount = buy_price * quantity
                fee = buy_amount * self.config['trade_settings']['trading_fee_rate']
                total_cost = buy_amount + fee

                # 更新可用资金
                buy_capital = self.update_available_capital(buy_amount, is_buy=True)

                # 检查trade_history表中是否已有该信号的记录
                self.cursor.execute("SELECT id FROM trade_history WHERE trading_signal_id = %s", (signal_id,))
                existing = self.cursor.fetchone()

                if existing:
                    logger.info(f"信号 #{signal_id} 的交易记录已存在，更新买入信息")
                    update_query = """
                    UPDATE trade_history SET
                        buy_price = %s,
                        buy_time = %s,
                        quantity = %s,
                        buy_capital = %s,
                        updated_at = NOW()
                    WHERE trading_signal_id = %s
                    """
                    self.cursor.execute(update_query, (
                        buy_price, buy_time, quantity, buy_capital, signal_id
                    ))
                else:
                    # 插入新记录
                    insert_query = """
                    INSERT INTO trade_history (
                        stock_code, stock_name, buy_price, quantity, 
                        buy_time, buy_capital, actual_position, trading_signal_id
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """

                    target_position = signal['target_position'] or 0.1

                    self.cursor.execute(insert_query, (
                        stock_code, stock_name, buy_price, quantity,
                        buy_time, buy_capital, target_position, signal_id
                    ))

                self.conn.commit()
                logger.info(f"成功保存买入交易: {stock_name}({stock_code}), 价格: {buy_price}, 数量: {quantity}, 剩余资金: {buy_capital}")
                return True

            # 如果是卖出操作
            else:
                # 如果设置了override_check，跳过is_sold检查
                if not signal['is_sold'] and not override_check:
                    logger.warning(f"信号 #{signal_id} 标记为未卖出，无法记录卖出交易")
                    return False

                # 检查trade_history表中是否有该信号的记录
                self.cursor.execute("SELECT id, buy_price, quantity FROM trade_history WHERE trading_signal_id = %s", (signal_id,))
                trade_record = self.cursor.fetchone()

                if not trade_record:
                    logger.warning(f"未找到信号 #{signal_id} 的买入记录，无法更新卖出信息")
                    return False

                # 使用传入的价格或从信号中获取价格
                sell_price = price if price is not None else signal['sell_price']
                sell_time = signal['sell_time'] or datetime.now()

                if not sell_price:
                    logger.warning(f"信号 #{signal_id} 缺少卖出价格，无法记录交易")
                    return False

                # 从记录中获取买入价格和数量
                buy_price = trade_record['buy_price']
                quantity = trade_record['quantity']

                # 确保数值类型一致 - 将所有值转换为float类型
                sell_price = float(sell_price)
                buy_price = float(buy_price)
                quantity = float(quantity)

                # 计算交易结果
                trade_date = sell_time.date() if isinstance(sell_time, datetime) else datetime.now().date()
                sell_amount = sell_price * quantity
                profit_amount = (sell_price - buy_price) * quantity
                profit_rate = ((sell_price - buy_price) / buy_price) * 100 if buy_price > 0 else 0

                # 更新可用资金（卖出）
                sell_capital = self.update_available_capital(sell_amount, is_buy=False)

                # 判断是否盈利
                is_win = profit_amount > 0

                # 更新交易记录
                update_query = """
                UPDATE trade_history SET
                    sell_price = %s,
                    sell_time = %s,
                    profit_amount = %s,
                    profit_rate = %s,
                    sell_capital = %s,
                    is_win = %s,
                    trade_date = %s
                WHERE id = %s
                """

                self.cursor.execute(update_query, (
                    sell_price, sell_time, profit_amount, profit_rate,
                    sell_capital, is_win, trade_date, trade_record['id']
                ))

                self.conn.commit()
                logger.info(f"成功更新卖出交易: {stock_name}({stock_code}), 价格: {sell_price}, "
                            f"盈亏: {profit_amount:.2f}元 ({profit_rate:.2f}%), 剩余资金: {sell_capital:.2f}元")
                return True

        except Exception as e:
            logger.error(f"保存交易记录失败: {e}")
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
            if not self.reconnect_if_needed():
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
            if not self.reconnect_if_needed():
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

            logger.info(f"成功更新卖出信号 #{signal_id}: 价格: {actual_sell_price}")
            return True

        except Exception as e:
            logger.error(f"更新卖出信号失败: {e}")
            logger.error(traceback.format_exc())
            return False

    def sync_trade_history_from_logs(self):
        """从trade_execution_logs表中同步卖出交易记录到trade_history表

        查找trade_execution_logs中所有成功的卖出交易记录，并用这些信息更新trade_history表中缺失的卖出信息

        Returns:
            int: 更新的记录数量
        """
        try:
            # 首先获取trade_execution_logs中所有成功的记录
            sell_logs_query = """
            SELECT * FROM trade_execution_logs 
            WHERE success = 1 
            ORDER BY trade_time DESC
            """

            self.cursor.execute(sell_logs_query)
            all_logs = self.cursor.fetchall()

            if not all_logs:
                logger.info("没有找到任何交易记录")
                return 0

            logger.info(f"找到 {len(all_logs)} 条交易记录，开始筛选卖出记录")

            # 获取所有trade_history记录，用于后续匹配
            self.cursor.execute("SELECT * FROM trade_history")
            all_history = {record['trading_signal_id']: record for record in self.cursor.fetchall()}

            logger.info(f"trade_history表中共有 {len(all_history)} 条记录")

            updated_count = 0

            for log in all_logs:
                try:
                    # 解析trade_info
                    trade_info = json.loads(log['trade_info'])

                    # 只处理卖出记录
                    if trade_info.get('signal_type') != 'sell':
                        continue

                    signal_id = log['signal_id']
                    logger.info(f"处理卖出记录 - 信号ID: {signal_id}")

                    # 跳过已经处理过的信号
                    if signal_id not in all_history:
                        logger.warning(f"信号ID {signal_id} 在trade_history表中不存在，跳过")
                        continue

                    history_record = all_history[signal_id]
                    trade_history_id = history_record['id']

                    # 获取卖出价格
                    sell_price = float(trade_info['price'])
                    logger.info(f"信号ID {signal_id} 的卖出价格为 {sell_price}")

                    # 获取其他必要信息
                    buy_price = float(history_record['buy_price'])
                    quantity = float(history_record['quantity'])
                    sell_time = log['trade_time']

                    # 计算交易结果
                    sell_amount = sell_price * quantity
                    profit_amount = (sell_price - buy_price) * quantity
                    profit_rate = ((sell_price - buy_price) / buy_price) * 100 if buy_price > 0 else 0
                    is_win = profit_amount > 0
                    trade_date = sell_time.date() if isinstance(sell_time, datetime) else datetime.strptime(sell_time, '%Y-%m-%d %H:%M:%S').date()

                    logger.info(f"计算结果 - 卖出金额: {sell_amount:.2f}, 盈亏: {profit_amount:.2f}元, 盈亏率: {profit_rate:.2f}%, 盈利: {is_win}")

                    # 更新可用资金
                    sell_capital = self.update_available_capital(sell_amount, is_buy=False)

                    # 更新trade_history记录
                    update_query = """
                    UPDATE trade_history SET
                        sell_price = %s,
                        sell_time = %s,
                        profit_amount = %s,
                        profit_rate = %s,
                        sell_capital = %s,
                        is_win = %s,
                        trade_date = %s
                    WHERE id = %s
                    """

                    self.cursor.execute(update_query, (
                        sell_price, sell_time, profit_amount, profit_rate,
                        sell_capital, is_win, trade_date, trade_history_id
                    ))

                    updated_count += 1
                    logger.info(f"已更新trade_history记录 #{trade_history_id}: 卖出价格 {sell_price}, 盈亏 {profit_amount:.2f}元 ({profit_rate:.2f}%)")

                except Exception as e:
                    logger.error(f"处理记录时出错: {e}")
                    logger.error(f"问题记录: {log}")
                    continue

            self.conn.commit()
            logger.info(f"成功从交易日志同步了 {updated_count} 条卖出交易记录")
            return updated_count

        except Exception as e:
            logger.error(f"同步交易记录失败: {e}")
            logger.error(traceback.format_exc())
            return 0


def main():
    """主函数"""
    print("初始化凯利公式仓位管理器...")

    config_file = "kelly_config.json"

    # 如果配置文件不存在，创建示例配置
    if not os.path.exists(config_file):
        print(f"配置文件 {config_file} 不存在，创建示例配置...")

        example_config = {
            "mysql_config": {
                "host": "localhost",
                "port": 3306,
                "user": "root",
                "password": "zyb123456668866",
                "database": "stock_analysis"
            },
            "kelly_config": {
                "default_win_rate": 0.55,
                "max_position_ratio": 0.3,
                "min_position_ratio": 0.05,
                "half_kelly": True,
                "stop_loss_ratio": 0.05,
                "take_profit_ratio": 0.1,
                "max_kelly_score": 0.5,
                "win_loss_ratio": 1.5
            },
            "trade_settings": {
                "total_capital": 100000,
                "available_capital": 80000,
                "max_stocks": 5,
                "min_score_to_buy": 80,
                "process_interval": 1,
                "trading_fee_rate": 0.0005
            }
        }

        try:
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(example_config, f, ensure_ascii=False, indent=4)
            print(f"已创建示例配置文件: {config_file}")
            print("请修改配置文件中的参数，特别是数据库连接信息，然后重新运行程序")
            return
        except Exception as e:
            print(f"创建示例配置文件失败: {e}")

    # 初始化管理器
    manager = KellyPositionManager(config_file)

    # 更新表结构
    manager.update_trading_signals_table()

    # 从交易日志中同步卖出交易记录到trade_history表
    print("正在从交易日志同步卖出交易记录...")
    updated_count = manager.sync_trade_history_from_logs()
    print(f"已同步 {updated_count} 条卖出交易记录")

    # 设置检查间隔
    interval = manager.config['trade_settings'].get('process_interval', 1)

    try:
        # 启动管理器
        print(f"启动仓位管理器，处理间隔: {interval} 秒")
        manager.run(interval=interval)
    except Exception as e:
        print(f"运行出错: {e}")
        traceback.print_exc()
    finally:
        print("程序已退出")


if __name__ == "__main__":
    main()