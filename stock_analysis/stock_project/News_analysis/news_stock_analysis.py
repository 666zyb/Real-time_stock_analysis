import json
import redis
import asyncio
import logging
import os
import time
import hashlib
import httpx
from typing import List, Dict, Tuple, Set
from datetime import datetime, timedelta
import pandas as pd

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('news_stock_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# 配置
class Config:
    def __init__(self, config_path='config/config.json'):
        try:
            with open('config/config.json', 'r', encoding='utf-8') as f:
                config = json.load(f)

            # Redis配置
            redis_config = config.get('redis_config', {})
            self.redis_host = redis_config.get('host', '172.16.0.4')
            self.redis_port = redis_config.get('port', 6379)
            self.redis_db = redis_config.get('db', 0)
            self.redis_password = redis_config.get('password', None)

            # AI接口配置 - 保存整个配置字典
            self.ai_config = config.get('ai_config', {})
            self.ai_api_key = self.ai_config.get('api_key', '')
            self.ai_api_url = self.ai_config.get('api_url', 'https://api.openai.com/v1/chat/completions')
            self.ai_model = self.ai_config.get('model', 'gpt-3.5-turbo')

            # 股票信息
            self.stocks = config.get('stocks', [])

            # 新闻和分析配置
            self.hot_news_key = "stock:hot_news"
            self.analyzed_news_key = "stock:analyzed_news_hashes"  # 存储已分析新闻的哈希值
            self.analysis_result_key = "stock:news_analysis_result"
            self.news_days = config.get('news_days', 3)  # 分析最近几天的新闻
            self.batch_size = config.get('batch_size', 10)  # 每批处理的新闻数量

        except Exception as e:
            logger.error(f"加载配置文件出错: {str(e)}")
            raise


class NewsStockAnalyzer:
    def __init__(self, config_path='config/config.json'):
        """初始化分析器"""
        self.config = Config(config_path)

        # 连接Redis
        self.redis_client = redis.Redis(
            host=self.config.redis_host,
            port=self.config.redis_port,
            db=self.config.redis_db,
            password=self.config.redis_password,
            decode_responses=True
        )

        # 初始化异步HTTP客户端
        self.http_client = None

        # 股票代码到名称的映射
        self.stock_code_to_name = {
            stock["code"]: stock["name"] for stock in self.config.stocks
        }

    async def initialize(self):
        """初始化异步资源"""
        self.http_client = httpx.AsyncClient(timeout=30.0)

    async def close(self):
        """关闭异步资源"""
        if self.http_client:
            await self.http_client.aclose()

    def generate_news_hash(self, news: Dict) -> str:
        """为新闻生成唯一哈希值"""
        # 使用新闻内容和日期时间生成哈希
        content = news.get('content', '')
        datetime_str = news.get('datetime', '')
        hash_str = f"{content}|{datetime_str}"
        return hashlib.md5(hash_str.encode('utf-8')).hexdigest()

    def get_analyzed_news_hashes(self) -> Set[str]:
        """获取已分析新闻的哈希值集合"""
        try:
            # 使用Redis集合存储已分析的新闻哈希
            hash_list = self.redis_client.smembers(self.config.analyzed_news_key)
            return set(hash_list)
        except Exception as e:
            logger.error(f"获取已分析新闻哈希时出错: {str(e)}")
            return set()

    def mark_news_as_analyzed(self, news_hashes: List[str]):
        """标记新闻为已分析"""
        try:
            if news_hashes:
                # 将哈希值添加到Redis集合
                self.redis_client.sadd(self.config.analyzed_news_key, *news_hashes)
                logger.info(f"标记 {len(news_hashes)} 条新闻为已分析")
        except Exception as e:
            logger.error(f"标记新闻为已分析时出错: {str(e)}")

    async def get_news_from_redis(self, limit=30) -> Tuple[List[Dict], List[str]]:
        """从Redis获取最近的指定数量新闻，返回未分析的新闻列表和对应的哈希值"""
        try:
            # 获取所有热点新闻
            news_list = self.redis_client.lrange(self.config.hot_news_key, 0, -1)

            # 获取已分析的新闻哈希集合
            analyzed_hashes = self.get_analyzed_news_hashes()
            logger.info(f"已有 {len(analyzed_hashes)} 条新闻被分析过")

            # 解析所有新闻并按时间排序
            news_with_time = []

            for news_item in news_list:
                try:
                    news_data = json.loads(news_item)
                    news_hash = self.generate_news_hash(news_data)
                    news_date = datetime.strptime(news_data.get('datetime', ''), '%Y-%m-%d %H:%M:%S')
                    news_with_time.append((news_data, news_hash, news_date))
                except Exception as e:
                    logger.warning(f"解析新闻数据出错: {str(e)}")

            # 按时间倒序排序，最新的新闻排在前面
            news_with_time.sort(key=lambda x: x[2], reverse=True)

            # 只保留最新的limit条新闻
            recent_news_with_time = news_with_time[:limit]

            # 从最新的limit条中过滤出未分析的新闻
            filtered_news = []
            filtered_hashes = []

            for news, news_hash, news_date in recent_news_with_time:
                if news_hash not in analyzed_hashes:
                    filtered_news.append(news)
                    filtered_hashes.append(news_hash)

            logger.info(f"获取最新的 {limit} 条新闻，其中 {len(filtered_news)} 条未分析")
            return filtered_news, filtered_hashes

        except Exception as e:
            logger.error(f"从Redis获取新闻时出错: {str(e)}")
            return [], []

    async def analyze_news_with_ai(self, news: Dict) -> Dict:
        """使用AI分析单条新闻，增强错误处理"""
        try:
            # 准备新闻内容
            news_content = f"{news.get('content', '')} ({news.get('datetime', '')})"

            # 构建提示词
            prompt = f"""
            请分析以下中国A股市场的最新财经新闻。
            从专业金融分析师的角度，找出这条新闻短期内可能对哪些股票产生最大影响。

            请直接分析新闻中可能影响的股票，找出：
            1. 两只最可能受到正面影响的股票（代码和名称）及上涨理由（简短）
            2. 两只最可能受到负面影响的股票（代码和名称）及下跌理由（简短）

            新闻:
            {news_content}

            输出需要简洁易于在网页展示，请使用以下JSON格式返回结果：
            {{
                "analysis": "对该新闻的简要分析（30字以内）",
                "potential_risers": [
                    {{"code": "股票代码1", "name": "股票名称1", "reason": "上涨理由（15字以内）", "influence": "影响程度(强/中/弱)"}},
                    {{"code": "股票代码2", "name": "股票名称2", "reason": "上涨理由（15字以内）", "influence": "影响程度(强/中/弱)"}}
                ],
                "potential_fallers": [
                    {{"code": "股票代码1", "name": "股票名称1", "reason": "下跌理由（15字以内）", "influence": "影响程度(强/中/弱)"}},
                    {{"code": "股票代码2", "name": "股票名称2", "reason": "下跌理由（15字以内）", "influence": "影响程度(强/中/弱)"}}
                ]
            }}
            """

            # 设置请求超时时间更长
            timeout = httpx.Timeout(60.0, connect=15.0)  # 增加超时时间到60秒

            # 处理DeepSeek API
            if self.config.ai_config.get('provider') == 'deepseek':
                api_key = self.config.ai_config.get('api_key')
                api_base = self.config.ai_config.get('api_base', 'https://api.deepseek.com')
                api_version = self.config.ai_config.get('api_version', 'v1')
                model = self.config.ai_config.get('model', 'deepseek-chat')

                # 构建API URL
                api_url = f"{api_base}/{api_version}/chat/completions"

                # 记录开始分析的时间
                start_time = time.time()
                logger.info(f"开始分析新闻: {news_content[:30]}...")

                # 发送请求，增加超时设置
                response = await self.http_client.post(
                    api_url,
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {api_key}"
                    },
                    json={
                        "model": model,
                        "messages": [
                            {"role": "system", "content": "你是一位专业的A股金融分析师，擅长从新闻中识别可能对股票价格产生影响的信息。你的回答简洁明了，适合在网页上展示。"},
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": self.config.ai_config.get('temperature', 0.2),
                        "response_format": {"type": "json_object"}  # 指定返回JSON格式
                    },
                    timeout=timeout
                )

                # 记录分析完成的时间和耗时
                end_time = time.time()
                logger.info(f"分析完成，耗时: {end_time - start_time:.2f}秒")

                if response.status_code == 200:
                    result = response.json()
                    ai_response = result['choices'][0]['message']['content']

                    # 提取JSON部分
                    try:
                        # 尝试直接解析整个响应
                        analysis_result = json.loads(ai_response)
                        return analysis_result
                    except json.JSONDecodeError:
                        # 如果失败，尝试提取JSON部分
                        import re
                        json_pattern = r'```json\s*(.*?)\s*```|{.*}'
                        match = re.search(json_pattern, ai_response, re.DOTALL)
                        if match:
                            json_str = match.group(1) if match.group(1) else match.group(0)
                            return json.loads(json_str)
                        else:
                            logger.error(f"无法从AI响应中提取JSON: {ai_response}")
                            return {}
                else:
                    logger.error(f"DeepSeek API请求失败 (状态码: {response.status_code}): {response.text}")
                    return {}

            # 处理其他API提供商的逻辑
            else:
                # 记录开始分析的时间
                start_time = time.time()
                logger.info(f"开始分析新闻: {news_content[:30]}...")

                # 原有的API调用逻辑（适配您当前的API调用方式）
                response = await self.http_client.post(
                    self.config.ai_api_url,
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.config.ai_api_key}"
                    },
                    json={
                        "model": self.config.ai_model,
                        "messages": [
                            {"role": "system", "content": "你是一位专业的A股金融分析师，擅长从新闻中识别可能对股票价格产生影响的信息。你的回答简洁明了，适合在网页上展示。"},
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": 0.3
                    },
                    timeout=timeout
                )

                # 记录分析完成的时间和耗时
                end_time = time.time()
                logger.info(f"分析完成，耗时: {end_time - start_time:.2f}秒")

                # 处理响应
                if response.status_code == 200:
                    result = response.json()
                    ai_response = result['choices'][0]['message']['content']

                    try:
                        # 尝试直接解析整个响应
                        analysis_result = json.loads(ai_response)
                        return analysis_result
                    except json.JSONDecodeError:
                        # 如果失败，尝试提取JSON部分
                        import re
                        json_pattern = r'```json\s*(.*?)\s*```|{.*}'
                        match = re.search(json_pattern, ai_response, re.DOTALL)
                        if match:
                            json_str = match.group(1) if match.group(1) else match.group(0)
                            return json.loads(json_str)
                        else:
                            logger.error(f"无法从AI响应中提取JSON: {ai_response}")
                            return {}
                else:
                    logger.error(f"AI请求失败 (状态码: {response.status_code}): {response.text}")
                    return {}

        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            logger.error(f"分析新闻时出错: {str(e)}\n{error_details}")
            return {}

    async def process_news_batches(self, news_list: List[Dict], news_hashes: List[str], max_concurrent=3) -> List[str]:
        """处理每条新闻，使用串行处理避免API超时"""
        processed_hashes = []
        retry_delay = 5  # 重试延迟秒数
        max_retries = 2  # 最大重试次数

        logger.info(f"开始逐条分析 {len(news_list)} 条新闻")

        # 按时间反序排序新闻列表，最新的新闻优先处理
        news_with_hash = list(zip(news_list, news_hashes))

        # 逐条处理新闻，不使用并行处理
        for idx, (news, news_hash) in enumerate(news_with_hash):
            logger.info(f"处理新闻 {idx + 1}/{len(news_list)}")

            # 分析单条新闻
            result = await self.analyze_single_news_with_retry(news, max_retries, retry_delay)

            # 处理分析结果
            if result:
                # 记录成功分析的新闻哈希
                processed_hashes.append(news_hash)

                # 将结果保存到Redis
                analysis_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                result['timestamp'] = analysis_timestamp

                # 使用Redis哈希表存储所有新闻分析结果
                self.redis_client.hset(
                    "stock:news_all_analyses",
                    news_hash,
                    json.dumps(result, ensure_ascii=False)
                )

                logger.info(f"新闻 {idx + 1}/{len(news_list)} 分析完成")

                # 每完成一条新闻就立即标记为已分析并更新分析汇总
                self.mark_news_as_analyzed([news_hash])

                # 更新分析汇总，以提供最新结果
                self._update_analysis_with_latest_news([news_hash])
            else:
                logger.warning(f"新闻 {idx + 1}/{len(news_list)} 分析结果为空")

            # 每条新闻处理完后添加短暂延迟，避免过度请求
            await asyncio.sleep(2)

        return processed_hashes

    async def analyze_single_news_with_retry(self, news: Dict, max_retries: int, retry_delay: float) -> Dict:
        """使用重试机制分析单条新闻"""
        retries = 0
        while retries <= max_retries:
            try:
                result = await self.analyze_news_with_ai(news)
                if result:
                    return result

                logger.warning(f"分析结果为空，重试 {retries + 1}/{max_retries}")
                retries += 1
                if retries <= max_retries:
                    await asyncio.sleep(retry_delay)
            except Exception as e:
                import traceback
                error_details = traceback.format_exc()
                logger.error(f"分析新闻时出错: {str(e)}\n{error_details}")
                retries += 1
                if retries <= max_retries:
                    logger.info(f"等待 {retry_delay} 秒后重试...")
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error(f"达到最大重试次数，放弃分析此新闻")
                    return None

        return None

    def combine_analysis_results(self, results: List[Dict]) -> Dict:
        """合并多个分析结果"""
        if not results:
            return {
                "analysis": "无有效分析结果",
                "potential_risers": [],
                "potential_fallers": []
            }

        # 收集所有上涨和下跌股票
        all_risers = []
        all_fallers = []

        for result in results:
            risers = result.get('potential_risers', [])
            fallers = result.get('potential_fallers', [])

            all_risers.extend(risers)
            all_fallers.extend(fallers)

        # 统计每支股票被提名的次数和影响程度
        riser_scores = self._calculate_stock_scores(all_risers)
        faller_scores = self._calculate_stock_scores(all_fallers)

        # 获取得分最高的两支上涨股票和两支下跌股票
        top_risers = sorted(riser_scores.items(), key=lambda x: x[1]['score'], reverse=True)[:2]
        top_fallers = sorted(faller_scores.items(), key=lambda x: x[1]['score'], reverse=True)[:2]

        # 初始化最终结果列表
        final_risers = []
        final_fallers = []

        # 综合分析
        analysis_summary = "基于最新新闻的分析，"
        if top_risers:
            # 先构建股票名称列表
            riser_stocks = [f"{info['name']}({code})" for code, info in dict(top_risers).items()]
            riser_str = ', '.join(riser_stocks)
            analysis_summary += f"预计短期内 {riser_str} 可能会有较好表现；"

            if top_fallers:
                faller_stocks = [f"{info['name']}({code})" for code, info in dict(top_fallers).items()]
                faller_str = ', '.join(faller_stocks)
                analysis_summary += f"而 {faller_str} 可能面临下行压力。"

            # 转换为原始格式
            final_risers = [
                {
                    "code": code,
                    "name": info['name'],
                    "reason": info['reason'],
                    "influence": info['influence']
                }
                for code, info in dict(top_risers).items()
            ]

            final_fallers = [
                {
                    "code": code,
                    "name": info['name'],
                    "reason": info['reason'],
                    "influence": info['influence']
                }
                for code, info in dict(top_fallers).items()
            ]

        return {
            "analysis": analysis_summary,
            "potential_risers": final_risers,
            "potential_fallers": final_fallers,
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "analyzed_days": self.config.news_days,
            "is_new_analysis": True  # 标记这是新的分析结果
        }

    def _calculate_stock_scores(self, stocks_list: List[Dict]) -> Dict[str, Dict]:
        """计算每支股票的得分"""
        stock_scores = {}

        for stock in stocks_list:
            code = stock.get('code')
            name = stock.get('name')
            reason = stock.get('reason', '')
            influence = stock.get('influence', '中')

            if not code or not name:
                continue

            # 根据影响程度赋予分数
            influence_score = {
                '强': 3.0,
                '中': 2.0,
                '弱': 1.0
            }.get(influence, 2.0)

            if code in stock_scores:
                current_score = stock_scores[code]['score']
                # 新的原因比旧的更强或者相同，用新的替换
                if influence_score >= current_score:
                    stock_scores[code] = {
                        'name': name,
                        'reason': reason,
                        'influence': influence,
                        'score': current_score + influence_score * 0.5  # 增加总分但不是简单相加
                    }
                else:
                    # 增加总分但保留旧理由
                    stock_scores[code]['score'] += influence_score * 0.5
            else:
                stock_scores[code] = {
                    'name': name,
                    'reason': reason,
                    'influence': influence,
                    'score': influence_score
                }

        return stock_scores

    async def run_analysis(self, max_news=5, parallel_count=1):
        """运行完整的新闻分析流程，只处理最新的max_news条新闻中未分析的部分"""
        try:
            # 初始化异步客户端
            await self.initialize()

            # 获取最新的max_news条新闻中未分析的部分
            # 默认限制为5条，避免处理过多导致超时
            news_list, news_hashes = await self.get_news_from_redis(limit=max_news)

            if not news_list:
                logger.info(f"最近{max_news}条新闻都已分析过")
                # 如果没有新的新闻，获取最近的分析结果
                latest_analysis = self.redis_client.get("stock:news_analysis_summary")
                if latest_analysis:
                    latest_result = json.loads(latest_analysis)
                    latest_result["is_new_analysis"] = False
                    logger.info("返回最近的分析结果")
                    return latest_result
                return {
                    "analysis": f"最近{max_news}条新闻都已分析过，无需更新分析结果",
                    "potential_risers": [],
                    "potential_fallers": [],
                    "is_new_analysis": False
                }

            logger.info(f"开始处理最新的 {len(news_list)} 条未分析新闻")

            # 处理新闻，使用串行处理
            processed_hashes = await self.process_news_batches(news_list, news_hashes, max_concurrent=1)

            # 获取分析结果
            analysis_results = []

            # 处理所有分析过的新闻结果
            for news_hash in processed_hashes:
                result = self.get_news_analysis(news_hash)
                if result:
                    analysis_results.append(result)

            if not analysis_results:
                logger.warning("没有有效的分析结果")
                return {
                    "analysis": "分析未能产生有效结果",
                    "potential_risers": [],
                    "potential_fallers": [],
                    "is_new_analysis": False
                }

            # 合并分析结果
            final_result = self.combine_analysis_results(analysis_results)

            # 输出最终分析结果
            self.redis_client.set(
                "stock:news_analysis_summary",
                json.dumps(final_result, ensure_ascii=False)
            )

            logger.info(f"分析完成，最终结果: {json.dumps(final_result, ensure_ascii=False, indent=2)}")
            return final_result

        except Exception as e:
            logger.error(f"运行分析时出错: {str(e)}")
            return {"error": str(e)}
        finally:
            # 关闭异步客户端
            await self.close()

    def get_news_analysis(self, news_hash: str) -> Dict:
        """获取单条新闻的分析结果"""
        try:
            analysis_json = self.redis_client.hget("stock:news_all_analyses", news_hash)
            if analysis_json:
                return json.loads(analysis_json)
            return None
        except Exception as e:
            logger.error(f"获取新闻分析结果时出错: {str(e)}")
            return None

    def _update_analysis_with_latest_news(self, news_hashes: List[str]):
        """使用最新分析的新闻更新分析汇总"""
        if not news_hashes:
            return

        try:
            # 获取这些新闻的分析结果
            latest_results = []
            for news_hash in news_hashes:
                result = self.get_news_analysis(news_hash)
                if result:
                    latest_results.append(result)

            if not latest_results:
                return

            # 合并结果
            combined_result = self.combine_analysis_results(latest_results)
            combined_result['is_new_analysis'] = True
            combined_result['priority_update'] = True  # 标记为优先更新

            # 保存到Redis
            self.redis_client.set(
                "stock:news_latest_analysis",
                json.dumps(combined_result, ensure_ascii=False)
            )

            # 同时更新总体分析
            self.redis_client.set(
                "stock:news_analysis_summary",
                json.dumps(combined_result, ensure_ascii=False)
            )

            logger.info(f"已使用 {len(latest_results)} 条最新新闻更新分析汇总")
        except Exception as e:
            logger.error(f"更新分析汇总时出错: {str(e)}")


async def main():
    import argparse

    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='分析新闻对股票影响')
    parser.add_argument('--max-news', type=int, default=30, help='最多分析的新闻数量')
    parser.add_argument('--parallel', type=int, default=3, help='并行分析的新闻数量')

    args = parser.parse_args()

    logger.info(f"启动分析: 最大新闻数 {args.max_news}, 并行数量 {args.parallel}")

    analyzer = NewsStockAnalyzer()
    result = await analyzer.run_analysis(max_news=args.max_news, parallel_count=args.parallel)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())