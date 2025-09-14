import argparse
import requests
import json
import os
from time import sleep
from requests.auth import HTTPBasicAuth
from typing import List, Dict
import time
import re
import logging
from queue import Queue
from threading import Thread
from concurrent.futures import ThreadPoolExecutor, as_completed
import random

# 配置日志
logger = logging.getLogger(__name__)


class RetryQueue:
    def __init__(self, generator, max_retries=3, retry_delay=60):
        self.queue = Queue()
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.generator = generator  # 存储生成器的引用
        self.worker = Thread(target=self._process_queue, daemon=True)
        self.worker.start()

    def add(self, alpha: str, retry_count: int = 0):
        self.queue.put((alpha, retry_count))

    def _process_queue(self):
        while True:
            if not self.queue.empty():
                alpha, retry_count = self.queue.get()
                if retry_count >= self.max_retries:
                    logging.error(f"alpha 重试次数已达上限: {alpha}")
                    continue

                try:
                    result = self.generator._test_alpha_impl(
                        alpha
                    )  # 使用 _test_alpha_impl 避免递归
                    if result.get(
                        "status"
                    ) == "error" and "SIMULATION_LIMIT_EXCEEDED" in result.get(
                        "message", ""
                    ):
                        logging.info(
                            f"模拟限制已达上限，重新排队 alpha: {alpha}"
                        )
                        time.sleep(self.retry_delay)
                        self.add(alpha, retry_count + 1)
                    else:
                        self.generator.results.append(
                            {"alpha": alpha, "result": result}
                        )
                except Exception as e:
                    logging.error(f"处理 alpha 时出错: {str(e)}")

            time.sleep(1)  # 防止忙等待


class AlphaGenerator:
    def __init__(
        self,
        credentials_path: str,
        ollama_url: str = "http://localhost:11434",
        max_concurrent: int = 2,
    ):
        self.sess = requests.Session()
        self.credentials_path = credentials_path  # 存储路径用于重新认证
        self.setup_auth(credentials_path)
        self.ollama_url = ollama_url
        self.results = []
        self.pending_results = {}
        self.retry_queue = RetryQueue(self)
        # 减少并发工作线程以防止 VRAM 问题
        self.executor = ThreadPoolExecutor(
            max_workers=max_concurrent
        )  # 用于并发模拟
        self.vram_cleanup_interval = 10  # 每 10 次操作清理一次
        self.operation_count = 0

        # 模型降级跟踪
        self.initial_model = getattr(self, "model_name", "llama3.2:3b")
        self.error_count = 0
        self.max_errors_before_downgrade = 3
        self.model_fleet = [
            "llama3.2:3b",  # 首选稳定模型
            "deepseek-r1:8b",  # 更大的推理模型
            "deepseek-r1:1.5b",  # 较小的推理模型
            "phi3:mini",  # 紧急备用模型
        ]
        self.current_model_index = 0

    def setup_auth(self, credentials_path: str) -> None:
        """设置与 WorldQuant Brain 的认证"""
        logging.info(f"从 {credentials_path} 加载凭证")
        with open(credentials_path) as f:
            credentials = json.load(f)

        username, password = credentials
        self.sess.auth = HTTPBasicAuth(username, password)

        logging.info("正在与 WorldQuant Brain 进行认证...")
        response = self.sess.post("https://api.worldquantbrain.com/authentication")
        logging.info(f"认证响应状态: {response.status_code}")
        logging.debug(f"认证响应内容: {response.text[:500]}...")

        if response.status_code != 201:
            raise Exception(f"认证失败: {response.text}")

    def cleanup_vram(self):
        """通过强制垃圾回收和等待来清理 VRAM"""
        try:
            import gc

            gc.collect()
            logging.info("已执行 VRAM 清理")
            # 添加小延迟以释放 GPU 内存
            time.sleep(2)
        except Exception as e:
            logging.warning(f"VRAM 清理失败: {e}")

    def get_data_fields(self) -> List[Dict]:
        """从 WorldQuant Brain 获取多个数据集中的可用数据字段，通过随机采样。
        限制为稳定数据集以提高表达式的鲁棒性。
        """
        datasets = ["fundamental6", "fundamental2"]
        all_fields = []

        base_params = {
            "delay": 1,
            "instrumentType": "EQUITY",
            "limit": 20,
            "region": "USA",
            "universe": "TOP3000",
        }

        try:
            print("正在从多个数据集中请求数据字段...")
            for dataset in datasets:
                # 首先获取计数
                params = base_params.copy()
                params["dataset.id"] = dataset
                params["limit"] = 1  # 仅用于高效获取计数

                print(f"获取数据集 {dataset} 的字段计数")
                count_response = self.sess.get(
                    "https://api.worldquantbrain.com/data-fields", params=params
                )

                if count_response.status_code == 200:
                    count_data = count_response.json()
                    total_fields = count_data.get("count", 0)
                    print(f"数据集 {dataset} 中的总字段数: {total_fields}")

                    if total_fields > 0:
                        # 生成随机偏移
                        max_offset = max(0, total_fields - base_params["limit"])
                        random_offset = random.randint(0, max_offset)

                        # 获取随机子集
                        params["offset"] = random_offset
                        params["limit"] = min(
                            20, total_fields
                        )  # 不超过总字段数

                        print(
                            f"正在获取数据集 {dataset} 的字段，偏移量为 {random_offset}"
                        )
                        response = self.sess.get(
                            "https://api.worldquantbrain.com/data-fields", params=params
                        )

                        if response.status_code == 200:
                            data = response.json()
                            fields = data.get("results", [])
                            print(f"在数据集 {dataset} 中找到 {len(fields)} 个字段")
                            all_fields.extend(fields)
                        else:
                            print(
                                f"获取数据集 {dataset} 的字段失败: {response.text[:500]}"
                            )
                else:
                    print(
                        f"获取数据集 {dataset} 的计数失败: {count_response.text[:500]}"
                    )

            # 移除重复项（如果有）
            unique_fields = {field["id"]: field for field in all_fields}.values()
            print(f"找到的唯一字段总数: {len(unique_fields)}")
            return list(unique_fields)

        except Exception as e:
            logger.error(f"获取数据字段失败: {e}")
            return []

    def get_operators(self) -> List[Dict]:
        """从 WorldQuant Brain 获取可用的运算符"""
        print("正在请求运算符...")
        response = self.sess.get("https://api.worldquantbrain.com/operators")
        print(f"运算符响应状态: {response.status_code}")
        print(f"运算符响应内容: {response.text[:500]}...")  # 打印前 500 个字符

        if response.status_code != 200:
            raise Exception(f"获取运算符失败: {response.text}")

        data = response.json()
        # 运算符端点可能直接返回数组，而不是包含 'items' 或 'results' 的对象
        if isinstance(data, list):
            return data
        elif "results" in data:
            return data["results"]
        else:
            raise Exception(f"意外的运算符响应格式。响应: {data}")

    def clean_alpha_ideas(self, ideas: List[str]) -> List[str]:
        """清理和验证 alpha 想法，仅保留有效的表达式。
        强制要求：单行 FASTEXPR 表达式，无赋值/多语句，括号平衡，
        白名单运算符和常见转换，避免高风险的逻辑结构。
        """
        cleaned_ideas = []

        for idea in ideas:
            original = idea
            # 跳过仅为数字或单个单词的想法
            if re.match(r"^\d+\.?$|^[a-zA-Z]+$", idea):
                continue

            # 跳过描述性内容（包含常见英文单词）
            common_words = [
                "it",
                "the",
                "is",
                "are",
                "captures",
                "provides",
                "measures",
            ]
            if any(word in idea.lower() for word in common_words):
                continue

            # 硬性拒绝：赋值或多语句/注释
            if ("=" in idea) or (";" in idea) or idea.startswith("Comment:"):
                continue
            # 拒绝高风险的控制/逻辑结构（已知容易检查失败）
            risky_keywords = [
                "if_else",
                "trade_when",
                "bucket",
                "equal(",
                "greater(",
                "less(",
                "not_equal",
                "normalize(",
            ]
            if any(k in idea for k in risky_keywords):
                continue
            # 括号平衡检查
            if idea.count("(") != idea.count(")"):
                continue
            # 验证想法包含有效的运算符/函数（优先选择稳健的时间序列/组操作）
            valid_functions = [
                "ts_mean",
                "ts_sum",
                "ts_rank",
                "ts_std_dev",
                "rank",
                "zscore",
                "log",
                "sqrt",
                "divide",
                "subtract",
                "add",
                "multiply",
                "group_mean",
                "group_neutralize",
                "group_zscore",
                "ts_product",
            ]
            if not any(func in idea for func in valid_functions):
                continue

            # 如果缺少首选规范包装，则强制执行（通过提示软指导；此处仅保留想法）
            cleaned_ideas.append(idea)

        return cleaned_ideas

    def generate_alpha_ideas_with_ollama(
        self, data_fields: List[Dict], operators: List[Dict]
    ) -> List[str]:
        """使用 Ollama 和 FinGPT 模型生成 alpha 想法"""
        print("正在按类别组织运算符...")
        operator_by_category = {}
        for op in operators:
            category = op["category"]
            if category not in operator_by_category:
                operator_by_category[category] = []
            operator_by_category[category].append(
                {
                    "name": op["name"],
                    "type": op.get("type", "SCALAR"),
                    "definition": op["definition"],
                    "description": op["description"],
                }
            )

        try:
            # 如果之前达到令牌限制，清除已测试的表达式
            if hasattr(self, "_hit_token_limit"):
                logger.info("由于之前的令牌限制，清除已测试的表达式")
                self.results = []
                delattr(self, "_hit_token_limit")

            # 从每个类别中随机采样约 35% 的运算符（更紧凑、高精度的集合）
            sampled_operators = {}
            for category, ops in operator_by_category.items():
                sample_size = max(
                    1, int(len(ops) * 0.35)
                )  # 每个类别至少 1 个运算符
                sampled_operators[category] = random.sample(ops, sample_size)

            print("正在为 FinGPT 准备提示...")

            # 格式化运算符及其类型、定义和描述
            def format_operators(ops):
                formatted = []
                for op in ops:
                    formatted.append(
                        f"{op['name']} ({op['type']})\n"
                        f"  定义: {op['definition']}\n"
                        f"  描述: {op['description']}"
                    )
                return formatted

            prompt = f"""生成 5 个独特的、现实的 FASTEXPR alpha 表达式，仅使用提供的运算符和数据字段。仅返回表达式，每行一个，不带注释或解释。

可用数据字段:
{[field['id'] for field in data_fields]}

按类别可用的运算符:
时间序列:
{chr(10).join(format_operators(sampled_operators.get('Time Series', [])))}

横截面:
{chr(10).join(format_operators(sampled_operators.get('Cross Sectional', [])))}

算术:
{chr(10).join(format_operators(sampled_operators.get('Arithmetic', [])))}

逻辑:
{chr(10).join(format_operators(sampled_operators.get('Logical', [])))}

向量:
{chr(10).join(format_operators(sampled_operators.get('Vector', [])))}

转换:
{chr(10).join(format_operators(sampled_operators.get('Transformational', [])))}

组:
{chr(10).join(format_operators(sampled_operators.get('Group', [])))}

质量检查清单（硬性约束）:
1. 仅输出单行 FASTEXPR 表达式：禁止赋值/变量名/多语句/分号/注释。
2. 必须使用以下白名单算子：ts_mean/ts_std_dev/ts_rank/rank/zscore/divide/add/subtract/multiply/group_neutralize/group_mean/group_zscore/ts_product。
3. 强制平滑与中性：优先形如 group_neutralize(zscore(<ts_op>), "sector")。
4. 时间窗限制：仅可使用 {5, 20, 60, 120, 180, 252}。
5. 避免逻辑/条件类算子（if_else/trade_when/bucket/equal/greater/less/normalize），避免过深嵌套（最多3层）。
6. 严禁自造变量名（如 market_ret/rfr），仅使用提供的数据字段与白名单算子。

提示: 
- 可以使用分号分隔表达式。
- 注意运算符类型（SCALAR, VECTOR, MATRIX）以确保兼容性。
- 研究运算符定义和描述以理解其行为。

示例格式:
group_neutralize(zscore(ts_mean(returns, 120)), "sector")
group_neutralize(zscore(ts_mean(returns, 120) - ts_mean(returns, 252)), "sector")
group_neutralize(zscore(rank(divide(revenue, assets))), "sector")
"""

            # 准备 Ollama API 请求
            model_name = getattr(
                self, "model_name", self.model_fleet[self.current_model_index]
            )
            ollama_data = {
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "temperature": 0.2,
                "top_p": 0.8,
                "num_predict": 1000,  # 使用 num_predict 替代 max_tokens（Ollama）
            }

            print("正在向 Ollama API 发送请求...")
            try:
                response = requests.post(
                    f"{self.ollama_url}/api/generate",
                    json=ollama_data,
                    timeout=360,  # 6 minutes timeout
                )

                print(f"Ollama API response status: {response.status_code}")
                print(
                    f"Ollama API response: {response.text[:500]}..."
                )  # Print first 500 chars

                if response.status_code == 500:
                    logging.error(f"Ollama API returned 500 error: {response.text}")
                    # Trigger model downgrade for 500 errors
                    self._handle_ollama_error("500_error")
                    return []
                elif response.status_code != 200:
                    raise Exception(f"Ollama API request failed: {response.text}")

            except requests.exceptions.Timeout:
                logging.error("Ollama API request timed out (360s)")
                # Trigger model downgrade for timeouts
                self._handle_ollama_error("timeout")
                return []
            except requests.exceptions.ConnectionError as e:
                if "Read timed out" in str(e):
                    logging.error("Ollama API read timeout")
                    # Trigger model downgrade for read timeouts
                    self._handle_ollama_error("read_timeout")
                    return []
                else:
                    raise e

            response_data = response.json()
            print(f"Ollama API response JSON keys: {list(response_data.keys())}")

            if "response" not in response_data:
                raise Exception(
                    f"Unexpected Ollama API response format: {response_data}"
                )

            print("Processing Ollama API response...")
            content = response_data["response"]

            # Extract pure alpha expressions by:
            # 1. Remove markdown backticks
            # 2. Remove numbering (e.g., "1. ", "2. ")
            # 3. Skip comments
            alpha_ideas = []
            for line in content.split("\n"):
                line = line.strip()
                if not line or line.startswith("#") or line.startswith("*"):
                    continue
                # Remove numbering and backticks
                line = line.replace("`", "")
                if ". " in line:
                    line = line.split(". ", 1)[1]
                if line and not line.startswith("Comment:"):
                    alpha_ideas.append(line)

            print(f"Generated {len(alpha_ideas)} alpha ideas")
            for i, alpha in enumerate(alpha_ideas, 1):
                print(f"Alpha {i}: {alpha}")

            # Clean and validate ideas
            cleaned_ideas = self.clean_alpha_ideas(alpha_ideas)
            logging.info(f"Found {len(cleaned_ideas)} valid alpha expressions")

            return cleaned_ideas

        except Exception as e:
            if "token limit" in str(e).lower():
                self._hit_token_limit = True  # Mark that we hit token limit
            logging.error(f"Error generating alpha ideas: {str(e)}")
            return []

    def _handle_ollama_error(self, error_type: str):
        """处理 Ollama 错误，必要时降级模型"""
        self.error_count += 1
        logging.warning(
            f"Ollama 错误 ({error_type}) - 计数: {self.error_count}/{self.max_errors_before_downgrade}"
        )

        if self.error_count >= self.max_errors_before_downgrade:
            self._downgrade_model()
            self.error_count = 0  # 降级后重置错误计数

    def _downgrade_model(self):
        """降级到模型队列中的下一个较小模型"""
        if self.current_model_index >= len(self.model_fleet) - 1:
            logging.error("已经在使用模型队列中最小的模型！")
            # 如果已耗尽所有选项，则重置为初始模型
            self.current_model_index = 0
            self.model_name = self.initial_model
            logging.info(f"重置为初始模型: {self.initial_model}")
            return

        old_model = self.model_fleet[self.current_model_index]
        self.current_model_index += 1
        new_model = self.model_fleet[self.current_model_index]

        logging.warning(f"降级模型: {old_model} -> {new_model}")
        self.model_name = new_model

        # 如果存在协调器，则更新其中的模型
        try:
            # 尝试更新协调器的模型队列管理器
            if hasattr(self, "orchestrator") and hasattr(
                self.orchestrator, "model_fleet_manager"
            ):
                self.orchestrator.model_fleet_manager.current_model_index = (
                    self.current_model_index
                )
                self.orchestrator.model_fleet_manager.save_state()
                logging.info(f"更新协调器模型队列以使用: {new_model}")
        except Exception as e:
            logging.warning(f"无法更新协调器模型队列: {e}")

        logging.info(f"成功降级到 {new_model}")

    def test_alpha_batch(self, alphas: List[str]) -> None:
        """提交一批 alpha 进行测试，并监控进度，遵守并发限制"""
        logging.info(f"开始批量测试 {len(alphas)} 个 alpha")
        for alpha in alphas:
            logging.info(f"alpha 表达式: {alpha}")

        # 将 alpha 分成较小的块以遵守并发限制
        max_concurrent = self.executor._max_workers
        submitted = 0
        queued = 0

        for i in range(0, len(alphas), max_concurrent):
            chunk = alphas[i : i + max_concurrent]
            logging.info(
                f"提交块 {i//max_concurrent + 1}/{(len(alphas)-1)//max_concurrent + 1} ({len(chunk)} 个 alpha)"
            )

            # 提交块
            futures = []
            for j, alpha in enumerate(chunk, 1):
                logging.info(f"提交 alpha {i+j}/{len(alphas)}")
                future = self.executor.submit(self._test_alpha_impl, alpha)
                futures.append((alpha, future))

            # 处理此块的结果
            for alpha, future in futures:
                try:
                    result = future.result()
                    if result.get("status") == "error":
                        if "SIMULATION_LIMIT_EXCEEDED" in result.get("message", ""):
                            self.retry_queue.add(alpha)
                            queued += 1
                            logging.info(f"加入重试队列: {alpha}")
                        else:
                            logging.error(
                                f"alpha 模拟错误: {alpha}: {result.get('message')}"
                            )
                        continue

                    sim_id = result.get("result", {}).get("id")
                    progress_url = result.get("result", {}).get("progress_url")
                    if sim_id and progress_url:
                        self.pending_results[sim_id] = {
                            "alpha": alpha,
                            "progress_url": progress_url,
                            "status": "pending",
                            "attempts": 0,
                        }
                        submitted += 1
                        logging.info(f"成功提交 {alpha} (ID: {sim_id})")

                except Exception as e:
                    logging.error(f"提交 alpha {alpha} 时出错: {str(e)}")

            # 在块之间等待以避免压垮 API
            if i + max_concurrent < len(alphas):
                logging.info(f"等待 10 秒后再提交下一个块...")
                sleep(10)

        logging.info(
            f"批量提交完成: {submitted} 个已提交, {queued} 个加入重试队列"
        )

        # 监控进度直到所有完成或需要重试
        total_successful = 0
        max_monitoring_time = 21600  # 6 小时最大监控时间
        start_time = time.time()

        while self.pending_results:
            # 检查超时
            if time.time() - start_time > max_monitoring_time:
                logging.warning(
                    f"监控超时 ({max_monitoring_time} 秒), 停止监控"
                )
                logging.warning(
                    f"剩余的待处理模拟: {list(self.pending_results.keys())}"
                )
                break

            logging.info(
                f"正在监控 {len(self.pending_results)} 个待处理模拟..."
            )
            completed = self.check_pending_results()
            total_successful += completed
            sleep(5)  # Wait between checks

        logging.info(f"Batch complete: {total_successful} successful simulations")
        return total_successful

    def check_pending_results(self) -> int:
        """检查所有待处理模拟的状态，并正确处理重试"""
        completed = []
        retry_queue = []
        successful = 0

        for sim_id, info in self.pending_results.items():
            if info["status"] == "pending":
                # 检查模拟是否等待时间过长（30 分钟）
                if "start_time" not in info:
                    info["start_time"] = time.time()
                elif time.time() - info["start_time"] > 1800:  # 30 分钟
                    logging.warning(
                        f"模拟 {sim_id} 等待时间过长，标记为失败"
                    )
                    completed.append(sim_id)
                    continue
                try:
                    sim_progress_resp = self.sess.get(info["progress_url"])
                    logging.info(
                        f"正在检查模拟 {sim_id} 的 alpha: {info['alpha'][:50]}..."
                    )

                    # 处理速率限制
                    if sim_progress_resp.status_code == 429:
                        logging.info("达到速率限制，稍后重试")
                        continue

                    # 处理模拟限制
                    if "SIMULATION_LIMIT_EXCEEDED" in sim_progress_resp.text:
                        logging.info(
                            f"alpha 的模拟限制已达上限: {info['alpha']}"
                        )
                        retry_queue.append((info["alpha"], sim_id))
                        continue

                    # 处理重试等待
                    retry_after = sim_progress_resp.headers.get("Retry-After")
                    if retry_after:
                        try:
                            wait_time = int(
                                float(retry_after)
                            )  # 处理类似 "2.5" 的小数值
                            logging.info(f"需要等待 {wait_time} 秒后再检查")
                            time.sleep(wait_time)
                        except (ValueError, TypeError):
                            logging.warning(
                                f"无效的 Retry-After 头: {retry_after}, 使用默认 5 秒"
                            )
                            time.sleep(5)
                        continue

                    sim_result = sim_progress_resp.json()
                    status = sim_result.get("status")
                    logging.info(f"模拟 {sim_id} 状态: {status}")

                    # 记录调试细节
                    if status == "PENDING":
                        logging.debug(f"模拟 {sim_id} 仍在处理中...")
                    elif status == "RUNNING":
                        logging.debug(f"模拟 {sim_id} 正在运行...")
                    elif status not in ["COMPLETE", "ERROR"]:
                        logging.warning(
                            f"模拟 {sim_id} 状态未知: {status}"
                        )

                    if status == "COMPLETE":
                        alpha_id = sim_result.get("alpha")
                        if alpha_id:
                            alpha_resp = self.sess.get(
                                f"https://api.worldquantbrain.com/alphas/{alpha_id}"
                            )
                            if alpha_resp.status_code == 200:
                                alpha_data = alpha_resp.json()
                                fitness = alpha_data.get("is", {}).get("fitness")
                                logging.info(
                                    f"alpha {alpha_id} 完成，适应度: {fitness}"
                                )

                                self.results.append(
                                    {
                                        "alpha": info["alpha"],
                                        "result": sim_result,
                                        "alpha_data": alpha_data,
                                    }
                                )

                                # 候选标准收紧：优先选择高质量的候选
                                sharpe = alpha_data.get("is", {}).get("sharpe")
                                if fitness is not None and fitness > 1:
                                    logging.info(
                                        f"找到有潜力的 alpha! 适应度: {fitness}{', Sharpe: ' + str(sharpe) if sharpe is not None else ''}"
                                    )
                                    self.log_hopeful_alpha(info["alpha"], alpha_data)
                                    successful += 1
                                elif fitness is None:
                                    logging.warning(
                                        f"alpha {alpha_id} 没有适应度数据，跳过 hopeful alpha 记录"
                                    )
                    elif status == "ERROR":
                        logging.error(f"alpha 模拟失败: {info['alpha']}")
                    completed.append(sim_id)

                except Exception as e:
                    logging.error(f"检查结果时出错 {sim_id}: {str(e)}")

        # 移除已完成的模拟
        for sim_id in completed:
            del self.pending_results[sim_id]

        # 重新排队失败的模拟
        for alpha, sim_id in retry_queue:
            del self.pending_results[sim_id]
            self.retry_queue.add(alpha)

        return successful

    def test_alpha(self, alpha: str) -> Dict:
        result = self._test_alpha_impl(alpha)
        if result.get(
            "status"
        ) == "error" and "SIMULATION_LIMIT_EXCEEDED" in result.get("message", ""):
            self.retry_queue.add(alpha)
            return {"status": "queued", "message": "已加入重试队列"}
        return result

    def _test_alpha_impl(self, alpha_expression: str) -> Dict:
        """alpha 测试的实现，正确处理 URL"""

        def submit_simulation():
            simulation_data = {
                "type": "REGULAR",
                "settings": {
                    "instrumentType": "EQUITY",
                    "region": "USA",
                    "universe": "TOP3000",
                    "delay": 1,
                    "decay": 0,
                    "neutralization": "INDUSTRY",
                    "truncation": 0.01,
                    "pasteurization": "ON",
                    "unitHandling": "VERIFY",
                    "nanHandling": "OFF",
                    "language": "FASTEXPR",
                    "visualization": False,
                },
                "regular": alpha_expression,
            }
            return self.sess.post(
                "https://api.worldquantbrain.com/simulations", json=simulation_data
            )

        try:
            sim_resp = submit_simulation()

            # 处理认证错误
            if sim_resp.status_code == 401 or (
                sim_resp.status_code == 400
                and "authentication credentials" in sim_resp.text.lower()
            ):
                logger.warning("认证已过期，正在刷新会话...")
                self.setup_auth(self.credentials_path)  # 刷新认证
                sim_resp = submit_simulation()  # 使用新认证重试

            if sim_resp.status_code != 201:
                return {"status": "error", "message": sim_resp.text}

            sim_progress_url = sim_resp.headers.get("location")
            if not sim_progress_url:
                return {"status": "error", "message": "未收到进度 URL"}

            return {
                "status": "success",
                "result": {
                    "id": f"{time.time()}_{random.random()}",
                    "progress_url": sim_progress_url,
                },
            }

        except Exception as e:
            logger.error(f"测试 alpha {alpha_expression} 时出错: {str(e)}")
            return {"status": "error", "message": str(e)}

    def log_hopeful_alpha(self, expression: str, alpha_data: Dict) -> None:
        """将有潜力的 alpha 记录到 JSON 文件"""
        log_file = "hopeful_alphas.json"

        # 加载现有数据
        existing_data = []
        if os.path.exists(log_file):
            try:
                with open(log_file, "r") as f:
                    existing_data = json.load(f)
            except json.JSONDecodeError:
                print(f"Warning: Could not parse {log_file}, starting fresh")

        # Add new alpha with timestamp
        entry = {
            "expression": expression,  # Store just the expression string
            "timestamp": int(time.time()),
            "alpha_id": alpha_data.get("id", "unknown"),
            "fitness": alpha_data.get("is", {}).get("fitness"),
            "sharpe": alpha_data.get("is", {}).get("sharpe"),
            "turnover": alpha_data.get("is", {}).get("turnover"),
            "returns": alpha_data.get("is", {}).get("returns"),
            "grade": alpha_data.get("grade", "UNKNOWN"),
            "checks": alpha_data.get("is", {}).get("checks", []),
        }

        existing_data.append(entry)

        # Save updated data
        with open(log_file, "w") as f:
            json.dump(existing_data, f, indent=2)

        print(f"Logged promising alpha to {log_file}")

    def get_results(self) -> List[Dict]:
        """Get all processed results including retried alphas."""
        return self.results

    def fetch_submitted_alphas(self):
        """Fetch submitted alphas from the WorldQuant API with retry logic"""
        url = "https://api.worldquantbrain.com/users/self/alphas"
        params = {
            "limit": 100,
            "offset": 0,
            "status!=": "UNSUBMITTED%1FIS-FAIL",
            "order": "-dateCreated",
            "hidden": "false",
        }

        max_retries = 3
        retry_delay = 60  # seconds

        for attempt in range(max_retries):
            try:
                response = self.sess.get(url, params=params)
                if response.status_code == 429:  # Too Many Requests
                    wait_time = int(response.headers.get("Retry-After", retry_delay))
                    logger.info(
                        f"Rate limited. Waiting {wait_time} seconds before retry..."
                    )
                    time.sleep(wait_time)
                    continue

                response.raise_for_status()
                return response.json()["results"]

            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(
                        f"Attempt {attempt + 1} failed: {str(e)}. Retrying..."
                    )
                    time.sleep(retry_delay)
                else:
                    logger.error(
                        f"Failed to fetch submitted alphas after {max_retries} attempts: {e}"
                    )
                    return []

        return []


def extract_expressions(alphas):
    """Extract expressions from submitted alphas"""
    expressions = []
    for alpha in alphas:
        if alpha.get("regular") and alpha["regular"].get("code"):
            expressions.append(
                {
                    "expression": alpha["regular"]["code"],
                    "performance": {
                        "sharpe": alpha["is"].get("sharpe", 0),
                        "fitness": alpha["is"].get("fitness", 0),
                    },
                }
            )
    return expressions


def is_similar_to_existing(
    new_expression, existing_expressions, similarity_threshold=0.7
):
    """Check if new expression is too similar to existing ones"""
    for existing in existing_expressions:
        # Basic similarity checks
        if new_expression == existing["expression"]:
            return True

        # Check for structural similarity
        if (
            structural_similarity(new_expression, existing["expression"])
            > similarity_threshold
        ):
            return True

    return False


def calculate_similarity(expr1: str, expr2: str) -> float:
    """Calculate similarity between two expressions using token-based comparison."""
    # Normalize expressions
    expr1_tokens = set(tokenize_expression(normalize_expression(expr1)))
    expr2_tokens = set(tokenize_expression(normalize_expression(expr2)))

    if not expr1_tokens or not expr2_tokens:
        return 0.0

    # Calculate Jaccard similarity
    intersection = len(expr1_tokens.intersection(expr2_tokens))
    union = len(expr1_tokens.union(expr2_tokens))

    return intersection / union


def structural_similarity(expr1, expr2):
    """Calculate structural similarity between two expressions"""
    return calculate_similarity(expr1, expr2)  # Use our new similarity function


def normalize_expression(expr):
    """Normalize expression for comparison"""
    # Remove whitespace and convert to lowercase
    expr = re.sub(r"\s+", "", expr.lower())
    return expr


def tokenize_expression(expr):
    """Split expression into meaningful tokens"""
    # Split on operators and parentheses while keeping them
    tokens = re.findall(r"[\w._]+|[(),*/+-]", expr)
    return tokens


def generate_alpha():
    """Generate new alpha expression"""
    generator = AlphaGenerator("./credential.txt", "http://localhost:11434")
    data_fields = generator.get_data_fields()
    operators = generator.get_operators()

    # Fetch existing alphas first
    submitted_alphas = generator.fetch_submitted_alphas()
    existing_expressions = extract_expressions(submitted_alphas)

    max_attempts = 50
    attempts = 0

    while attempts < max_attempts:
        alpha_ideas = generator.generate_alpha_ideas_with_ollama(data_fields, operators)
        for idea in alpha_ideas:
            if not is_similar_to_existing(idea, existing_expressions):
                logger.info(f"Generated unique expression: {idea}")
                return idea

        attempts += 1
        logger.debug(f"Attempt {attempts}: All expressions were too similar")

    logger.warning("Failed to generate unique expression after maximum attempts")
    return None


def main():
    parser = argparse.ArgumentParser(
        description="使用 WorldQuant Brain API 和 Ollama/FinGPT 生成并测试 alpha 因子"
    )
    parser.add_argument(
        "--credentials",
        type=str,
        default="./credential.txt",
        help="凭证文件路径 (默认: ./credential.txt)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results",
        help="保存结果的目录 (默认: ./results)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=3,
        help="每批生成的 alpha 因子数量 (默认: 3)",
    )
    parser.add_argument(
        "--sleep-time",
        type=int,
        default=10,
        help="批次之间的休眠时间（秒） (默认: 10)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="设置日志级别 (默认: INFO)",
    )
    parser.add_argument(
        "--ollama-url",
        type=str,
        default="http://localhost:11434",
        help="Ollama API URL (默认: http://localhost:11434)",
    )
    parser.add_argument(
        "--ollama-model",
        type=str,
        default="deepseek-r1:8b",
        help="使用的 Ollama 模型 (默认: deepseek-r1:8b for RTX A4000)",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=2,
        help="最大并发模拟数 (默认: 2)",
    )

    args = parser.parse_args()

    # 配置日志
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),  # 输出到控制台
            logging.FileHandler("alpha_generator_ollama.log"),  # 同时记录到文件
        ],
    )

    # 如果输出目录不存在，则创建
    os.makedirs(args.output_dir, exist_ok=True)

    try:
        # 初始化 alpha 生成器
        generator = AlphaGenerator(
            args.credentials, args.ollama_url, args.max_concurrent
        )
        generator.model_name = args.ollama_model  # 设置模型名称
        generator.initial_model = args.ollama_model  # 设置初始模型用于重置

        # 获取数据字段和运算符
        print("正在获取数据字段和运算符...")
        data_fields = generator.get_data_fields()
        operators = generator.get_operators()

        batch_number = 1
        total_successful = 0

        print(f"开始连续 alpha 挖掘，批次大小: {args.batch_size}")
        print(f"结果将保存到 {args.output_dir}")
        print(f"使用 Ollama 地址: {args.ollama_url}")

        while True:
            try:
                logging.info(f"\n正在处理批次 #{batch_number}")
                logging.info("-" * 50)

                # 使用 Ollama 生成并提交批次
                alpha_ideas = generator.generate_alpha_ideas_with_ollama(
                    data_fields, operators
                )
                batch_successful = generator.test_alpha_batch(alpha_ideas)
                total_successful += batch_successful

                # 每几个批次执行一次 VRAM 清理
                generator.operation_count += 1
                if generator.operation_count % generator.vram_cleanup_interval == 0:
                    generator.cleanup_vram()

                # 保存批次结果
                results = generator.get_results()
                timestamp = int(time.time())
                output_file = os.path.join(
                    args.output_dir, f"batch_{batch_number}_{timestamp}.json"
                )
                with open(output_file, "w") as f:
                    json.dump(results, f, indent=2)

                logging.info(f"批次 {batch_number} 结果已保存到 {output_file}")
                logging.info(f"批次成功数: {batch_successful}")
                logging.info(f"总成功 alpha 数: {total_successful}")

                batch_number += 1

                # 批次之间休眠
                print(f"休眠 {args.sleep_time} 秒...")
                sleep(args.sleep_time)

            except Exception as e:
                logging.error(f"批次 {batch_number} 出错: {str(e)}")
                logging.info("休眠 5 分钟后重试...")
                sleep(300)
                continue

    except KeyboardInterrupt:
        logging.info("\n停止 alpha 挖掘...")
        logging.info(f"总处理批次: {batch_number - 1}")
        logging.info(f"总成功 alpha 数: {total_successful}")
        return 0

    except Exception as e:
        logging.error(f"致命错误: {str(e)}")
        return 1


if __name__ == "__main__":
    exit(main())
