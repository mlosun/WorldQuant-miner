import argparse
import requests
import json
import os
import re
from time import sleep
from requests.auth import HTTPBasicAuth
from typing import List, Dict, Tuple
import time
import logging

# 在文件顶部配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("alpha_miner.log")],
)
logger = logging.getLogger(__name__)


class AlphaExpressionMiner:
    def __init__(self, credentials_path: str):
        logger.info("初始化 AlphaExpressionMiner")
        self.sess = requests.Session()
        self.setup_auth(credentials_path)

    def setup_auth(self, credentials_path: str) -> None:
        """设置与 WorldQuant Brain 的认证"""
        logger.info(f"从 {credentials_path} 加载凭证")
        with open(credentials_path) as f:
            credentials = json.load(f)

        username, password = credentials
        self.sess.auth = HTTPBasicAuth(username, password)

        logger.info("正在与 WorldQuant Brain 进行认证...")
        response = self.sess.post("https://api.worldquantbrain.com/authentication")
        logger.info(f"认证响应状态: {response.status_code}")

        if response.status_code != 201:
            logger.error(f"认证失败: {response.text}")
            raise Exception(f"认证失败: {response.text}")
        logger.info("认证成功")

    def remove_alpha_from_hopeful(
        self, expression: str, hopeful_file: str = "hopeful_alphas.json"
    ) -> bool:
        """从 hopeful_alphas.json 中移除已挖掘的 alpha"""
        try:
            if not os.path.exists(hopeful_file):
                logger.warning(f"未找到 hopeful_alphas 文件 {hopeful_file}")
                return False

            # 修改前创建备份
            backup_file = f"{hopeful_file}.backup.{int(time.time())}"
            import shutil

            shutil.copy2(hopeful_file, backup_file)
            logger.debug(f"创建备份: {backup_file}")

            with open(hopeful_file, "r") as f:
                hopeful_alphas = json.load(f)

            # 查找并移除匹配表达式的 alpha
            original_count = len(hopeful_alphas)
            removed_alphas = []
            remaining_alphas = []

            for alpha in hopeful_alphas:
                if alpha.get("expression") == expression:
                    removed_alphas.append(alpha)
                else:
                    remaining_alphas.append(alpha)

            removed_count = len(removed_alphas)

            if removed_count > 0:
                # 保存更新后的文件
                with open(hopeful_file, "w") as f:
                    json.dump(remaining_alphas, f, indent=2)
                logger.info(
                    f"从 {hopeful_file} 中移除了 {removed_count} 个表达式为 '{expression}' 的 alpha"
                )
                logger.debug(f"文件中剩余的 alpha 数量: {len(remaining_alphas)}")
                return True
            else:
                logger.info(
                    f"在 {hopeful_file} 中未找到匹配表达式 {expression} 的 alpha"
                )
                logger.debug(
                    f"可用的表达式: {[alpha.get('expression', 'N/A') for alpha in hopeful_alphas[:5]]}"
                )
                return False

        except json.JSONDecodeError as e:
            logger.error(f"{hopeful_file} 中的 JSON 无效: {e}")
            return False
        except Exception as e:
            logger.error(f"从 {hopeful_file} 中移除 alpha 时出错: {e}")
            return False

    def parse_expression(self, expression: str) -> List[Dict]:
        """解析 alpha 表达式以查找数值参数及其位置"""
        logger.info(f"正在解析表达式: {expression}")
        parameters = []
        # 匹配以下条件的数字：
        # 1. 前面是 '(' 或 ',' 或空格
        # 2. 不是变量名的一部分（前后没有字母）
        # 3. 可以是整数或小数
        for match in re.finditer(r"(?<=[,()\s])(-?\d*\.?\d+)(?![a-zA-Z])", expression):
            number_str = match.group()
            try:
                number = float(number_str)
            except ValueError:
                continue
            start_pos = match.start()
            end_pos = match.end()
            parameters.append(
                {
                    "value": number,
                    "start": start_pos,
                    "end": end_pos,
                    "context": expression[
                        max(0, start_pos - 20) : min(len(expression), end_pos + 20)
                    ],
                    "is_integer": number.is_integer(),
                }
            )
            logger.debug(f"找到参数: {number} 在位置 {start_pos}-{end_pos}")

        logger.info(f"找到 {len(parameters)} 个需要变动的参数")
        return parameters

    def get_user_parameter_selection(self, parameters: List[Dict]) -> List[Dict]:
        """交互式获取用户选择的参数"""
        if not parameters:
            logger.info("表达式中未找到参数")
            return []

        print("\n在表达式中找到以下参数:")
        for i, param in enumerate(parameters, 1):
            print(f"{i}. 值: {param['value']} | 上下文: ...{param['context']}...")

        while True:
            try:
                selection = input(
                    "\n输入要变动的参数编号（逗号分隔，或输入 'all' 选择全部）: "
                )
                if selection.lower() == "all":
                    selected_indices = list(range(len(parameters)))
                else:
                    selected_indices = [
                        int(x.strip()) - 1 for x in selection.split(",")
                    ]
                    if not all(0 <= i < len(parameters) for i in selected_indices):
                        raise ValueError("无效的参数编号")
                break
            except ValueError as e:
                print(f"输入无效: {e}. 请重试。")

        selected_params = [parameters[i] for i in selected_indices]
        return selected_params

    def get_parameter_ranges(
        self, parameters: List[Dict], auto_mode: bool = False
    ) -> List[Dict]:
        """获取每个选定参数的范围和步长"""
        for param in parameters:
            if auto_mode:
                # 自动化模式下使用默认范围
                original_value = param["value"]
                if param["is_integer"]:
                    # 对于整数，使用 ±20% 范围，步长为 1
                    range_val = max(1, abs(original_value) * 0.2)
                    min_val = max(1, original_value - range_val)
                    max_val = original_value + range_val
                    step = 1
                else:
                    # 对于浮点数，使用 ±10% 范围，步长为范围的 10%
                    range_val = abs(original_value) * 0.1
                    min_val = original_value - range_val
                    max_val = original_value + range_val
                    step = range_val / 5  # 在范围内分 5 步

                logger.info(
                    f"自动化模式: 参数 {param['value']} -> 范围 [{min_val:.2f}, {max_val:.2f}], 步长 {step:.2f}"
                )
            else:
                # 交互模式 - 获取用户输入
                while True:
                    try:
                        print(
                            f"\n参数: {param['value']} | 上下文: ...{param['context']}..."
                        )
                        range_input = input(
                            "输入范围（例如 '10' 表示 ±10，或 '5,15' 表示 5 到 15）: "
                        )
                        if "," in range_input:
                            min_val, max_val = map(float, range_input.split(","))
                        else:
                            range_val = float(range_input)
                            min_val = param["value"] - range_val
                            max_val = param["value"] + range_val

                        step = float(input("Enter step size: "))
                        if step <= 0:
                            raise ValueError("Step size must be positive")
                        if min_val >= max_val:
                            raise ValueError("Min value must be less than max value")

                        # If the original value was an integer, ensure step is also an integer
                        if param["is_integer"] and not step.is_integer():
                            print(
                                "Warning: Original value is integer, rounding step to nearest integer"
                            )
                            step = round(step)

                        break
                    except ValueError as e:
                        print(f"Invalid input: {e}. Please try again.")

            param["min"] = min_val
            param["max"] = max_val
            param["step"] = step

        return parameters

    def generate_variations(self, expression: str, parameters: List[Dict]) -> List[str]:
        """Generate variations of the expression based on user-selected parameters and ranges."""
        logger.info("Generating variations based on selected parameters")
        variations = []

        # Sort parameters in reverse order to modify from end to start
        parameters.sort(reverse=True, key=lambda x: x["start"])

        # Generate all combinations of parameter values
        param_values = []
        for param in parameters:
            values = []
            current = param["min"]
            while current <= param["max"]:
                # Format the number appropriately based on whether it's an integer
                if param["is_integer"]:
                    value = str(int(round(current)))
                else:
                    # Format to remove trailing zeros and unnecessary decimal points
                    value = f"{current:.10f}".rstrip("0").rstrip(".")
                values.append(value)
                current += param["step"]

            # Add original value if not already included
            original_value = (
                str(int(param["value"]))
                if param["is_integer"]
                else f"{param['value']:.10f}".rstrip("0").rstrip(".")
            )
            if original_value not in values:
                values.append(original_value)

            param_values.append(values)

        # Generate all combinations
        from itertools import product

        for value_combination in product(*param_values):
            new_expr = expression
            for param, value in zip(parameters, value_combination):
                new_expr = new_expr[: param["start"]] + value + new_expr[param["end"] :]
            variations.append(new_expr)
            logger.debug(f"Generated variation: {new_expr}")

        logger.info(f"Generated {len(variations)} total variations")
        return variations

    def test_alpha(self, alpha_expression: str) -> Dict:
        """Test an alpha expression using WorldQuant Brain simulation."""
        logger.info(f"Testing alpha: {alpha_expression}")

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

        sim_resp = self.sess.post(
            "https://api.worldquantbrain.com/simulations", json=simulation_data
        )
        logger.info(f"Simulation creation response: {sim_resp.status_code}")

        if sim_resp.status_code != 201:
            logger.error(f"Simulation creation failed: {sim_resp.text}")
            return {"status": "error", "message": sim_resp.text}

        sim_progress_url = sim_resp.headers.get("location")
        if not sim_progress_url:
            logger.error("No simulation ID received in response headers")
            return {"status": "error", "message": "No simulation ID received"}

        logger.info(f"Monitoring simulation at: {sim_progress_url}")

        # Monitor simulation progress
        retry_count = 0
        max_retries = 3
        while True:
            try:
                sim_progress_resp = self.sess.get(sim_progress_url)

                # Handle empty response
                if not sim_progress_resp.text.strip():
                    logger.debug("Empty response, simulation still initializing...")
                    sleep(10)
                    continue

                # Try to parse JSON response
                try:
                    progress_data = sim_progress_resp.json()
                except json.JSONDecodeError as e:
                    logger.warning(
                        f"Failed to decode JSON response: {sim_progress_resp.text}"
                    )
                    retry_count += 1
                    if retry_count > max_retries:
                        logger.error("Max retries exceeded for JSON decode")
                        return {
                            "status": "error",
                            "message": "Failed to decode simulation response",
                        }
                    sleep(10)
                    continue

                status = progress_data.get("status")
                logger.info(f"Simulation status: {status}")

                if status == "COMPLETE" or status == "WARNING":
                    logger.info("Simulation completed successfully")
                    return {"status": "success", "result": progress_data}
                elif status in ["FAILED", "ERROR"]:
                    logger.error(f"Simulation failed: {progress_data}")
                    return {"status": "error", "message": progress_data}

                sleep(10)

            except requests.exceptions.RequestException as e:
                logger.error(f"Request error: {str(e)}")
                retry_count += 1
                if retry_count > max_retries:
                    return {
                        "status": "error",
                        "message": f"Request failed after {max_retries} retries",
                    }
                sleep(10)


def main():
    parser = argparse.ArgumentParser(description="Mine alpha expression variations")
    parser.add_argument(
        "--credentials",
        type=str,
        default="./credential.txt",
        help="Path to credentials file (default: ./credential.txt)",
    )
    parser.add_argument(
        "--expression",
        type=str,
        required=True,
        help="Base alpha expression to mine variations from",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="mined_expressions.json",
        help="Output file for results (default: mined_expressions.json)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (default: INFO)",
    )
    parser.add_argument(
        "--auto-mode",
        action="store_true",
        help="Run in automated mode without user interaction",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="mined_expressions.json",
        help="Output file for results (default: mined_expressions.json)",
    )

    args = parser.parse_args()

    # Update log level if specified
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    logger.info(f"Starting alpha expression mining with parameters:")
    logger.info(f"Expression: {args.expression}")
    logger.info(f"Output file: {args.output}")

    miner = AlphaExpressionMiner(args.credentials)

    # Parse expression and get parameters
    parameters = miner.parse_expression(args.expression)

    # Get parameter selection (automated or interactive)
    if args.auto_mode:
        # In auto mode, select all parameters
        selected_params = parameters
        logger.info(f"Auto mode: selected all {len(selected_params)} parameters")
    else:
        # Get user selection for parameters to vary
        selected_params = miner.get_user_parameter_selection(parameters)

    if not selected_params:
        logger.info("No parameters selected for variation")
        # Still remove the alpha from hopeful_alphas.json even if no parameters found
        logger.info(
            "Mining completed (no parameters to vary), removing alpha from hopeful_alphas.json"
        )
        removed = miner.remove_alpha_from_hopeful(args.expression)
        if removed:
            logger.info(
                f"Successfully removed alpha '{args.expression}' from hopeful_alphas.json"
            )
        else:
            logger.warning(
                f"Could not remove alpha '{args.expression}' from hopeful_alphas.json (may not exist)"
            )
        return

    # Get ranges and steps for selected parameters
    selected_params = miner.get_parameter_ranges(
        selected_params, auto_mode=args.auto_mode
    )

    # Generate variations
    variations = miner.generate_variations(args.expression, selected_params)

    # Test variations
    results = []
    total = len(variations)
    for i, var in enumerate(variations, 1):
        logger.info(f"Testing variation {i}/{total}: {var}")
        result = miner.test_alpha(var)
        if result["status"] == "success":
            logger.info(f"Successful test for: {var}")
            results.append({"expression": var, "result": result["result"]})
        else:
            logger.error(f"Failed to test variation: {var}")
            logger.error(f"Error: {result['message']}")

    # Save results
    output_file = args.output_file if hasattr(args, "output_file") else args.output
    logger.info(f"Saving {len(results)} results to {output_file}")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    # Always remove the mined alpha from hopeful_alphas.json after completion
    # This prevents the same alpha from being processed again
    logger.info("Mining completed, removing alpha from hopeful_alphas.json")
    removed = miner.remove_alpha_from_hopeful(args.expression)
    if removed:
        logger.info(
            f"Successfully removed alpha '{args.expression}' from hopeful_alphas.json"
        )
    else:
        logger.warning(
            f"Could not remove alpha '{args.expression}' from hopeful_alphas.json (may not exist)"
        )

    logger.info("Mining complete")


if __name__ == "__main__":
    main()
