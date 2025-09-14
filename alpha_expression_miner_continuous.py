#!/usr/bin/env python3

import json
import time
import logging
import argparse
import os
from alpha_expression_miner import AlphaExpressionMiner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("alpha_expression_miner_continuous.log"),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger(__name__)


class ContinuousAlphaExpressionMiner:
    def __init__(self, credentials_path, ollama_url=None, mining_interval=6):
        self.miner = AlphaExpressionMiner(credentials_path)
        self.ollama_url = ollama_url
        self.mining_interval = mining_interval * 3600  # 将小时转换为秒
        self.hopeful_alphas_file = "hopeful_alphas.json"

    def get_hopeful_alphas(self):
        """从hopeful_alphas.json读取alpha表达式"""
        try:
            if os.path.exists(self.hopeful_alphas_file):
                with open(self.hopeful_alphas_file, "r") as f:
                    data = json.load(f)
                    return data.get("alphas", [])
            else:
                logger.warning(f"文件 {self.hopeful_alphas_file} 未找到")
                return []
        except Exception as e:
            logger.error(f"读取 {self.hopeful_alphas_file} 时出错: {e}")
            return []

    def mine_alpha_expression(self, expression):
        """挖掘单个alpha表达式的变体"""
        try:
            logger.info(f"开始挖掘表达式: {expression}")

            # 解析表达式并获取参数
            parameters = self.miner.parse_expression(expression)

            if not parameters:
                logger.info(f"未找到表达式 {expression} 的参数")
                return False

            # 选择所有参数进行变体生成
            selected_params = parameters
            logger.info(f"选择了 {len(selected_params)} 个参数进行变体生成")

            # 获取选定参数的范围和步长
            selected_params = self.miner.get_parameter_ranges(
                selected_params, auto_mode=True
            )

            # 生成变体
            variations = self.miner.generate_variations(expression, selected_params)
            logger.info(f"生成了 {len(variations)} 个变体")

            # 测试变体
            results = []
            total = len(variations)
            for i, var in enumerate(variations, 1):
                logger.info(f"测试变体 {i}/{total}: {var}")
                result = self.miner.test_alpha(var)
                if result["status"] == "success":
                    logger.info(f"测试成功: {var}")
                    results.append({"expression": var, "result": result["result"]})
                else:
                    logger.error(f"测试变体失败: {var}")
                    logger.error(f"错误: {result['message']}")

            # 保存结果
            if results:
                timestamp = int(time.time())
                output_file = f"mined_expressions_{timestamp}.json"
                logger.info(f"将 {len(results)} 个结果保存到 {output_file}")
                with open(output_file, "w") as f:
                    json.dump(results, f, indent=2)

            # 从hopeful_alphas.json中移除已挖掘的alpha
            logger.info("挖掘完成，从hopeful_alphas.json中移除alpha")
            removed = self.miner.remove_alpha_from_hopeful(expression)
            if removed:
                logger.info(
                    f"成功从hopeful_alphas.json中移除alpha '{expression}'"
                )
            else:
                logger.warning(
                    f"无法从hopeful_alphas.json中移除alpha '{expression}'"
                )

            return True

        except Exception as e:
            logger.error(f"挖掘表达式 {expression} 时出错: {e}")
            return False

    def run_continuous_mining(self):
        """持续挖掘alpha表达式"""
        logger.info(
            f"开始持续挖掘alpha表达式，间隔为 {self.mining_interval/3600} 小时"
        )

        while True:
            try:
                # 获取待挖掘的alpha
                hopeful_alphas = self.get_hopeful_alphas()

                if not hopeful_alphas:
                    logger.info("未找到待挖掘的alpha，等待下一轮...")
                    time.sleep(self.mining_interval)
                    continue

                logger.info(f"找到 {len(hopeful_alphas)} 个待挖掘的alpha")

                # 处理每个alpha
                for alpha in hopeful_alphas:
                    try:
                        success = self.mine_alpha_expression(alpha)
                        if success:
                            logger.info(f"成功挖掘alpha: {alpha}")
                        else:
                            logger.warning(f"挖掘alpha失败: {alpha}")

                        # 每个alpha之间的小延迟
                        time.sleep(10)

                    except Exception as e:
                        logger.error(f"处理alpha {alpha} 时出错: {e}")
                        continue

                logger.info(
                    f"挖掘周期完成，等待 {self.mining_interval/3600} 小时后开始下一轮..."
                )
                time.sleep(self.mining_interval)

            except KeyboardInterrupt:
                logger.info("收到中断信号，停止持续挖掘...")
                break
            except Exception as e:
                logger.error(f"持续挖掘周期中出错: {e}")
                logger.info("等待5分钟后重试...")
                time.sleep(300)


def main():
    parser = argparse.ArgumentParser(description="持续Alpha表达式挖掘器")
    parser.add_argument(
        "--credentials",
        type=str,
        default="./credential.txt",
        help="凭证文件路径 (默认: ./credential.txt)",
    )
    parser.add_argument(
        "--ollama-url",
        type=str,
        default="http://localhost:11434",
        help="Ollama API URL (默认: http://localhost:11434)",
    )
    parser.add_argument(
        "--mining-interval",
        type=int,
        default=6,
        help="挖掘间隔（小时） (默认: 6)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="设置日志级别 (默认: INFO)",
    )

    args = parser.parse_args()

    # 配置日志
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    try:
        miner = ContinuousAlphaExpressionMiner(
            args.credentials, args.ollama_url, args.mining_interval
        )
        miner.run_continuous_mining()
    except Exception as e:
        logger.error(f"致命错误: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
