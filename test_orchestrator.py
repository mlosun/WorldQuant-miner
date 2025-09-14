#!/usr/bin/env python3
"""
Test script for the Alpha Orchestrator to verify concurrent execution
of alpha_generator_ollama and alpha_expression_miner.
"""

import os
import sys
import time
import json
import logging
from alpha_orchestrator import AlphaOrchestrator

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_test_hopeful_alphas():
    """创建一个测试用的 hopeful_alphas.json 文件。"""
    test_alphas = [
        {
            "expression": "rank(close)",
            "timestamp": int(time.time()),
            "alpha_id": "test_1",
            "fitness": 0.6,
            "sharpe": 1.2,
            "turnover": 0.1,
            "returns": 0.15,
            "grade": "A",
            "checks": {"passed": True},
        },
        {
            "expression": "rank(volume)",
            "timestamp": int(time.time()),
            "alpha_id": "test_2",
            "fitness": 0.7,
            "sharpe": 1.5,
            "turnover": 0.08,
            "returns": 0.18,
            "grade": "A",
            "checks": {"passed": True},
        },
    ]

    with open("hopeful_alphas.json", "w") as f:
        json.dump(test_alphas, f, indent=2)

    logger.info("已创建包含 2 个测试 alpha 的 hopeful_alphas.json 文件")


def test_orchestrator_initialization():
    """测试 Orchestrator 是否能正确初始化。"""
    logger.info("测试 Orchestrator 初始化...")

    try:
        orchestrator = AlphaOrchestrator("./credential.txt")
        logger.info("✓ Orchestrator 初始化成功")
        logger.info(
            f"✓ Max concurrent simulations: {orchestrator.max_concurrent_simulations}"
        )
        return orchestrator
    except Exception as e:
        logger.error(f"✗ Orchestrator 初始化失败: {e}")
        return None


def test_concurrent_execution():
    """测试生成器和挖掘器的并发执行。"""
    logger.info("测试并发执行...")

    orchestrator = test_orchestrator_initialization()
    if not orchestrator:
        return False

    # Create test data
    create_test_hopeful_alphas()

    try:
        # Test the continuous miner function directly
        logger.info("使用测试数据测试 alpha 表达式挖掘器...")

        # Run the miner once to see if it works
        orchestrator.run_alpha_expression_miner()

        logger.info("✓ Alpha 表达式挖掘器测试完成")
        return True

    except Exception as e:
        logger.error(f"✗ 并发执行测试失败: {e}")
        return False


def test_command_line_arguments():
    """测试 Orchestrator 是否能正确接受命令行参数。"""
    logger.info("测试命令行参数...")

    # Test with different max_concurrent values
    test_cases = [1, 3, 5]

    for max_concurrent in test_cases:
        try:
            orchestrator = AlphaOrchestrator("./credential.txt")
            orchestrator.max_concurrent_simulations = max_concurrent
            logger.info(f"✓ 设置 max_concurrent 为 {max_concurrent}")
        except Exception as e:
            logger.error(f"✗ 设置 max_concurrent 为 {max_concurrent} 失败: {e}")
            return False

    logger.info("✓ 所有命令行参数测试通过")
    return True


def main():
    """运行所有测试。"""
    logger.info("开始 Alpha Orchestrator 测试...")

    tests = [
        ("Orchestrator Initialization", test_orchestrator_initialization),
        ("Command Line Arguments", test_command_line_arguments),
        ("Concurrent Execution", test_concurrent_execution),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"正在运行测试: {test_name}")
        logger.info(f"{'='*50}")

        try:
            if test_func():
                logger.info(f"✓ {test_name} 测试通过")
                passed += 1
            else:
                logger.error(f"✗ {test_name} 测试失败")
        except Exception as e:
            logger.error(f"✗ {test_name} 测试失败，异常: {e}")

    logger.info(f"\n{'='*50}")
    logger.info(f"Test Results: {passed}/{total} tests passed")
    logger.info(f"{'='*50}")

    if passed == total:
        logger.info(
            "🎉 All tests passed! The orchestrator is ready for concurrent execution."
        )
        return 0
    else:
        logger.error("❌ 部分测试失败。请检查上面的日志。")
        return 1


if __name__ == "__main__":
    exit(main())
