#!/usr/bin/env python3
"""
VRAM 监控器 - WorldQuant Alpha 挖矿系统
监控 GPU 内存使用情况，并在 VRAM 使用过高时重启服务。
"""

import subprocess
import time
import logging
import json
import os
from typing import Dict, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("vram_monitor.log")],
)
logger = logging.getLogger(__name__)


class VRAMMonitor:
    def __init__(self, vram_threshold: float = 0.9, check_interval: int = 60):
        """
        初始化 VRAM 监控器。

        参数:
            vram_threshold: 触发清理的 VRAM 使用百分比 (0.0-1.0)
            check_interval: 检查 VRAM 使用情况的间隔时间（秒）
        """
        self.vram_threshold = vram_threshold
        self.check_interval = check_interval
        self.restart_count = 0
        self.max_restarts = 3

    def get_gpu_info(self) -> List[Dict]:
        """使用 nvidia-smi 获取 GPU 信息。"""
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode != 0:
                logger.error(f"nvidia-smi failed: {result.stderr}")
                return []

            gpu_info = []
            for line in result.stdout.strip().split("\n"):
                if line.strip():
                    parts = line.split(", ")
                    if len(parts) >= 5:
                        gpu_info.append(
                            {
                                "index": int(parts[0]),
                                "name": parts[1],
                                "memory_used": int(parts[2]),
                                "memory_total": int(parts[3]),
                                "utilization": int(parts[4]),
                            }
                        )

            return gpu_info

        except subprocess.TimeoutExpired:
            logger.error("nvidia-smi command timed out")
            return []
        except Exception as e:
            logger.error(f"Error getting GPU info: {e}")
            return []

    def check_vram_usage(self) -> bool:
        """检查 VRAM 使用是否超过阈值。"""
        gpu_info = self.get_gpu_info()

        for gpu in gpu_info:
            memory_usage = gpu["memory_used"] / gpu["memory_total"]
            logger.info(
                f"GPU {gpu['index']} ({gpu['name']}): "
                f"{gpu['memory_used']}MB/{gpu['memory_total']}MB "
                f"({memory_usage:.1%}) - Utilization: {gpu['utilization']}%"
            )

            if memory_usage > self.vram_threshold:
                logger.warning(
                    f"GPU {gpu['index']} VRAM usage ({memory_usage:.1%}) "
                    f"exceeds threshold ({self.vram_threshold:.1%})"
                )
                return True

        return False

    def restart_ollama_service(self) -> bool:
        """重启 Ollama 服务以释放 VRAM。"""
        try:
            logger.info("正在重启 Ollama 服务...")

            # Stop Ollama
            subprocess.run(
                [
                    "docker-compose",
                    "-f",
                    "docker-compose.gpu.yml",
                    "stop",
                    "naive-ollma",
                ],
                timeout=30,
            )

            # Wait a bit for cleanup
            time.sleep(10)

            # Start Ollama
            subprocess.run(
                [
                    "docker-compose",
                    "-f",
                    "docker-compose.gpu.yml",
                    "start",
                    "naive-ollma",
                ],
                timeout=30,
            )

            # Wait for service to be ready
            time.sleep(30)

            logger.info("Ollama 服务重启成功")
            return True

        except subprocess.TimeoutExpired:
            logger.error("Timeout while restarting Ollama service")
            return False
        except Exception as e:
            logger.error(f"Error restarting Ollama service: {e}")
            return False

    def cleanup_vram(self) -> bool:
        """尝试在不重启的情况下清理 VRAM。"""
        try:
            logger.info("正在尝试清理 VRAM...")

            # Try to restart just the Ollama container
            result = subprocess.run(
                [
                    "docker",
                    "exec",
                    "naive-ollma-gpu",
                    "sh",
                    "-c",
                    "pkill -f ollama && sleep 5 && ollama serve",
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )

            if result.returncode == 0:
                logger.info("VRAM 清理完成")
                return True
            else:
                logger.warning(f"VRAM cleanup failed: {result.stderr}")
                return False

        except Exception as e:
            logger.error(f"Error during VRAM cleanup: {e}")
            return False

    def run(self):
        """主监控循环。"""
        logger.info(f"启动 VRAM 监控器，阈值设置为 {self.vram_threshold:.1%}")
        logger.info(f"检查间隔: {self.check_interval} 秒")

        while True:
            try:
                if self.check_vram_usage():
                    logger.warning("检测到 VRAM 使用过高！")

                    # Try cleanup first
                    if self.cleanup_vram():
                        logger.info("VRAM 清理成功")
                        time.sleep(30)  # Wait and check again
                        continue

                    # If cleanup failed and we haven't exceeded max restarts
                    if self.restart_count < self.max_restarts:
                        logger.warning(
                            f"正在尝试重启服务 ({self.restart_count + 1}/{self.max_restarts})"
                        )
                        if self.restart_ollama_service():
                            self.restart_count += 1
                            time.sleep(60)  # Wait longer after restart
                            continue
                    else:
                        logger.error(
                            "Maximum restart attempts exceeded. Manual intervention required."
                        )
                        break

                # Reset restart count if VRAM usage is normal
                if self.restart_count > 0:
                    logger.info("VRAM usage normal, resetting restart count")
                    self.restart_count = 0

                time.sleep(self.check_interval)

            except KeyboardInterrupt:
                logger.info("VRAM monitor stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.check_interval)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Monitor VRAM usage and manage GPU resources"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.9,
        help="VRAM usage threshold (0.0-1.0, default: 0.9)",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Check interval in seconds (default: 60)",
    )

    args = parser.parse_args()

    monitor = VRAMMonitor(vram_threshold=args.threshold, check_interval=args.interval)

    monitor.run()


if __name__ == "__main__":
    main()
