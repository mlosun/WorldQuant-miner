#!/usr/bin/env python3
"""
Naive-Ollma Docker 环境的健康检查脚本
"""

import requests
import json
import sys
import os
from requests.auth import HTTPBasicAuth


def check_ollama():
    """检查 Ollama 是否正在运行且 FinGPT 模型是否可用"""
    print("🔍 正在检查 Ollama...")

    try:
        # 检查 Ollama 是否正在运行
        response = requests.get("http://localhost:11434/api/tags", timeout=10)
        if response.status_code == 200:
            models = response.json().get("models", [])
            print(f"✅ Ollama 正在运行，共有 {len(models)} 个模型")

            # 检查 FinGPT 模型
            fingpt_models = [m for m in models if "fingpt" in m.get("name", "").lower()]
            if fingpt_models:
                print(f"✅ 已找到 FinGPT 模型: {fingpt_models[0]['name']}")
                return True
            else:
                print("⚠️  未找到 FinGPT 模型。您可能需要拉取它:")
                print("   docker-compose exec naive-ollma ollama pull fingpt")
                return False
        else:
            print(f"❌ Ollama API 返回状态码 {response.status_code}")
            return False

    except requests.exceptions.ConnectionError:
        print("❌ 无法连接到 Ollama。它是否正在运行？")
        return False
    except Exception as e:
        print(f"❌ 检查 Ollama 时出错: {e}")
        return False


def check_worldquant_credentials():
    """检查 WorldQuant Brain 凭证是否有效"""
    print("🔍 正在检查 WorldQuant Brain 凭证...")

    if not os.path.exists("credential.txt"):
        print("❌ 未找到 credential.txt")
        return False

    try:
        with open("credential.txt") as f:
            credentials = json.load(f)

        if not isinstance(credentials, list) or len(credentials) != 2:
            print('❌ 凭证格式无效。预期格式: ["username", "password"]')
            return False

        username, password = credentials

        # 测试认证
        session = requests.Session()
        session.auth = HTTPBasicAuth(username, password)

        response = session.post(
            "https://api.worldquantbrain.com/authentication", timeout=30
        )

        if response.status_code == 201:
            print("✅ WorldQuant Brain 认证成功")
            return True
        else:
            print(f"❌ WorldQuant Brain 认证失败: {response.status_code}")
            print(f"   响应: {response.text[:200]}...")
            return False

    except json.JSONDecodeError:
        print("❌ credential.txt 中的 JSON 格式无效")
        return False
    except Exception as e:
        print(f"❌ 检查凭证时出错: {e}")
        return False


def check_docker_services():
    """检查 Docker 服务是否正在运行"""
    print("🔍 正在检查 Docker 服务...")

    try:
        import subprocess

        result = subprocess.run(
            ["docker-compose", "ps"], capture_output=True, text=True, timeout=10
        )

        if result.returncode == 0:
            if "Up" in result.stdout:
                print("✅ Docker 服务正在运行")
                return True
            else:
                print("⚠️  Docker 服务未运行")
                return False
        else:
            print(f"❌ 检查 Docker 服务时出错: {result.stderr}")
            return False

    except FileNotFoundError:
        print("❌ 未找到 docker-compose。Docker 是否已安装？")
        return False
    except Exception as e:
        print(f"❌ 检查 Docker 服务时出错: {e}")
        return False


def test_ollama_generation():
    """测试 Ollama 是否能生成响应"""
    print("🔍 正在测试 Ollama 生成能力...")

    try:
        test_prompt = "Generate a simple alpha factor expression: ts_mean(returns, 20)"

        data = {
            "model": "fingpt",
            "prompt": test_prompt,
            "stream": False,
            "temperature": 0.1,
            "num_predict": 100,  # 对于 Ollama 使用 num_predict 而不是 max_tokens
        }

        response = requests.post(
            "http://localhost:11434/api/generate", json=data, timeout=60
        )

        if response.status_code == 200:
            result = response.json()
            if "response" in result:
                print("✅ Ollama 生成测试成功")
                return True
            else:
                print("❌ 意外的 Ollama 响应格式")
                return False
        else:
            print(f"❌ Ollama 生成测试失败: {response.status_code}")
            return False

    except Exception as e:
        print(f"❌ 测试 Ollama 生成时出错: {e}")
        return False


def main():
    """运行所有健康检查"""
    print("🏥 Naive-Ollma 健康检查")
    print("=" * 40)

    checks = [
        ("Docker 服务", check_docker_services),
        ("Ollama", check_ollama),
        ("Ollama 生成", test_ollama_generation),
        ("WorldQuant 凭证", check_worldquant_credentials),
    ]

    results = []

    for name, check_func in checks:
        print(f"\n{name}:")
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ {name} 检查出错: {e}")
            results.append((name, False))

    print("\n" + "=" * 40)
    print("📊 健康检查摘要:")

    all_passed = True
    for name, passed in results:
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"   {name}: {status}")
        if not passed:
            all_passed = False

    print("\n" + "=" * 40)
    if all_passed:
        print("🎉 所有检查都通过了！您的环境已准备就绪。")
        print("   运行: docker-compose up -d")
        return 0
    else:
        print("⚠️  某些检查失败。请查看上面的问题。")
        print("   查看 README_Docker.md 获取故障排除提示。")
        return 1


if __name__ == "__main__":
    sys.exit(main())
