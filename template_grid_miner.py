import argparse
import json
import logging
import os
import time
from typing import List
import requests
from requests.auth import HTTPBasicAuth

logger = logging.getLogger(__name__)


def auth_session(credentials_path: str) -> requests.Session:
    with open(credentials_path, "r", encoding="utf-8") as f:
        u, p = json.load(f)
    s = requests.Session()
    s.auth = HTTPBasicAuth(u, p)
    r = s.post("https://api.worldquantbrain.com/authentication")
    if r.status_code != 201:
        raise RuntimeError(f"认证失败: {r.text}")
    return s


def submit_sim(s: requests.Session, expression: str) -> dict:
    data = {
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
        "regular": expression,
    }
    r = s.post("https://api.worldquantbrain.com/simulations", json=data)
    if r.status_code != 201:
        return {"status": "error", "message": r.text}
    progress_url = r.headers.get("location")
    return {"status": "success", "progress_url": progress_url}


def monitor_sim(s: requests.Session, progress_url: str, timeout_s: int = 3600) -> dict:
    start = time.time()
    while time.time() - start < timeout_s:
        r = s.get(progress_url)
        if r.status_code == 429:
            time.sleep(5)
            continue
        if r.status_code != 200:
            return {"status": "error", "message": r.text}
        data = r.json()
        st = data.get("status")
        if st == "COMPLETE":
            alpha_id = data.get("alpha")
            if not alpha_id:
                return {"status": "error", "message": "缺少 alpha id"}
            a = s.get(f"https://api.worldquantbrain.com/alphas/{alpha_id}")
            if a.status_code != 200:
                return {"status": "error", "message": a.text}
            return {"status": "complete", "alpha": a.json()}
        if st == "ERROR":
            return {"status": "error", "message": "模拟错误"}
        time.sleep(5)
    return {"status": "timeout"}


def generate_templates() -> List[str]:
    wins = [20, 60, 120, 252]
    ranks = [5, 20]
    exprs: List[str] = []
    # 分组中性化(zscore(ts_mean(returns,W)), "sector")
    for w in wins:
        exprs.append(f'group_neutralize(zscore(ts_mean(returns, {w})), "sector")')
    # 分组中性化(zscore(ts_mean(returns,W1)-ts_mean(returns,W2)), "sector")
    for w1 in wins:
        for w2 in wins:
            if w2 > w1:
                exprs.append(
                    f'group_neutralize(zscore(ts_mean(returns, {w1}) - ts_mean(returns, {w2})), "sector")'
                )
    # 分组中性化(zscore(ts_rank(ts_mean(returns,W), R)), "sector")
    for w in wins:
        for r in ranks:
            exprs.append(
                f'group_neutralize(zscore(ts_rank(ts_mean(returns, {w}), {r})), "sector")'
            )
    # 分组中性化(zscore(rank(divide(revenue, assets)))), "sector")
    exprs.append('group_neutralize(zscore(rank(divide(revenue, assets))), "sector")')
    return exprs


def main():
    parser = argparse.ArgumentParser(description="Template+Grid Alpha Miner")
    parser.add_argument("--credentials", type=str, default="./credential.txt")
    parser.add_argument("--max", type=int, default=30, help="Max expressions to test")
    parser.add_argument(
        "--timeout", type=int, default=3600, help="Per-simulation timeout (s)"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    s = auth_session(args.credentials)
    exprs = generate_templates()[: args.max]
    results = []

    for i, expr in enumerate(exprs, 1):
        logger.info(f"测试中 {i}/{len(exprs)}: {expr}")
        sub = submit_sim(s, expr)
        if sub.get("status") != "success":
            logger.warning(f"提交错误: {sub.get('message')}")
            continue
        mon = monitor_sim(s, sub["progress_url"], timeout_s=args.timeout)
        if mon.get("status") == "complete":
            alpha = mon["alpha"]
            is_data = alpha.get("is", {})
            fitness = is_data.get("fitness")
            sharpe = is_data.get("sharpe")
            logger.info(f"完成. 适应度={fitness}, 夏普比率={sharpe}")
            # Append to hopeful if strong
            if fitness is not None and fitness > 1:
                try:
                    # Log into hopeful file
                    hopeful = []
                    if os.path.exists("hopeful_alphas.json"):
                        hopeful = json.load(open("hopeful_alphas.json", "r"))
                    hopeful.append(
                        {
                            "expression": expr,
                            "timestamp": int(time.time()),
                            "alpha_id": alpha.get("id", "unknown"),
                            "fitness": fitness,
                            "sharpe": sharpe,
                            "turnover": is_data.get("turnover"),
                            "returns": is_data.get("returns"),
                            "grade": alpha.get("grade", "UNKNOWN"),
                            "checks": is_data.get("checks", []),
                        }
                    )
                    json.dump(hopeful, open("hopeful_alphas.json", "w"), indent=2)
                except Exception as e:
                    logger.error(f"写入 hopeful_alphas.json 失败: {e}")
        elif mon.get("status") == "timeout":
            logger.warning("监控超时")
        else:
            logger.warning(f"模拟失败: {mon.get('message')}")
        results.append({"expression": expr, "result": mon})

    with open("template_grid_results.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info("模板+网格挖掘完成")


if __name__ == "__main__":
    main()
