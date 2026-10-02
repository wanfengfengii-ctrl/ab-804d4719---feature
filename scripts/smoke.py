"""对运行中的反演 API 做端到端冒烟（仅标准库）。

由 verify 一次性服务在 api 健康后执行。逐项回算每个观测窗的长度加权
应变和、两级平滑指标，全部由提交数据直接复核；任一不符即以非零退出。
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

BASE = os.environ.get("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")


def request(method: str, path: str, body: object | None = None) -> tuple[int, object]:
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        BASE + path,
        data=data,
        method=method,
        headers={"content-type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


def check(condition: bool, message: str) -> None:
    if not condition:
        print(f"  [FAIL] {message}")
        raise SystemExit(1)
    print(f"  [ok]   {message}")


def feasible_payload() -> dict:
    lengths = [10, 12, 11, 13, 10, 14]
    truth = [3, 5, 4, 2, 6, 1]
    spans = [(0, 5), (0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (0, 2)]
    windows = []
    for s, e in spans:
        total = sum(lengths[i] * truth[i] for i in range(s, e + 1))
        windows.append(
            {
                "start_segment": s + 1,
                "end_segment": e + 1,
                "min_elongation": total - 5,
                "max_elongation": total + 30,
            }
        )
    return {
        "segment_lengths": lengths,
        "strain_bounds": {"min": -50, "max": 50},
        "windows": windows,
    }


def feasible_payload_with_slip() -> dict:
    """真值 [0,0,0,5,5,5]：仅边界 3 存在 +5 的突跳，等式窗钉死。"""
    lengths = [10, 12, 11, 13, 10, 14]
    truth = [0, 0, 0, 5, 5, 5]
    spans = [(0, 5), (0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (0, 2)]
    windows = []
    for s, e in spans:
        total = sum(lengths[i] * truth[i] for i in range(s, e + 1))
        windows.append(
            {
                "start_segment": s + 1,
                "end_segment": e + 1,
                "min_elongation": total,
                "max_elongation": total,
            }
        )
    return {
        "segment_lengths": lengths,
        "strain_bounds": {"min": -50, "max": 50},
        "windows": windows,
        "slip": {"candidate_boundaries": [1, 3, 5], "min_jump": 5},
    }


def main() -> None:
    print(f"smoke against {BASE}")

    print("1) health")
    status, body = request("GET", "/health")
    check(status == 200, f"GET /health -> 200 (got {status})")
    check(body.get("status") == "ok", "health payload status == ok")

    print("2) successful inversion + 证据回算")
    payload = feasible_payload()
    status, body = request("POST", "/api/v1/invert", payload)
    check(status == 200, f"POST /api/v1/invert -> 200 (got {status}: {body})")
    check(body.get("code") == "OK", "response code == OK")
    result = body["result"]
    lengths = payload["segment_lengths"]
    strains = result["strains"]
    check(len(strains) == 6, "返回 6 段应变")
    check(
        all(payload["strain_bounds"]["min"] <= v <= payload["strain_bounds"]["max"]
            for v in strains),
        "每段应变均落入统一应变闭区间",
    )
    for check_item, win in zip(result["window_checks"], payload["windows"]):
        total = sum(
            lengths[i] * strains[i]
            for i in range(win["start_segment"] - 1, win["end_segment"])
        )
        check(
            check_item["weighted_strain_sum"] == total,
            f"窗[{win['start_segment']},{win['end_segment']}] 回算和 {total} 与响应一致",
        )
        check(
            win["min_elongation"] <= total <= win["max_elongation"],
            f"窗[{win['start_segment']},{win['end_segment']}] {total} ∈ 提交闭区间",
        )
    diffs = [strains[i] - strains[i - 1] for i in range(1, 6)]
    check(
        result["adjacent_diffs"] == diffs,
        "adjacent_diffs 可由应变序列直接复核",
    )
    check(
        result["objectives"]["max_adjacent_diff"] == max(abs(d) for d in diffs),
        "一级指标 max_adjacent_diff 可由相邻差复核",
    )
    check(
        result["objectives"]["sum_adjacent_abs_diff"] == sum(abs(d) for d in diffs),
        "二级指标 sum_adjacent_abs_diff 可由相邻差复核",
    )

    print("3) conflicting windows -> INFEASIBLE")
    bad = feasible_payload()
    bad["windows"][0] = {
        "start_segment": 1,
        "end_segment": 6,
        "min_elongation": 10**9,
        "max_elongation": 10**9,
    }
    status, body = request("POST", "/api/v1/invert", bad)
    check(status == 409, f"冲突观测 -> 409 (got {status})")
    check(body.get("code") == "INFEASIBLE", "错误码 == INFEASIBLE")

    print("4) invalid input -> 字段错误")
    bad = feasible_payload()
    bad["segment_lengths"] = [1, 2, 3]
    status, body = request("POST", "/api/v1/invert", bad)
    check(status == 422, f"非法输入 -> 422 (got {status})")
    check(body.get("code") == "INVALID_INPUT", "错误码 == INVALID_INPUT")
    check(bool(body.get("fields")), "返回明确 fields 列表")

    print("5) slip 护套滑移联合裁决 + 证据回算")
    slip_payload = feasible_payload_with_slip()
    status, body = request("POST", "/api/v1/invert", slip_payload)
    check(status == 200, f"滑移反演 -> 200 (got {status}: {body})")
    result = body["result"]
    slip = result["slip"]
    check(slip["selected_boundary"] == 3, "选中边界 == 3")
    check(slip["signed_jump"] == 5, f"带符号突跳 == 5 (got {slip['signed_jump']})")
    check(slip["jump_direction"] == 1, "突跳方向 == +1")
    check(abs(slip["signed_jump"]) >= slip_payload["slip"]["min_jump"], "突跳不低于阈值")
    check(
        slip["excluded_from_smoothness"] == [3],
        "只有选中边界 3 被排除在平滑指标之外",
    )
    check(
        slip["smoothness_boundaries"] == [1, 2, 4, 5],
        "其余相邻差全部参与平滑指标",
    )
    strains = result["strains"]
    diffs = [strains[i] - strains[i - 1] for i in range(1, 6)]
    kept = [diffs[k - 1] for k in (1, 2, 4, 5)]
    check(
        result["objectives"]["max_adjacent_diff"] == max(abs(d) for d in kept),
        "一级指标只由其余相邻差复核",
    )
    check(
        result["objectives"]["sum_adjacent_abs_diff"] == sum(abs(d) for d in kept),
        "二级指标只由其余相邻差复核",
    )
    reports = {c["boundary"]: c for c in slip["candidates"]}
    check(
        [reports[b]["feasible"] for b in (1, 3, 5)] == [False, True, False],
        "各候选可行性报告正确（仅边界 3 可行）",
    )
    for check_item, win in zip(result["window_checks"], slip_payload["windows"]):
        total = sum(
            slip_payload["segment_lengths"][i] * strains[i]
            for i in range(win["start_segment"] - 1, win["end_segment"])
        )
        check(
            check_item["weighted_strain_sum"] == total
            and win["min_elongation"] <= total <= win["max_elongation"],
            f"滑移植窗[{win['start_segment']},{win['end_segment']}] 回算 {total} 落在闭区间",
        )

    print("6) slip 无满足最小突跳的解释 -> 409")
    bad = json.loads(json.dumps(slip_payload))
    bad["slip"]["min_jump"] = 6
    status, body = request("POST", "/api/v1/invert", bad)
    check(status == 409, f"滑移不可行 -> 409 (got {status})")
    check(body.get("code") == "INFEASIBLE", "错误码 == INFEASIBLE")

    print("7) slip 候选越界/重复 -> 422 字段错误")
    for bad_bounds in ([0, 6], [3, 3]):
        bad = json.loads(json.dumps(slip_payload))
        bad["slip"]["candidate_boundaries"] = bad_bounds
        status, body = request("POST", "/api/v1/invert", bad)
        check(status == 422, f"候选 {bad_bounds} -> 422 (got {status})")
        check(
            any("candidate_boundaries" in f["field"] for f in body.get("fields", [])),
            f"候选 {bad_bounds} 报字段级错误",
        )

    print("ALL SMOKE CHECKS PASSED")


if __name__ == "__main__":
    main()
