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


def slip_feasible_payload() -> tuple[dict, list[int]]:
    """单段窗零松驰钉死序列：diffs=[2,-1,-2,4,-5]。

    候选 [2,5] 且 min_jump=4 时，仅边界 5（|差|=5）可行，选中后其余差
    绝对值为 [2,1,2,4]，供冒烟逐项复核。
    """
    lengths = [10, 12, 11, 13, 10, 14]
    truth = [3, 5, 4, 2, 6, 1]
    spans = [(i, i) for i in range(6)] + [(0, 2), (0, 5)]
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
    payload = {
        "segment_lengths": lengths,
        "strain_bounds": {"min": -50, "max": 50},
        "windows": windows,
        "slip_joint": {"candidate_boundaries": [2, 5], "min_jump": 4},
    }
    return payload, truth


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

    print("5) 滑移接头模式：成功定位 + 仅选中边界排除平滑指标")
    slip_payload, slip_truth = slip_feasible_payload()
    status, body = request("POST", "/api/v1/invert", slip_payload)
    check(status == 200, f"滑移请求 -> 200 (got {status}: {body})")
    result = body["result"]
    slip = result["slip_joint"]
    selected = slip["selected_boundary"]
    check(selected in (2, 5), f"选中边界为候选之一（实际 {selected}）")
    strains = result["strains"]
    diffs = [strains[i] - strains[i - 1] for i in range(1, 6)]
    check(
        slip["signed_jump"] == diffs[selected - 1],
        "带符号突跳可由相邻差直接复核",
    )
    check(
        abs(slip["signed_jump"]) >= slip_payload["slip_joint"]["min_jump"],
        "带符号突跳绝对值不低于 min_jump",
    )
    reports = slip["candidates"]
    check(
        [r["boundary"] for r in reports]
        == slip_payload["slip_joint"]["candidate_boundaries"],
        "候选报告按录入顺序给出且包含每个候选",
    )
    check(
        all(("objectives" in r) == r["feasible"] for r in reports),
        "每个候选给出可行性，且仅可行候选附最优指标",
    )
    # 检修组直接复核：只有选中边界被排除在平滑指标之外。
    participation = result["smooth_diff_participation"]
    excluded = [p["boundary"] for p in participation if not p["participates_in_smoothness"]]
    check(excluded == [selected], f"仅选中边界 {selected} 不参与平滑指标（实际排除 {excluded}）")
    check(
        all(
            p["adjacent_diff"] == diffs[p["boundary"] - 1] for p in participation
        ),
        "参与性清单的相邻差可逐行复核",
    )
    other_abs = [abs(d) for k, d in enumerate(diffs) if k != selected - 1]
    check(
        len(other_abs) == 4,
        "其余 4 个相邻差全部参与（n=6，仅排除 1 个）",
    )
    check(
        result["objectives"]["max_other_adjacent_diff"] == max(other_abs),
        "滑移一级指标仅由其余相邻差复核",
    )
    check(
        result["objectives"]["sum_other_adjacent_abs_diff"] == sum(other_abs),
        "滑移二级指标仅由其余相邻差复核",
    )
    for check_item, win in zip(result["window_checks"], slip_payload["windows"]):
        total = sum(
            slip_payload["segment_lengths"][i] * strains[i]
            for i in range(win["start_segment"] - 1, win["end_segment"])
        )
        check(
            win["min_elongation"] <= total <= win["max_elongation"],
            f"滑移解窗[{win['start_segment']},{win['end_segment']}] 仍满足闭区间",
        )

    print("6) 滑移候选越界/重复 -> 字段错误")
    bad = slip_feasible_payload()[0]
    bad["slip_joint"] = {"candidate_boundaries": [0, 6], "min_jump": 1}
    status, body = request("POST", "/api/v1/invert", bad)
    check(status == 422, f"候选越界 -> 422 (got {status})")
    check(
        any("candidate_boundaries" in f["field"] for f in body.get("fields", [])),
        "越界字段被逐条指出",
    )
    bad = slip_feasible_payload()[0]
    bad["slip_joint"] = {"candidate_boundaries": [3, 3], "min_jump": 1}
    status, body = request("POST", "/api/v1/invert", bad)
    check(status == 422, f"候选重复 -> 422 (got {status})")
    check(
        any("duplicate" in f["message"] for f in body.get("fields", [])),
        "重复候选明确报错",
    )

    print("7) 滑移合法配置但所有候选不可行 -> INFEASIBLE")
    bad = slip_feasible_payload()[0]
    bad["slip_joint"]["min_jump"] = 10**6
    status, body = request("POST", "/api/v1/invert", bad)
    check(status == 409, f"全部候选不可行 -> 409 (got {status})")
    check(body.get("code") == "INFEASIBLE", "错误码 == INFEASIBLE")

    print("ALL SMOKE CHECKS PASSED")


if __name__ == "__main__":
    main()
