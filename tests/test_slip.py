"""护套滑移联合裁决测试：与全枚举暴力解对照，并覆盖方向、并列、校验与不可行。"""

from __future__ import annotations

import itertools
import random
import time

import pytest

from app.solver import (
    InfeasibleError,
    ValidationErrors,
    invert_payload,
    validate,
)


def brute_slip(lengths, strain_min, strain_max, windows_raw, candidates, min_jump):
    """全枚举参考实现：在 (边界, 方向, 逐段应变) 上联合裁决。

    返回 (overall, per_direction)：
    - overall = ((M, S, xs, entry_idx), b, direction) 或 None
    - per_direction[(b, d)] = (M, S, xs) 该边界该方向的三级最优
    """
    n = len(lengths)
    overall = None
    per_direction: dict[tuple[int, int], tuple] = {}
    for xs in itertools.product(range(strain_min, strain_max + 1), repeat=n):
        if not all(
            lo <= sum(lengths[i] * xs[i] for i in range(s, e + 1)) <= hi
            for s, e, lo, hi in windows_raw
        ):
            continue
        for idx, b in enumerate(candidates):
            delta = xs[b] - xs[b - 1]
            if abs(delta) < min_jump:
                continue
            others = [xs[i] - xs[i - 1] for i in range(1, n) if i != b]
            key = (max(abs(d) for d in others), sum(abs(d) for d in others), tuple(xs))
            direction = 1 if delta > 0 else -1
            prev = per_direction.get((b, direction))
            if prev is None or key < prev:
                per_direction[(b, direction)] = key
            full_key = (key[0], key[1], key[2], idx)
            if overall is None or full_key < overall[0]:
                overall = (full_key, b, direction)
    return overall, per_direction


def make_payload(lengths, strain_min, strain_max, windows_raw, candidates, min_jump):
    return {
        "segment_lengths": lengths,
        "strain_bounds": {"min": strain_min, "max": strain_max},
        "windows": [
            {
                "start_segment": s + 1,
                "end_segment": e + 1,
                "min_elongation": lo,
                "max_elongation": hi,
            }
            for s, e, lo, hi in windows_raw
        ],
        "slip": {
            "candidate_boundaries": candidates,
            "min_jump": min_jump,
        },
    }


def pinned_payload(truth, candidates, min_jump):
    """用单段等式窗把每段应变钉成 truth（另加两个等式窗凑足 8 个）。"""
    n = len(truth)
    lengths = [1] * n
    windows = [
        {"start_segment": i, "end_segment": i,
         "min_elongation": truth[i - 1], "max_elongation": truth[i - 1]}
        for i in range(1, n + 1)
    ]
    total = sum(truth)
    windows.append(
        {"start_segment": 1, "end_segment": n,
         "min_elongation": total, "max_elongation": total}
    )
    windows.append(
        {"start_segment": 1, "end_segment": 2,
         "min_elongation": truth[0] + truth[1],
         "max_elongation": truth[0] + truth[1]}
    )
    return {
        "segment_lengths": lengths,
        "strain_bounds": {"min": -10, "max": 10},
        "windows": windows,
        "slip": {"candidate_boundaries": candidates, "min_jump": min_jump},
    }


@pytest.mark.parametrize("seed", range(10))
def test_slip_random_matches_bruteforce(seed):
    rng = random.Random(500 + seed)
    n = 6
    lengths = [rng.randint(1, 4) for _ in range(n)]
    strain_min, strain_max = -2, 2
    truth = [rng.randint(strain_min, strain_max) for _ in range(n)]
    pairs = [(s, e) for s in range(n) for e in range(s, n)]
    rng.shuffle(pairs)
    windows_raw = []
    for s, e in pairs[:10]:
        total = sum(lengths[i] * truth[i] for i in range(s, e + 1))
        slack = rng.randint(0, 2)
        windows_raw.append((s, e, total - slack, total + slack))
    nc = rng.randint(2, 4)
    candidates = sorted(rng.sample(range(1, n), nc))
    min_jump = rng.randint(1, 3)

    payload = make_payload(
        lengths, strain_min, strain_max, windows_raw, candidates, min_jump
    )
    overall, per_direction = brute_slip(
        lengths, strain_min, strain_max, windows_raw, candidates, min_jump
    )
    if overall is None:
        with pytest.raises(InfeasibleError):
            invert_payload(payload)
        return

    result = invert_payload(payload)
    (m, s_abs, xs, idx), b, direction = overall
    slip = result["slip"]
    assert slip["selected_boundary"] == b
    assert slip["selected_boundary_entry_order"] == idx
    assert result["strains"] == list(xs)
    assert slip["jump_direction"] == direction
    assert slip["signed_jump"] == xs[b] - xs[b - 1]
    assert abs(slip["signed_jump"]) >= min_jump
    assert result["objectives"]["max_adjacent_diff"] == m
    assert result["objectives"]["sum_adjacent_abs_diff"] == s_abs

    # 只有选中边界被排除在两级平滑指标之外。
    assert slip["excluded_from_smoothness"] == [b]
    assert sorted(slip["smoothness_boundaries"]) == [
        k for k in range(1, n) if k != b
    ]
    diffs = result["adjacent_diffs"]
    kept = [diffs[k - 1] for k in range(1, n) if k != b]
    assert result["objectives"]["max_adjacent_diff"] == max(abs(d) for d in kept)
    assert result["objectives"]["sum_adjacent_abs_diff"] == sum(abs(d) for d in kept)
    # 被排除边界上的突跳不受 M 限制（可能小于 M，也可能大于 M）。
    assert abs(diffs[b - 1]) >= min_jump

    # 候选与方向报告必须与全枚举一致。
    reports = {c["boundary"]: c for c in slip["candidates"]}
    assert sorted(reports) == sorted(candidates)
    for c0 in candidates:
        rep = reports[c0]
        assert rep["entry_order"] == candidates.index(c0)
        for d in (-1, 1):
            rdir = next(x for x in rep["directions"] if x["direction"] == d)
            expected_key = per_direction.get((c0, d))
            if expected_key is None:
                assert rdir["feasible"] is False
            else:
                assert rdir["feasible"] is True
                assert rdir["max_adjacent_diff"] == expected_key[0]
                assert rdir["sum_adjacent_abs_diff"] == expected_key[1]
        feasible_dirs = [d for d in (-1, 1) if (c0, d) in per_direction]
        assert rep["feasible"] is bool(feasible_dirs)
        if feasible_dirs:
            best_ms_here = min(
                (per_direction[(c0, d)][0], per_direction[(c0, d)][1])
                for d in feasible_dirs
            )
            expected_optimal_dirs = sorted(
                d
                for d in feasible_dirs
                if (
                    per_direction[(c0, d)][0],
                    per_direction[(c0, d)][1],
                )
                == best_ms_here
            )
            assert sorted(rep["optimal_directions"]) == expected_optimal_dirs


@pytest.mark.parametrize("seed", range(4))
def test_slip_random_n7_matches_bruteforce(seed):
    rng = random.Random(700 + seed)
    n = 7
    lengths = [rng.randint(1, 3) for _ in range(n)]
    strain_min, strain_max = -1, 1  # 3^7 = 2187
    truth = [rng.randint(strain_min, strain_max) for _ in range(n)]
    pairs = [(s, e) for s in range(n) for e in range(s, n)]
    rng.shuffle(pairs)
    windows_raw = []
    for s, e in pairs[:12]:
        total = sum(lengths[i] * truth[i] for i in range(s, e + 1))
        slack = rng.randint(0, 1)
        windows_raw.append((s, e, total - slack, total + slack))
    candidates = sorted(rng.sample(range(1, n), rng.randint(2, 5)))
    min_jump = rng.randint(1, 2)

    payload = make_payload(
        lengths, strain_min, strain_max, windows_raw, candidates, min_jump
    )
    overall, _ = brute_slip(
        lengths, strain_min, strain_max, windows_raw, candidates, min_jump
    )
    if overall is None:
        with pytest.raises(InfeasibleError):
            invert_payload(payload)
        return
    (m, s_abs, xs, idx), b, direction = overall
    result = invert_payload(payload)
    slip = result["slip"]
    assert slip["selected_boundary"] == b
    assert slip["selected_boundary_entry_order"] == idx
    assert result["strains"] == list(xs)
    assert slip["jump_direction"] == direction
    assert abs(slip["signed_jump"]) >= min_jump
    assert result["objectives"] == {
        "max_adjacent_diff": m,
        "sum_adjacent_abs_diff": s_abs,
    }


def test_slip_single_possible_boundary_and_direction():
    # 序列钉死为 [0,0,0,5,5,5]：只有边界 3 有大小 5 的上升突跳。
    payload = pinned_payload([0, 0, 0, 5, 5, 5], [1, 2, 3, 4], 5)
    result = invert_payload(payload)
    slip = result["slip"]
    assert slip["selected_boundary"] == 3
    assert slip["signed_jump"] == 5
    assert slip["jump_direction"] == 1
    assert result["objectives"] == {"max_adjacent_diff": 0, "sum_adjacent_abs_diff": 0}
    reports = {c["boundary"]: c for c in slip["candidates"]}
    assert reports[3]["feasible"] is True
    for b in (1, 2, 4):
        assert reports[b]["feasible"] is False
        assert all(d["feasible"] is False for d in reports[b]["directions"])


def test_slip_tie_broken_by_entry_order():
    # 序列 [0,5,5,5,5,0]：边界 1（+5）与边界 5（-5）三级指标完全并列，
    # 边界 3 因突跳为 0 不可行；完全并列时按候选录入顺序裁决。
    for order, expected in (([5, 1, 3], 5), ([1, 5, 3], 1), ([3, 5, 1], 5)):
        payload = pinned_payload([0, 5, 5, 5, 5, 0], order, 5)
        result = invert_payload(payload)
        slip = result["slip"]
        assert slip["selected_boundary"] == expected, order
        assert slip["selected_boundary_entry_order"] == order.index(expected)
        assert result["objectives"] == {"max_adjacent_diff": 5, "sum_adjacent_abs_diff": 5}


def test_slip_threshold_is_closed_interval():
    # |Δ| 恰好等于 min_jump 即可行（闭区间）。
    payload = pinned_payload([0, 0, 0, 3, 3, 3], [2, 3, 4], 3)
    result = invert_payload(payload)
    assert result["slip"]["selected_boundary"] == 3
    assert abs(result["slip"]["signed_jump"]) == 3


def test_slip_infeasible_when_jump_too_large():
    payload = pinned_payload([0, 0, 0, 5, 5, 5], [1, 3, 5], 6)
    with pytest.raises(InfeasibleError):
        invert_payload(payload)


def test_slip_infeasible_when_windows_conflict():
    payload = pinned_payload([0, 0, 0, 5, 5, 5], [1, 3, 5], 1)
    payload["windows"].append(
        {"start_segment": 1, "end_segment": 6,
         "min_elongation": 10**9, "max_elongation": 10**9}
    )
    with pytest.raises(InfeasibleError):
        invert_payload(payload)


def test_slip_validation_errors():
    base = pinned_payload([0, 0, 0, 5, 5, 5], [1, 3, 5], 5)

    def fields(payload):
        with pytest.raises(ValidationErrors) as exc:
            validate(payload)
        return {f["field"] for f in exc.value.fields}

    p = {**base, "slip": {"candidate_boundaries": [0, 6], "min_jump": 1}}
    fs = fields(p)
    assert "slip.candidate_boundaries[0]" in fs
    assert "slip.candidate_boundaries[1]" in fs

    p = {**base, "slip": {"candidate_boundaries": [3, 3], "min_jump": 1}}
    assert "slip.candidate_boundaries[1]" in fields(p)

    p = {**base, "slip": {"candidate_boundaries": [3], "min_jump": 1}}
    assert "slip.candidate_boundaries" in fields(p)

    # 6 个候选在允许数量 2..6 内，但 n=6 时第 6 个边界越界（上界 n-1=5）。
    p = {**base, "slip": {"candidate_boundaries": [1, 2, 3, 4, 5, 6], "min_jump": 1}}
    assert "slip.candidate_boundaries[5]" in fields(p)

    # 7 个候选则超出 2..6 的数量限制。
    p = {**base, "slip": {"candidate_boundaries": [1, 2, 3, 4, 5, 6, 7], "min_jump": 1}}
    assert "slip.candidate_boundaries" in fields(p)

    p = {**base, "slip": {"candidate_boundaries": [1, 3], "min_jump": 0}}
    assert "slip.min_jump" in fields(p)

    p = {**base, "slip": {"candidate_boundaries": [1, 3], "min_jump": -2}}
    assert "slip.min_jump" in fields(p)

    p = {**base, "slip": {"candidate_boundaries": [1, 3.0], "min_jump": 1}}
    assert "slip.candidate_boundaries[1]" in fields(p)

    p = {**base, "slip": {"candidate_boundaries": [1, 3], "min_jump": True}}
    assert "slip.min_jump" in fields(p)

    p = {**base, "slip": {"candidate_boundaries": [1, 3], "min_jump": 1, "bogus": 0}}
    assert any("slip.bogus" in f for f in fields(p))

    p = {**base, "slip": [1, 3]}
    assert "slip" in fields(p)

    p = {**base, "slip": {"min_jump": 1}}
    assert "slip.candidate_boundaries" in fields(p)

    p = {**base, "slip": {"candidate_boundaries": [1, 3]}}
    assert "slip.min_jump" in fields(p)


def test_slip_six_candidates_valid_at_n12():
    n = 12
    lengths = [1] * n
    windows = [
        {"start_segment": i, "end_segment": i, "min_elongation": 0, "max_elongation": 0}
        for i in range(1, n + 1)
    ][:7]
    windows += [
        {"start_segment": 1, "end_segment": n, "min_elongation": 0, "max_elongation": 0}
    ]
    payload = {
        "segment_lengths": lengths,
        "strain_bounds": {"min": -10, "max": 10},
        "windows": windows,
        "slip": {"candidate_boundaries": [1, 2, 3, 4, 5, 6], "min_jump": 1},
    }
    with pytest.raises(InfeasibleError):
        invert_payload(payload)  # 全零序列在任何边界都没有 >=1 的突跳


def test_no_slip_key_when_disabled():
    payload = pinned_payload([0, 0, 0, 5, 5, 5], [1, 3, 5], 5)
    del payload["slip"]
    result = invert_payload(payload)
    assert "slip" not in result
    assert "smoothness_boundaries" not in result
    # 旧行为：全部相邻差参与，突跳处主导一级指标。
    assert result["objectives"]["max_adjacent_diff"] == 5


def test_slip_max_size_performance():
    rng = random.Random(77)
    n = 12
    lengths = [rng.randint(1, 100) for _ in range(n)]
    truth = [rng.randint(-5000, 5000) for _ in range(n)]
    truth[7] = truth[6] + 400
    windows = []
    for s0 in range(n):
        e0 = rng.randint(s0, n - 1)
        total = sum(lengths[i] * truth[i] for i in range(s0, e0 + 1))
        windows.append(
            {"start_segment": s0 + 1, "end_segment": e0 + 1,
             "min_elongation": total - 100, "max_elongation": total + 100}
        )
    while len(windows) < 20:
        s0 = rng.randint(0, n - 1)
        e0 = rng.randint(s0, n - 1)
        total = sum(lengths[i] * truth[i] for i in range(s0, e0 + 1))
        windows.append(
            {"start_segment": s0 + 1, "end_segment": e0 + 1,
             "min_elongation": total - 50, "max_elongation": total + 50}
        )
    payload = {
        "segment_lengths": lengths,
        "strain_bounds": {"min": -100000, "max": 100000},
        "windows": windows,
        "slip": {"candidate_boundaries": [2, 4, 6, 8, 9, 11], "min_jump": 200},
    }
    start = time.monotonic()
    result = invert_payload(payload)
    assert time.monotonic() - start < 15.0
    slip = result["slip"]
    assert len(slip["candidates"]) == 6
    assert abs(slip["signed_jump"]) >= 200
    assert all(c["satisfied"] for c in result["window_checks"])
