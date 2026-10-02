"""滑移接头模式测试：全枚举暴力对照联合裁决，并覆盖校验/不可行/响应结构。"""

from __future__ import annotations

import itertools
import random
import time

import pytest

from app.solver import (
    InfeasibleError,
    ValidationErrors,
    invert_payload,
)


def make_windows(lengths, windows_raw):
    return [
        {
            "start_segment": s + 1,
            "end_segment": e + 1,
            "min_elongation": lo,
            "max_elongation": hi,
        }
        for s, e, lo, hi in windows_raw
    ]


def brute_force_slip(
    lengths, strain_min, strain_max, windows_raw, candidates_1based, min_jump
):
    """全枚举参考：枚举 x 与候选边界，按 (M', S', x, 候选顺序) 联合裁决。

    返回 (overall, per_candidate)：overall=(boundary1, m, s, xs)，
    per_candidate[order] = (m, s) 或 None。
    """
    n = len(lengths)

    def windows_ok(xs):
        for s, e, lo, hi in windows_raw:
            total = sum(lengths[i] * xs[i] for i in range(s, e + 1))
            if not (lo <= total <= hi):
                return False
        return True

    per_candidate: list[tuple[int, int] | None] = []
    best = None  # ((m, s, xs_tuple, order), boundary1, xs)
    for order, b1 in enumerate(candidates_1based):
        b = b1 - 1
        local_best = None
        for xs in itertools.product(range(strain_min, strain_max + 1), repeat=n):
            if abs(xs[b + 1] - xs[b]) < min_jump:
                continue
            if not windows_ok(xs):
                continue
            others = [
                abs(xs[i] - xs[i - 1]) for i in range(1, n) if i - 1 != b
            ]
            key = (max(others), sum(others), tuple(xs))
            if local_best is None or key < local_best[0]:
                local_best = (key, list(xs))
        if local_best is None:
            per_candidate.append(None)
            continue
        (m, s, xs_tuple), xs = local_best
        per_candidate.append((m, s))
        overall_key = (m, s, xs_tuple, order)
        if best is None or overall_key < best[0]:
            best = (overall_key, b1, xs)
    if best is None:
        return None, per_candidate
    (m, s, _, _), b1, xs = best
    return (b1, m, s, xs), per_candidate


def assert_slip_result(
    result, lengths, strain_min, strain_max, windows_raw, candidates, min_jump
):
    """对滑移成功结果做全部可复核断言。"""
    n = len(lengths)
    strains = result["strains"]
    assert len(strains) == n
    assert all(strain_min <= v <= strain_max for v in strains)

    # 观测窗逐条重算。
    for check, (s, e, lo, hi) in zip(result["window_checks"], windows_raw):
        total = sum(lengths[i] * strains[i] for i in range(s, e + 1))
        assert check["weighted_strain_sum"] == total
        assert lo <= total <= hi
        assert check["satisfied"] is True

    diffs = [strains[i] - strains[i - 1] for i in range(1, n)]
    assert result["adjacent_diffs"] == diffs

    slip = result["slip_joint"]
    selected = slip["selected_boundary"]
    assert selected in candidates
    # 只选择一个候选边界。
    assert len({selected}) == 1
    jump = slip["signed_jump"]
    assert jump == diffs[selected - 1]
    assert abs(jump) >= min_jump
    assert slip["min_jump"] == min_jump
    assert slip["jump_satisfied"] is True

    # 关键复核项：只有选中边界不参与平滑指标。
    participation = result["smooth_diff_participation"]
    assert [p["boundary"] for p in participation] == list(range(1, n))
    assert [p["adjacent_diff"] for p in participation] == diffs
    excluded = [p["boundary"] for p in participation if not p["participates_in_smoothness"]]
    included = [p["boundary"] for p in participation if p["participates_in_smoothness"]]
    assert excluded == [selected]
    assert included == [b for b in range(1, n) if b != selected]

    other_abs = [abs(d) for k, d in enumerate(diffs) if k != selected - 1]
    assert len(other_abs) == n - 2  # 其余相邻差全部参与
    assert result["objectives"]["max_other_adjacent_diff"] == max(other_abs)
    assert result["objectives"]["sum_other_adjacent_abs_diff"] == sum(other_abs)
    # 旧指标名不得出现在滑移响应中。
    assert "max_adjacent_diff" not in result["objectives"]

    # 候选报告：顺序、可行性与各候选自身最优指标。
    reports = slip["candidates"]
    assert [r["boundary"] for r in reports] == candidates
    return reports


@pytest.mark.parametrize("seed", range(18))
def test_random_slip_matches_bruteforce(seed):
    rng = random.Random(1000 + seed)
    n = 6
    lengths = [rng.randint(1, 4) for _ in range(n)]
    strain_min, strain_max = -2, 2
    truth = [rng.randint(strain_min, strain_max) for _ in range(n)]
    windows_raw = []
    seen = set()
    while len(windows_raw) < 10:
        s = rng.randint(0, n - 1)
        e = rng.randint(s, n - 1)
        if (s, e) in seen:
            continue
        seen.add((s, e))
        total = sum(lengths[i] * truth[i] for i in range(s, e + 1))
        slack = rng.randint(0, 2)
        windows_raw.append((s, e, total - slack, total + slack))

    k = rng.randint(2, 4)
    candidates = sorted(rng.sample(range(1, n), k))
    rng.shuffle(candidates)  # 打乱录入顺序以考验顺序裁决
    min_jump = rng.choice([1, 2, 3])

    payload = {
        "segment_lengths": lengths,
        "strain_bounds": {"min": strain_min, "max": strain_max},
        "windows": make_windows(lengths, windows_raw),
        "slip_joint": {
            "candidate_boundaries": candidates,
            "min_jump": min_jump,
        },
    }
    expected, per_candidate = brute_force_slip(
        lengths, strain_min, strain_max, windows_raw, candidates, min_jump
    )
    if expected is None:
        with pytest.raises(InfeasibleError):
            invert_payload(payload)
        return

    exp_boundary, exp_m, exp_s, exp_xs = expected
    result = invert_payload(payload)
    reports = assert_slip_result(
        result, lengths, strain_min, strain_max, windows_raw, candidates, min_jump
    )
    assert result["slip_joint"]["selected_boundary"] == exp_boundary
    assert result["strains"] == exp_xs
    assert result["objectives"]["max_other_adjacent_diff"] == exp_m
    assert result["objectives"]["sum_other_adjacent_abs_diff"] == exp_s
    for report, cand_best in zip(reports, per_candidate):
        if cand_best is None:
            assert report["feasible"] is False
            assert "objectives" not in report
        else:
            assert report["feasible"] is True
            assert report["objectives"]["max_other_adjacent_diff"] == cand_best[0]
            assert report["objectives"]["sum_other_adjacent_abs_diff"] == cand_best[1]
    # 顶层指标必须与选中候选的报告一致。
    selected_report = next(
        r for r in reports if r["boundary"] == exp_boundary
    )
    assert selected_report["objectives"] == {
        "max_other_adjacent_diff": exp_m,
        "sum_other_adjacent_abs_diff": exp_s,
    }


def test_pinned_sequence_selects_only_feasible_boundary():
    """单段窗零松驰钉死序列：仅真实跳变边界可行，且其方向带符号。"""
    lengths = [10, 12, 11, 13, 10, 14]
    truth = [3, 5, 4, 2, 6, 1]  # diffs = [2,-1,-2,4,-5]
    windows_raw = [(i, i, lengths[i] * truth[i], lengths[i] * truth[i])
                   for i in range(6)]
    total02 = sum(lengths[i] * truth[i] for i in range(0, 3))
    windows_raw.append((0, 2, total02, total02))
    windows_raw.append((0, 5, sum(lengths[i] * truth[i] for i in range(6)),
                        sum(lengths[i] * truth[i] for i in range(6))))
    candidates = [2, 5]
    payload = {
        "segment_lengths": lengths,
        "strain_bounds": {"min": -50, "max": 50},
        "windows": make_windows(lengths, windows_raw),
        "slip_joint": {"candidate_boundaries": candidates, "min_jump": 4},
    }
    result = invert_payload(payload)
    reports = assert_slip_result(
        result, lengths, -50, 50, windows_raw, candidates, 4
    )
    slip = result["slip_joint"]
    assert slip["selected_boundary"] == 5
    assert slip["signed_jump"] == -5  # 方向联合裁决：x6-x5 = 1-6
    by_boundary = {r["boundary"]: r for r in reports}
    assert by_boundary[2]["feasible"] is False
    assert by_boundary[5]["feasible"] is True
    # 其余差 abs = [2,1,2,4]
    assert result["objectives"] == {
        "max_other_adjacent_diff": 4,
        "sum_other_adjacent_abs_diff": 9,
    }


def test_exact_tie_broken_by_candidate_input_order():
    """两个候选的 (M',S',x) 完全并列时，取录入顺序靠前者。"""
    n = 6
    lengths = [1] * n
    truth = [0, 3, 0, 3, 0, 0]  # diffs = [3,-3,3,-3,0]
    windows_raw = [(i, i, truth[i], truth[i]) for i in range(n)]
    total = sum(truth)
    windows_raw.append((0, n - 1, total, total))
    windows_raw.append((0, 2, sum(truth[0:3]), sum(truth[0:3])))

    def run(order):
        payload = {
            "segment_lengths": lengths,
            "strain_bounds": {"min": -5, "max": 5},
            "windows": make_windows(lengths, windows_raw),
            "slip_joint": {"candidate_boundaries": order, "min_jump": 2},
        }
        return invert_payload(payload)

    r1 = run([1, 3])
    assert r1["slip_joint"]["selected_boundary"] == 1
    r2 = run([3, 1])
    assert r2["slip_joint"]["selected_boundary"] == 3
    # 序列与指标完全一致，仅边界按录入顺序裁决。
    assert r1["strains"] == r2["strains"] == truth
    assert r1["objectives"] == r2["objectives"] == {
        "max_other_adjacent_diff": 3,
        "sum_other_adjacent_abs_diff": 9,
    }


def test_jump_direction_part_of_joint_decision():
    """阈值可达但真值方向被窗口禁止时，应选出反方向可行的候选解释。"""
    n = 6
    lengths = [1] * n
    # 钉死 x0=0；若要求边界1正向跳变（x1>=3）则与全长窗 0 冲突，
    # 必须取负向（x1<=-3），并令其余段补偿回 0。
    truth = [0, -3, 3, 0, 0, 0]
    windows_raw = [(i, i, lengths[i] * truth[i], lengths[i] * truth[i])
                   for i in range(n)]
    windows_raw.append((0, n - 1, 0, 0))
    windows_raw.append((0, 0, 0, 0))
    candidates = [1, 4]
    payload = {
        "segment_lengths": lengths,
        "strain_bounds": {"min": -5, "max": 5},
        "windows": make_windows(lengths, windows_raw),
        "slip_joint": {"candidate_boundaries": candidates, "min_jump": 3},
    }
    result = invert_payload(payload)
    assert result["strains"] == truth
    slip = result["slip_joint"]
    assert slip["selected_boundary"] == 1
    assert slip["signed_jump"] == -3
    assert_slip_result(
        result, lengths, -5, 5, windows_raw, candidates, 3
    )


def test_all_candidates_infeasible_raises():
    """合法配置但阈值超过任何可达突跳 -> 不可行。"""
    n = 6
    lengths = [3, 1, 4, 1, 5, 9]
    truth = [1, 2, 1, 2, 1, 2]  # 最大相邻差 1
    windows_raw = [(i, i, lengths[i] * truth[i], lengths[i] * truth[i])
                   for i in range(n)]
    total = sum(lengths[i] * truth[i] for i in range(n))
    windows_raw.append((0, n - 1, total, total))
    windows_raw.append((0, 2, sum(lengths[i] * truth[i] for i in range(3)),
                        sum(lengths[i] * truth[i] for i in range(3))))
    payload = {
        "segment_lengths": lengths,
        "strain_bounds": {"min": -10, "max": 10},
        "windows": make_windows(lengths, windows_raw),
        "slip_joint": {"candidate_boundaries": [1, 3, 5], "min_jump": 4},
    }
    with pytest.raises(InfeasibleError):
        invert_payload(payload)


def test_slip_validation_errors():
    base = {
        "segment_lengths": [1] * 6,
        "strain_bounds": {"min": -5, "max": 5},
        "windows": make_windows(
            [1] * 6,
            [(i, i, 0, 0) for i in range(6)]
            + [(0, 5, 0, 0), (0, 2, 0, 0)],
        ),
    }

    def fields_for(slip):
        payload = dict(base)
        payload["slip_joint"] = slip
        with pytest.raises(ValidationErrors) as exc:
            invert_payload(payload)
        return {e["field"] for e in exc.value.fields}

    # 越界：0 与 n（n=6 时合法边界仅 1..5）
    fs = fields_for({"candidate_boundaries": [0, 3], "min_jump": 1})
    assert any("candidate_boundaries[0]" in f for f in fs)
    fs = fields_for({"candidate_boundaries": [1, 6], "min_jump": 1})
    assert any("candidate_boundaries[1]" in f for f in fs)
    # 重复
    fs = fields_for({"candidate_boundaries": [2, 2], "min_jump": 1})
    assert any("candidate_boundaries" in f for f in fs)
    # 数量：1 个（少于 2）、7 个（多于 6）
    fs = fields_for({"candidate_boundaries": [1], "min_jump": 1})
    assert any("candidate_boundaries" in f for f in fs)
    fs = fields_for({"candidate_boundaries": [1, 2, 3, 4, 5, 1, 2], "min_jump": 1})
    assert any("candidate_boundaries" in f for f in fs)
    # min_jump 非正整数 / 浮点 / 布尔
    assert any("min_jump" in f for f in fields_for(
        {"candidate_boundaries": [1, 2], "min_jump": 0}))
    assert any("min_jump" in f for f in fields_for(
        {"candidate_boundaries": [1, 2], "min_jump": -3}))
    assert any("min_jump" in f for f in fields_for(
        {"candidate_boundaries": [1, 2], "min_jump": 1.5}))
    assert any("min_jump" in f for f in fields_for(
        {"candidate_boundaries": [1, 2], "min_jump": True}))
    # 候选元素非整数 / 缺字段 / 未知字段
    assert any("candidate_boundaries" in f for f in fields_for(
        {"candidate_boundaries": [1, "2"], "min_jump": 1}))
    assert any("candidate_boundaries" in f for f in fields_for(
        {"min_jump": 1}))
    fs = fields_for({"candidate_boundaries": [1, 2]})
    assert any("min_jump" in f for f in fs)
    fs = fields_for({"candidate_boundaries": [1, 2], "min_jump": 1, "bogus": 0})
    assert any("slip_joint.bogus" in f for f in fs)
    # slip_joint 本身类型错误
    payload = dict(base)
    payload["slip_joint"] = [1, 2]
    with pytest.raises(ValidationErrors) as exc:
        invert_payload(payload)
    assert any(e["field"] == "slip_joint" for e in exc.value.fields)


def test_duplicate_boundary_message_explicit():
    payload = {
        "segment_lengths": [1] * 6,
        "strain_bounds": {"min": -5, "max": 5},
        "windows": make_windows(
            [1] * 6,
            [(i, i, 0, 0) for i in range(6)]
            + [(0, 5, 0, 0), (0, 2, 0, 0)],
        ),
        "slip_joint": {"candidate_boundaries": [3, 3], "min_jump": 2},
    }
    with pytest.raises(ValidationErrors) as exc:
        invert_payload(payload)
    messages = [e["message"] for e in exc.value.fields]
    assert any("duplicate" in m for m in messages)


def test_slip_deterministic():
    rng = random.Random(31337)
    n = 7
    lengths = [rng.randint(1, 4) for _ in range(n)]
    truth = [rng.randint(-3, 3) for _ in range(n)]
    windows_raw = []
    pairs = [(s, e) for s in range(n) for e in range(s, n)]
    rng.shuffle(pairs)
    for s, e in pairs[:10]:
        total = sum(lengths[i] * truth[i] for i in range(s, e + 1))
        windows_raw.append((s, e, total - 2, total + 2))
    payload = {
        "segment_lengths": lengths,
        "strain_bounds": {"min": -6, "max": 6},
        "windows": make_windows(lengths, windows_raw),
        "slip_joint": {"candidate_boundaries": [2, 4, 6], "min_jump": 3},
    }
    r1 = invert_payload(payload)
    r2 = invert_payload(payload)
    assert r1 == r2


def test_slip_large_integers_preserved():
    big = 10**30
    n = 6
    lengths = [big + i for i in range(n)]
    jump_val = 10**10 + big % 7
    truth = [0, jump_val, 1, -1, 0, 0]  # 仅边界1含大突跳，其余相邻差均 <=2
    windows_raw = [(i, i, lengths[i] * truth[i], lengths[i] * truth[i])
                   for i in range(n)]
    total = sum(lengths[i] * truth[i] for i in range(n))
    windows_raw.append((0, n - 1, total, total))
    windows_raw.append((0, 2, sum(lengths[i] * truth[i] for i in range(3)),
                        sum(lengths[i] * truth[i] for i in range(3))))
    payload = {
        "segment_lengths": lengths,
        "strain_bounds": {"min": -10**12, "max": 10**12},
        "windows": make_windows(lengths, windows_raw),
        "slip_joint": {"candidate_boundaries": [1, 3, 5], "min_jump": 10**9},
    }
    result = invert_payload(payload)
    assert result["strains"] == truth
    jump = result["slip_joint"]["signed_jump"]
    assert jump == jump_val and abs(jump) >= 10**9
    for check in result["window_checks"]:
        assert check["satisfied"] is True


def test_slip_max_size_performance():
    rng = random.Random(4242)
    n = 12
    lengths = [rng.randint(1, 100) for _ in range(n)]
    truth = [rng.randint(-5000, 5000) for _ in range(n)]
    # 在随机一处制造显著突跳。
    jump_at = rng.randint(0, n - 2)
    truth[jump_at + 1] = truth[jump_at] + rng.choice([-1, 1]) * 20000
    truth[jump_at + 1] = max(-100000, min(100000, truth[jump_at + 1]))
    windows = []
    starts = list(range(n))
    rng.shuffle(starts)
    for s0 in starts:
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
    candidates = sorted({jump_at + 1, rng.randint(1, n - 1), rng.randint(1, n - 1)})
    while len(candidates) < 2:
        c = rng.randint(1, n - 1)
        if c not in candidates:
            candidates.append(c)
    candidates = candidates[:6]
    payload = {
        "segment_lengths": lengths,
        "strain_bounds": {"min": -100000, "max": 100000},
        "windows": windows,
        "slip_joint": {"candidate_boundaries": candidates, "min_jump": 10000},
    }
    start = time.monotonic()
    result = invert_payload(payload)
    assert time.monotonic() - start < 15.0
    assert all(c["satisfied"] for c in result["window_checks"])
    assert abs(result["slip_joint"]["signed_jump"]) >= 10000
