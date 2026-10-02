"""海缆连续缆段整数微应变联合反演核心（仅使用 Python 标准库）。

输入
----
- 6..12 段按顺序排列的缆段长度（正整数）
- 统一应变闭区间 [strain_min, strain_max]（整数微应变）
- 8..20 个观测窗，每窗给出连续起止段（1 基，含端点）与累计伸长量闭区间

模型（所有比较与运算均为 Python 任意精度精确整数）
--------------------------------------------------
待求逐段应变为 x_0..x_{n-1}（整数微应变）。

观测窗 [s,e] 约束：Σ_{i=s..e} L_i·x_i ∈ [lo, hi]。
每段还须满足 strain_min ≤ x_i ≤ strain_max。

引入前缀和 P_0=0, P_i = Σ_{j<i} L_j·x_j，则
    P_{e+1} - P_s ∈ [lo, hi]
    P_{i+1} - P_i ∈ [L_i·strain_min, L_i·strain_max]
全部是 P 上的**差分约束**，其最长路闭包（Floyd-Warshall）：
- 出现正环 ⇒ 整体不可行（可靠的充分必要判定，针对松弛后的实值域；
  整数可行性仍以下层整数搜索为准）；
- 给出每个 P_i 的精确上下界，进而导出每段 x_i 的紧整数域：
      x_i ≥ ceil( D[i][i+1] / L_i )
      x_i ≤ floor( -D[i+1][i] / L_i )

优选准则（字典序三级，逐级不可放宽）
1. 最小化相邻段最大应变差 M = max_k |x_k - x_{k-1}|
   —— 在 x 空间就是差分约束 x_k - x_{k-1} ∈ [-M, M]，同样并入最长路闭包。
2. 在 1 的最优解中最小化 S = Σ_k |x_k - x_{k-1}|
   —— 辅助变量 e_k ≥ |Δ_k|，以 Σ e_k ≤ S 的线性松弛精确等价判定。
3. 在 1、2 的最优解中取应变序列 (x_0,...,x_{n-1}) 字典序最小者。

求解：最长路闭包缩域 + 通用整数界传播 + MRV/折半回溯；
最小 M、最小 S 及字典序各值均以「可行性关于阈值单调」二分得到。

护套滑移模式（可选 slip 配置）
------------------------------
给定 2..6 个候选段间边界与正整数最小突跳 T，恰选一个边界 b：
- 突跳约束 |x_b - x_{b-1}| >= T 放松为两个方向（升/降）分别判定；
- 边界 b 不参与两级平滑指标（不并入 |Δ|<=M 闭包、不计入 Σ|Δ|），
  其余 n-2 个相邻差全部参与；
- 对每个 (边界, 方向) 二分钉死 (M, S)，再仅在达到全局最优 (M, S) 的
  方向间以「严格小于当前最优」的 cutoff 逐段钉死字典序；
- 全局依次最小化 (M, S, 序列字典序)，完全并列按候选录入顺序。
所有候选方向均不可行时抛 InfeasibleError。
"""

from __future__ import annotations

from dataclasses import dataclass

MIN_SEGMENTS = 6
MAX_SEGMENTS = 12
MIN_WINDOWS = 8
MAX_WINDOWS = 20
MIN_SLIP_CANDIDATES = 2
MAX_SLIP_CANDIDATES = 6


class ValidationErrors(ValueError):
    """输入字段错误（可一次性包含多个字段问题）。"""

    def __init__(self, fields: list[dict[str, str]]):
        self.fields = fields
        super().__init__("; ".join(f"{f['field']}: {f['message']}" for f in fields))


class InfeasibleError(ValueError):
    """全部输入合法，但观测窗彼此冲突（含统一应变界），无可行解。"""


@dataclass(frozen=True)
class Window:
    start: int  # 0 基，含端点
    end: int  # 0 基，含端点
    lo: int  # 累计伸长量（长度加权应变和）闭区间下端
    hi: int  # 闭区间上端


# --------------------------------------------------------------------------- #
# 输入校验
# --------------------------------------------------------------------------- #
def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _err(fields: list[dict[str, str]], field: str, message: str) -> None:
    fields.append({"field": field, "message": message})


def _validate_lengths(payload: dict, fields: list[dict[str, str]]) -> list[int] | None:
    raw = payload.get("segment_lengths")
    if raw is None:
        _err(fields, "segment_lengths", "field is required")
        return None
    if not isinstance(raw, list):
        _err(fields, "segment_lengths", "must be an array of positive integers")
        return None
    lengths: list[int] = []
    for i, item in enumerate(raw):
        if not _is_int(item) or item < 1:
            _err(fields, f"segment_lengths[{i}]", "must be a positive integer")
        else:
            lengths.append(item)
    if not (MIN_SEGMENTS <= len(raw) <= MAX_SEGMENTS):
        _err(
            fields,
            "segment_lengths",
            f"must contain between {MIN_SEGMENTS} and {MAX_SEGMENTS} segments"
            f" (got {len(raw)})",
        )
    return lengths if len(lengths) == len(raw) else None


def _validate_strain_bounds(
    payload: dict, fields: list[dict[str, str]]
) -> tuple[int, int] | None:
    raw = payload.get("strain_bounds")
    if raw is None:
        _err(fields, "strain_bounds", "field is required")
        return None
    if not isinstance(raw, dict):
        _err(fields, "strain_bounds", 'must be an object {"min": int, "max": int}')
        return None
    for key in raw:
        if key not in ("min", "max"):
            _err(fields, f"strain_bounds.{key}", "unknown field")
    lo = raw.get("min")
    hi = raw.get("max")
    ok = True
    if not _is_int(lo):
        _err(fields, "strain_bounds.min", "must be an integer")
        ok = False
    if not _is_int(hi):
        _err(fields, "strain_bounds.max", "must be an integer")
        ok = False
    if ok and lo > hi:
        _err(fields, "strain_bounds", "min must be less than or equal to max")
        return None
    return (lo, hi) if ok else None


def _validate_windows(
    payload: dict, n: int | None, fields: list[dict[str, str]]
) ->list[Window] | None:
    raw = payload.get("windows")
    if raw is None:
        _err(fields, "windows", "field is required")
        return None
    if not isinstance(raw, list):
        _err(fields, "windows", "must be an array of observation windows")
        return None
    if not (MIN_WINDOWS <= len(raw) <= MAX_WINDOWS):
        _err(
            fields,
            "windows",
            f"must contain between {MIN_WINDOWS} and {MAX_WINDOWS} windows"
            f" (got {len(raw)})",
        )
    windows: list[Window] = []
    all_ok = True
    for i, item in enumerate(raw):
        prefix = f"windows[{i}]"
        if not isinstance(item, dict):
            _err(fields, prefix, "must be an object")
            all_ok = False
            continue
        ok = True
        for key in ("start_segment", "end_segment", "min_elongation", "max_elongation"):
            if key not in item:
                _err(fields, f"{prefix}.{key}", "field is required")
                ok = False
            elif not _is_int(item[key]):
                _err(fields, f"{prefix}.{key}", "must be an integer")
                ok = False
        for key in item:
            if key not in ("start_segment", "end_segment", "min_elongation", "max_elongation"):
                _err(fields, f"{prefix}.{key}", "unknown field")
                ok = False
        if not ok:
            all_ok = False
            continue
        s = item["start_segment"]
        e = item["end_segment"]
        lo = item["min_elongation"]
        hi = item["max_elongation"]
        if n is not None:
            if not (1 <= s <= n):
                _err(fields, f"{prefix}.start_segment", f"must be between 1 and {n}")
                ok = False
            if not (1 <= e <= n):
                _err(fields, f"{prefix}.end_segment", f"must be between 1 and {n}")
                ok = False
            if ok and s > e:
                _err(fields, f"{prefix}.start_segment", "must be <= end_segment")
                ok = False
        if lo > hi:
            _err(
                fields,
                f"{prefix}.min_elongation",
                "must be less than or equal to max_elongation",
            )
            ok = False
        if ok and n is not None:
            windows.append(Window(start=s - 1, end=e - 1, lo=lo, hi=hi))
        else:
            all_ok = False
    return windows if all_ok else None


_ALLOWED_TOP_LEVEL = {
    "segment_lengths",
    "strain_bounds",
    "windows",
    "slip",
}


def _validate_slip(
    payload: dict, n: int | None, fields: list[dict[str, str]]
) -> tuple[list[int], int] | None:
    """校验可选的护套滑移配置；缺省返回 None（完全保持旧行为）。

    返回 (候选边界列表[k：x_k 与 x_{k-1} 之间，即第 k 段与第 k+1 段之间，1 基],
    最小突跳正整数)。
    越界、重复、数量越界或阈值非正整数均记为字段错误。
    """
    raw = payload.get("slip")
    if raw is None:
        return None
    if not isinstance(raw, dict):
        _err(fields, "slip", 'must be an object {"candidate_boundaries": [int], "min_jump": int}')
        return None
    for key in raw:
        if key not in ("candidate_boundaries", "min_jump"):
            _err(fields, f"slip.{key}", "unknown field")
    raw_bounds = raw.get("candidate_boundaries")
    raw_jump = raw.get("min_jump")
    if raw_bounds is None:
        _err(fields, "slip.candidate_boundaries", "field is required")
    if raw_jump is None:
        _err(fields, "slip.min_jump", "field is required")
    boundaries: list[int] = []
    bounds_ok = True
    if raw_bounds is not None:
        if not isinstance(raw_bounds, list):
            _err(
                fields,
                "slip.candidate_boundaries",
                "must be an array of 2..6 distinct segment boundaries (1-based gap index)",
            )
            bounds_ok = False
        else:
            if not (MIN_SLIP_CANDIDATES <= len(raw_bounds) <= MAX_SLIP_CANDIDATES):
                _err(
                    fields,
                    "slip.candidate_boundaries",
                    f"must contain between {MIN_SLIP_CANDIDATES} and {MAX_SLIP_CANDIDATES}"
                    f" distinct boundaries (got {len(raw_bounds)})",
                )
                bounds_ok = False
            seen: set[int] = set()
            for i, item in enumerate(raw_bounds):
                prefix = f"slip.candidate_boundaries[{i}]"
                if not _is_int(item):
                    _err(fields, prefix, "must be an integer")
                    bounds_ok = False
                    continue
                if n is not None and not (1 <= item <= n - 1):
                    _err(fields, prefix, f"must be a segment boundary between 1 and {n - 1}")
                    bounds_ok = False
                    continue
                if item in seen:
                    _err(fields, prefix, "candidate boundary is duplicated")
                    bounds_ok = False
                    continue
                seen.add(item)
                boundaries.append(item)
    jump_ok = True
    min_jump = 0
    if raw_jump is not None:
        if not _is_int(raw_jump) or raw_jump < 1:
            _err(fields, "slip.min_jump", "must be a positive integer")
            jump_ok = False
        else:
            min_jump = raw_jump
    if not (bounds_ok and jump_ok):
        return None
    return boundaries, min_jump


def validate(
    payload: object,
) -> tuple[list[int], int, int, list[Window], tuple[list[int], int] | None]:
    """校验并归一化输入；非法时抛 ValidationErrors。"""
    fields: list[dict[str, str]] = []
    if not isinstance(payload, dict):
        raise ValidationErrors(
            [{"field": ".", "message": "request body must be a JSON object"}]
        )
    for key in payload:
        if key not in _ALLOWED_TOP_LEVEL:
            _err(fields, key, "unknown field")
    lengths = _validate_lengths(payload, fields)
    bounds = _validate_strain_bounds(payload, fields)
    n = len(lengths) if lengths is not None else None
    windows = _validate_windows(payload, n, fields)
    slip = _validate_slip(payload, n, fields)
    if fields:
        raise ValidationErrors(fields)
    assert lengths is not None and bounds is not None and windows is not None
    return lengths, bounds[0], bounds[1], windows, slip


# --------------------------------------------------------------------------- #
# 差分约束最长路闭包
# --------------------------------------------------------------------------- #
def _longest_path_closure(
    node_count: int, edges: list[tuple[int, int, int]]
) -> list[list[int | None]] | None:
    """约束 P_v >= P_u + w 的最长路闭包；存在正环返回 None。

    D[i][j] = P_j - P_i 的最强（最大）下界；None 表示尚无路径（-∞）。
    D[i][i] > 0 即正环，矛盾。以 None 为哨兵可兼容任意精度输入。
    """
    d: list[list[int | None]] = [[None] * node_count for _ in range(node_count)]
    for i in range(node_count):
        d[i][i] = 0
    for u, v, w in edges:
        if d[u][v] is None or w > d[u][v]:
            d[u][v] = w
    for k in range(node_count):
        dk = d[k]
        for i in range(node_count):
            dik = d[i][k]
            if dik is None:
                continue
            di = d[i]
            for j in range(node_count):
                if dk[j] is None:
                    continue
                cand = dik + dk[j]
                if di[j] is None or cand > di[j]:
                    di[j] = cand
    for i in range(node_count):
        if d[i][i] is not None and d[i][i] > 0:
            return None
    return d


def _ceil_div(a: int, b: int) -> int:
    """数学上的 ceil(a / b)（b>0），纯整数。"""
    return -((-a) // b)


def _prefix_closure(
    n: int, lengths: list[int], strain_min: int, strain_max: int,
    windows: list[Window],
) -> list[list[int]] | None:
    """前缀和 P 上的差分约束闭包（窗 + 每段应变界）。"""
    edges: list[tuple[int, int, int]] = []
    for i, length in enumerate(lengths):  # P_{i+1} - P_i ∈ [L·smin, L·smax]
        edges.append((i, i + 1, length * strain_min))
        edges.append((i + 1, i, -length * strain_max))
    for win in windows:  # P_{e+1} - P_s ∈ [lo, hi]
        edges.append((win.start, win.end + 1, win.lo))
        edges.append((win.end + 1, win.start, -win.hi))
    return _longest_path_closure(n + 1, edges)


def _x_bounds_from_prefix(
    n: int,
    lengths: list[int],
    strain_min: int,
    strain_max: int,
    p_closure: list[list[int]],
) -> tuple[list[int], list[int]]:
    """由 P 的最强界导出每段 x_i 的紧整数域。"""
    los: list[int] = []
    his: list[int] = []
    for i, length in enumerate(lengths):
        forward = p_closure[i][i + 1]
        backward = p_closure[i + 1][i]
        # i -> i+1 与 i+1 -> i 均为直接边，闭包后必非 None。
        assert forward is not None and backward is not None
        lo = max(strain_min, _ceil_div(forward, length))
        hi = min(strain_max, (-backward) // length)
        los.append(lo)
        his.append(hi)
    return los, his


def _x_diff_closure(
    n: int,
    los: list[int],
    his: list[int],
    m: int,
    excluded: int = 0,
    jump_edges: tuple[tuple[int, int, int], ...] = (),
) -> tuple[list[int], list[int]] | None:
    """并入一元域与 |x_k-x_{k-1}|<=M 后的 x 最长路闭包（锚点节点 n）。

    excluded 给出不参与平滑的段间边界 k（x_k 与 x_{k-1} 之间，1 基），
    该边界不加入 |Δ|<=M 约束；excluded=0 表示全部边界参与（旧行为）。
    jump_edges 为额外的 x 空间差分边 (u,v,w)（如滑移突跳方向约束
    x_v - x_u >= w），同样并入闭包以提前缩域/发现正环。
    """
    anchor = n
    edges: list[tuple[int, int, int]] = []
    for i in range(n):  # x_i >= anchor + lo_i；anchor >= x_i - hi_i
        edges.append((anchor, i, los[i]))
        edges.append((i, anchor, -his[i]))
    for k in range(1, n):
        if k == excluded:
            continue
        edges.append((k - 1, k, -m))  # x_k >= x_{k-1} - M
        edges.append((k, k - 1, -m))  # x_{k-1} >= x_k - M
    edges.extend(jump_edges)
    closure = _longest_path_closure(n + 1, edges)
    if closure is None:
        return None
    out_lo = [closure[anchor][i] if closure[anchor][i] is not None else los[i] for i in range(n)]
    out_hi = [-closure[i][anchor] if closure[i][anchor] is not None else his[i] for i in range(n)]
    for i in range(n):
        if out_lo[i] > out_hi[i]:
            return None
    return out_lo, out_hi


# --------------------------------------------------------------------------- #
# 通用整数界传播 + 回溯（x 空间）
# --------------------------------------------------------------------------- #
# 约束：sum_j w_j * var_j ∈ [lo, hi]，端点可为 None。
Constraint = tuple[tuple[tuple[int, int], ...], int | None, int | None]


def _propagate(
    los: list[int], his: list[int], constraints: list[Constraint]
) -> tuple[list[int], list[int]] | None:
    """对所有线性区间约束做一元界传播；矛盾返回 None。

    采用队列驱动（AC-3 风格）：初始全部约束入队，仅当某变量域被收紧时，
    才重新检查涉及该变量的约束，避免每轮全量重扫；收缩单调，收敛到的
    不动点与朴素反复全扫完全相同。
    """
    los = list(los)
    his = list(his)
    var_count = len(los)
    involving: list[list[int]] = [[] for _ in range(var_count)]
    for ci, (terms, _a, _b) in enumerate(constraints):
        for j, _w in terms:
            involving[j].append(ci)
    queue = list(range(len(constraints)))
    in_queue = [True] * len(constraints)
    head = 0
    while head < len(queue):
        ci = queue[head]
        head += 1
        in_queue[ci] = False
        terms, a, b = constraints[ci]
        for j, w in terms:
            # 即使域已为单点也不能跳过：约束可能与其冲突。
            others_min = 0
            others_max = 0
            for j2, w2 in terms:
                if j2 == j:
                    continue
                if w2 > 0:
                    others_min += w2 * los[j2]
                    others_max += w2 * his[j2]
                else:
                    others_min += w2 * his[j2]
                    others_max += w2 * los[j2]
            new_lo, new_hi = los[j], his[j]
            if b is not None:  # w*x_j <= b - others_min
                c = b - others_min
                if w > 0:
                    v = c // w
                    if v < new_hi:
                        new_hi = v
                else:
                    v = _ceil_div(c, w)
                    if v > new_lo:
                        new_lo = v
            if a is not None:  # w*x_j >= a - others_max
                c = a - others_max
                if w > 0:
                    v = _ceil_div(c, w)
                    if v > new_lo:
                        new_lo = v
                else:
                    v = c // w
                    if v < new_hi:
                        new_hi = v
            if new_lo > new_hi:
                return None
            if new_lo != los[j] or new_hi != his[j]:
                los[j], his[j] = new_lo, new_hi
                for ci2 in involving[j]:
                    if not in_queue[ci2]:
                        in_queue[ci2] = True
                        queue.append(ci2)
    return los, his


def _greedy_complete(
    los: list[int], his: list[int], constraints: list[Constraint]
) -> tuple[int, ...] | None:
    """贪心补全一个可行赋值；失败返回 None（不影响完全性，仅提前命中）。

    反复选 MRV 变量，取值偏向已赋值的相邻变量（迎合 |Δ|<=M 的平滑结构），
    无已赋值邻居时取域中点；每步做界传播，最终逐约束显式核验。
    对宽域松窗但存在近常数解的情形能避免折半分支盲目深搜。
    """
    los, his = list(los), list(his)
    assigned: dict[int, int] = {}
    for _ in range(len(los)):
        best = -1
        best_size = 0
        for j in range(len(los)):
            if j in assigned:
                continue
            size = his[j] - los[j]
            if best == -1 or size < best_size:
                best = j
                best_size = size
        if best == -1:
            break
        j = best
        neighbor_values = [
            assigned[j2]
            for j2 in (j - 1, j + 1)
            if j2 in assigned and j2 < len(los)
        ]
        if neighbor_values:
            target = sum(neighbor_values) // len(neighbor_values)
            value = min(his[j], max(los[j], target))
        else:
            value = (los[j] + his[j]) // 2
        los[j] = his[j] = value
        narrowed = _propagate(los, his, constraints)
        if narrowed is None:
            return None
        los, his = narrowed
        # 传播只缩不扩，被钉为单例的变量必仍为单例。
        assigned[j] = value
    values = tuple(los[j] for j in range(len(los)))
    for terms, a, b in constraints:
        total = sum(w * values[j] for j, w in terms)
        if a is not None and total < a:
            return None
        if b is not None and total > b:
            return None
    return values


def _search(
    los: list[int], his: list[int], constraints: list[Constraint]
) -> tuple[int, ...] | None:
    """找一个可行赋值（MRV 变量序，域折半分支）；无可行解返回 None。

    根节点先做一次贪心补全以快速命中近常数解；贪心失败不影响完全性。
    """
    narrowed = _propagate(los, his, constraints)
    if narrowed is None:
        return None
    los, his = narrowed
    greedy = _greedy_complete(los, his, constraints)
    if greedy is not None:
        return greedy
    return _search_split(los, his, constraints)


def _search_split(
    los: list[int], his: list[int], constraints: list[Constraint]
) -> tuple[int, ...] | None:
    best = -1
    best_size = 0
    for j in range(len(los)):
        if los[j] != his[j]:
            size = his[j] - los[j]
            if best == -1 or size < best_size:
                best = j
                best_size = size
    if best == -1:
        return tuple(los)
    j = best
    mid = (los[j] + his[j]) // 2
    lo2, hi2 = list(los), list(his)
    hi2[j] = mid
    result = _search_node(lo2, hi2, constraints)
    if result is not None:
        return result
    lo3, hi3 = list(los), list(his)
    lo3[j] = mid + 1
    return _search_node(lo3, hi3, constraints)


def _search_node(
    los: list[int], his: list[int], constraints: list[Constraint]
) -> tuple[int, ...] | None:
    """递归节点：传播后继续折半分支（不重复顶层贪心）。"""
    narrowed = _propagate(los, his, constraints)
    if narrowed is None:
        return None
    return _search_split(narrowed[0], narrowed[1], constraints)


# --------------------------------------------------------------------------- #
# 反演
# --------------------------------------------------------------------------- #
def _window_constraints(n: int, lengths: list[int], windows: list[Window]) -> list[Constraint]:
    cons: list[Constraint] = []
    for win in windows:
        terms = tuple(
            (i, lengths[i]) for i in range(win.start, win.end + 1)
        )
        cons.append((terms, win.lo, win.hi))
    return cons


def _diff_constraints(n: int, m: int, smooth_ks: list[int]) -> list[Constraint]:
    """对 smooth_ks 中的每个段间边界施加 |x_k - x_{k-1}| <= M。"""
    cons: list[Constraint] = []
    for k in smooth_ks:
        cons.append((((k, 1), (k - 1, -1)), -m, m))
    return cons


def _abs_sum_constraints(
    n: int, m: int, smooth_ks: list[int], s_lo: int | None, s_hi: int | None
) -> tuple[list[Constraint], list[int]]:
    """为每个参与平滑的边界引入 e_k>=|Δ_k|、0<=e_k<=M、Σe_k∈[s_lo,s_hi]。

    返回 (约束, 辅助变量编号列表)；辅助变量按 smooth_ks 顺序编号为
    n, n+1, ...（故全部参与时与旧布局完全一致）。
    """
    cons: list[Constraint] = []
    aux_vars: list[int] = []
    for idx, k in enumerate(smooth_ks):
        e_var = n + idx
        aux_vars.append(e_var)
        cons.append((((e_var, 1), (k, -1), (k - 1, 1)), 0, None))
        cons.append((((e_var, 1), (k, 1), (k - 1, -1)), 0, None))
        cons.append((((e_var, 1),), 0, m))
    cons.append((tuple((e_var, 1) for e_var in aux_vars), s_lo, s_hi))
    return cons, aux_vars


def _jump_constraints(k: int, min_jump: int) -> list[tuple[int, tuple[int, int, int], Constraint]]:
    """滑移边界 k 上的突跳约束：|Δ_k| >= min_jump，放松为两方向析取。

    方向 +1：x_k - x_{k-1} >= min_jump；方向 -1：x_{k-1} - x_k >= min_jump。
    每项给出 (方向, x 空间差分边(u,v,w): x_v-x_u>=w, 通用线性约束)。
    """
    return [
        (1, (k - 1, k, min_jump), (((k, 1), (k - 1, -1)), min_jump, None)),
        (-1, (k, k - 1, min_jump), (((k - 1, 1), (k, -1)), min_jump, None)),
    ]


def invert_payload(payload: object) -> dict:
    """完整反演，返回可直接复核的结果字典；校验失败/不可行抛对应异常。"""
    lengths, strain_min, strain_max, windows, slip_cfg = validate(payload)
    n = len(lengths)
    window_cons = _window_constraints(n, lengths, windows)

    # 前缀和差分闭包：与 M 无关，只算一次。
    p_closure = _prefix_closure(n, lengths, strain_min, strain_max, windows)
    if p_closure is None:
        raise InfeasibleError("observation windows are mutually inconsistent")
    base_x_lo, base_x_hi = _x_bounds_from_prefix(
        n, lengths, strain_min, strain_max, p_closure
    )
    full_range = strain_max - strain_min

    if slip_cfg is None:
        return _solve_classic(
            n, window_cons, base_x_lo, base_x_hi, full_range, lengths, windows
        )
    candidate_bounds, min_jump = slip_cfg
    return _solve_slip(
        n,
        window_cons,
        base_x_lo,
        base_x_hi,
        lengths,
        windows,
        candidate_bounds,
        min_jump,
    )


def _solve_classic(
    n: int,
    window_cons: list[Constraint],
    base_x_lo: list[int],
    base_x_hi: list[int],
    full_range: int,
    lengths: list[int],
    windows: list[Window],
) -> dict:
    """旧行为：全部 n-1 个相邻差参与两级平滑指标的三级裁决。"""
    smooth_ks = list(range(1, n))

    def feasible(m: int, s_hi: int | None = None, pin: tuple[int, int] | None = None) -> bool:
        """|Δ|<=m（若给 s_hi 则 Σ|Δ|<=s_hi），可附加 x_var<=t 的钉压。"""
        bounds = _x_diff_closure(n, base_x_lo, base_x_hi, m)
        if bounds is None:
            return False
        x_lo, x_hi = bounds
        constraints = list(window_cons)
        constraints += _diff_constraints(n, m, smooth_ks)
        if s_hi is not None:
            x_lo = x_lo + [0] * (n - 1)
            x_hi = x_hi + [m] * (n - 1)
            abs_cons, _ = _abs_sum_constraints(n, m, smooth_ks, None, s_hi)
            constraints += abs_cons
        if pin is not None:
            var, t = pin
            constraints.append((((var, 1),), None, t))
        return _search(list(x_lo), list(x_hi), constraints) is not None

    # 阶段 0：无平滑约束（M = 全量程）下的整数可行性。
    if not feasible(full_range):
        raise InfeasibleError("observation windows are mutually inconsistent")

    # 阶段 1：二分最小可行 M。
    m_lo, m_hi = 0, full_range
    while m_lo < m_hi:
        mid = (m_lo + m_hi) // 2
        if feasible(mid):
            m_hi = mid
        else:
            m_lo = mid + 1
    best_m = m_lo

    # 阶段 2：二分最小可行 S = Σ|Δ_k|。
    s_lo, s_hi = 0, (n - 1) * best_m
    while s_lo < s_hi:
        mid = (s_lo + s_hi) // 2
        if feasible(best_m, s_hi=mid):
            s_hi = mid
        else:
            s_lo = mid + 1
    best_s = s_lo

    # 阶段 3：逐段钉死字典序最小值（x_0, x_1, ... 顺序；可行性关于阈值单调）。
    strains = _lexicographic_min(
        n,
        base_x_lo,
        base_x_hi,
        window_cons,
        smooth_ks,
        best_m,
        best_s,
        extra_cons=[],
    )

    diffs = [strains[i] - strains[i - 1] for i in range(1, n)]
    return _build_result(lengths, windows, strains, diffs, smooth_ks, best_m, best_s)


def _lexicographic_min(
    n: int,
    base_x_lo: list[int],
    base_x_hi: list[int],
    window_cons: list[Constraint],
    smooth_ks: list[int],
    best_m: int,
    best_s: int,
    extra_cons: list[Constraint],
    excluded: int = 0,
    jump_edges: tuple[tuple[int, int, int], ...] = (),
    cutoff: tuple[int, ...] | None = None,
) -> list[int] | None:
    """在固定 M、S 与附加约束下逐段钉死字典序最小序列。

    给定 cutoff 时只寻找字典序严格小于 cutoff 的序列；不存在则返回 None。
    前缀一旦严格小于 cutoff，后续变量即解除 cutoff 限制。
    """
    bounds = _x_diff_closure(
        n, base_x_lo, base_x_hi, best_m, excluded=excluded, jump_edges=jump_edges
    )
    assert bounds is not None
    x_lo, x_hi = bounds
    n_aux = len(smooth_ks)
    x_lo = x_lo + [0] * n_aux
    x_hi = x_hi + [best_m] * n_aux
    constraints = list(window_cons)
    constraints += _diff_constraints(n, best_m, smooth_ks)
    abs_cons, _ = _abs_sum_constraints(n, best_m, smooth_ks, best_s, best_s)
    constraints += abs_cons
    constraints += extra_cons

    def feasible_with_cap(var: int, cap: int) -> bool:
        """x_var <= cap（作为收紧的一元域，而非新增约束）下是否可行。"""
        if cap < x_lo[var]:
            return False
        lo2, hi2 = list(x_lo), list(x_hi)
        hi2[var] = min(hi2[var], cap)
        if lo2[var] > hi2[var]:
            return False
        return _search(lo2, hi2, constraints) is not None

    strains: list[int] = []
    prefix_equal = cutoff is not None  # 已钉前缀是否仍与 cutoff 完全相同
    for var in range(n):
        t_lo, t_hi = x_lo[var], x_hi[var]
        if prefix_equal:
            assert cutoff is not None
            # 前缀相同位置：x_var 不得超过 cutoff[var]。
            t_hi = min(t_hi, cutoff[var])
            if t_lo > t_hi:
                return None
            # 先在 [< cutoff[var]] 内找最小，使前缀在此处严格变小；
            # 找不到则取 x_var == cutoff[var] 继续保持前缀相同。
            strict_hi = cutoff[var] - 1
            if t_lo <= strict_hi:
                a, b2 = t_lo, strict_hi
                while a < b2:
                    mid = (a + b2) // 2
                    if feasible_with_cap(var, mid):
                        b2 = mid
                    else:
                        a = mid + 1
                if feasible_with_cap(var, a):
                    chosen = a
                    prefix_equal = False
                else:
                    chosen = cutoff[var]
            else:
                chosen = cutoff[var]
            strains.append(chosen)
            x_lo[var] = x_hi[var] = chosen
        else:
            while t_lo < t_hi:
                mid = (t_lo + t_hi) // 2
                if feasible_with_cap(var, mid):
                    t_hi = mid
                else:
                    t_lo = mid + 1
            strains.append(t_lo)
            x_lo[var] = x_hi[var] = t_lo
        narrowed = _propagate(x_lo, x_hi, constraints)
        if narrowed is None:
            return None
        x_lo, x_hi = narrowed
    if prefix_equal:
        # 序列与 cutoff 完全相等，并非严格更小。
        return None
    return strains


def _solve_slip(
    n: int,
    window_cons: list[Constraint],
    base_x_lo: list[int],
    base_x_hi: list[int],
    lengths: list[int],
    windows: list[Window],
    candidate_bounds: list[int],
    min_jump: int,
) -> dict:
    """护套滑移联合裁决。

    在每个候选边界 b 上：恰有该边界不参与两级平滑指标，且必须满足
    |Δ_b| >= min_jump；突跳方向（+1 / -1）与逐段整数应变一并裁决。
    跨候选依次最小化 (其余相邻差最大绝对值 M, 其余相邻差绝对值之和 S,
    应变序列字典序)，完全并列时按候选录入顺序取靠前者。
    """

    def direction_optimal_ms(
        b: int,
        jump_edge: tuple[int, int, int],
        jump_con: Constraint,
    ) -> tuple[int, int] | None:
        """固定边界与突跳方向，只钉死前两级指标 (M, S)；不可行返回 None。

        字典序（第三级）代价高且只影响最终裁决，故延迟到全部候选的
        (M, S) 决出全局最优后，仅对并列方向计算。
        """
        smooth_ks = [k for k in range(1, n) if k != b]
        n_smooth = len(smooth_ks)
        jump_edges = (jump_edge,)

        # 阶段 0：仅有观测窗、应变界与突跳方向约束时的整数可行性；
        # 见证解同时给出 M 的紧上界，避免对全量程二分。
        witness = _search(
            list(base_x_lo), list(base_x_hi), list(window_cons) + [jump_con]
        )
        if witness is None:
            return None
        m_wit = max(abs(witness[k] - witness[k - 1]) for k in smooth_ks)

        def feasible_m(m: int, s_hi: int | None = None) -> tuple[int, ...] | None:
            bounds = _x_diff_closure(
                n, base_x_lo, base_x_hi, m, excluded=b, jump_edges=jump_edges
            )
            if bounds is None:
                return None
            x_lo, x_hi = bounds
            constraints = list(window_cons)
            constraints += _diff_constraints(n, m, smooth_ks)
            constraints.append(jump_con)
            if s_hi is not None:
                x_lo = x_lo + [0] * n_smooth
                x_hi = x_hi + [m] * n_smooth
                abs_cons, _ = _abs_sum_constraints(n, m, smooth_ks, None, s_hi)
                constraints += abs_cons
            return _search(list(x_lo), list(x_hi), constraints)

        # 阶段 1：二分最小可行 M（突跳边界不受 M 限制）。
        m_lo, m_hi = 0, m_wit
        witness_m: tuple[int, ...] | None = None
        while m_lo < m_hi:
            mid = (m_lo + m_hi) // 2
            found = feasible_m(mid)
            if found is not None:
                m_hi = mid
                witness_m = found
            else:
                m_lo = mid + 1
        best_m = m_lo
        if witness_m is None or max(
            abs(witness_m[k] - witness_m[k - 1]) for k in smooth_ks
        ) != best_m:
            witness_m = feasible_m(best_m)
        assert witness_m is not None

        # 阶段 2：二分最小可行 S（只对参与平滑的边界求和）；
        # 上界取最优 M 下见证解的 S。
        s_wit = sum(abs(witness_m[k] - witness_m[k - 1]) for k in smooth_ks)
        s_lo, s_hi = 0, s_wit
        while s_lo < s_hi:
            mid = (s_lo + s_hi) // 2
            if feasible_m(best_m, s_hi=mid) is not None:
                s_hi = mid
            else:
                s_lo = mid + 1
        return best_m, s_lo

    # ---- 阶段 A：对每个候选边界的两个突跳方向钉死 (M, S) -------------- #
    candidate_reports: list[dict] = []
    feasible_options: list[dict] = []
    for idx, b in enumerate(candidate_bounds):
        direction_reports: list[dict] = []
        for direction, jump_edge, jump_con in _jump_constraints(b, min_jump):
            ms = direction_optimal_ms(b, jump_edge, jump_con)
            if ms is None:
                direction_reports.append({"direction": direction, "feasible": False})
                continue
            best_m, best_s = ms
            direction_reports.append(
                {
                    "direction": direction,
                    "feasible": True,
                    "max_adjacent_diff": best_m,
                    "sum_adjacent_abs_diff": best_s,
                }
            )
            feasible_options.append(
                {
                    "entry_order": idx,
                    "boundary": b,
                    "direction": direction,
                    "jump_edge": jump_edge,
                    "jump_con": jump_con,
                    "max_adjacent_diff": best_m,
                    "sum_adjacent_abs_diff": best_s,
                }
            )
        any_feasible = [d for d in direction_reports if d["feasible"]]
        report = {
            "boundary": b,
            "entry_order": idx,
            "feasible": bool(any_feasible),
            "directions": direction_reports,
        }
        if any_feasible:
            best_ms_here = min(
                (d["max_adjacent_diff"], d["sum_adjacent_abs_diff"])
                for d in any_feasible
            )
            report["max_adjacent_diff"] = best_ms_here[0]
            report["sum_adjacent_abs_diff"] = best_ms_here[1]
            report["optimal_directions"] = [
                d["direction"]
                for d in any_feasible
                if (d["max_adjacent_diff"], d["sum_adjacent_abs_diff"]) == best_ms_here
            ]
        candidate_reports.append(report)

    if not feasible_options:
        raise InfeasibleError(
            "no integer strain assignment satisfies all observation windows and "
            "the minimum jump at any candidate slip boundary"
        )

    # ---- 阶段 B：全局最优 (M, S) 仅在并列方向间以字典序决胜 ------------ #
    # 三级裁决：依次以 (序列字典序, 候选录入顺序) 取最小。为避免对每个
    # 并列方向都做完整 12 变量钉压，按录入顺序逐个用「严格小于当前最优」
    # 的 cutoff 判定：落败方向通常在头一两个变量即被剪枝。
    best_ms = min(
        (o["max_adjacent_diff"], o["sum_adjacent_abs_diff"]) for o in feasible_options
    )
    tied = sorted(
        (
            o
            for o in feasible_options
            if (o["max_adjacent_diff"], o["sum_adjacent_abs_diff"]) == best_ms
        ),
        key=lambda o: o["entry_order"],
    )
    winner: dict | None = None
    strains: list[int] | None = None
    for option in tied:
        b = option["boundary"]
        smooth_ks = [k for k in range(1, n) if k != b]
        candidate = _lexicographic_min(
            n,
            base_x_lo,
            base_x_hi,
            window_cons,
            smooth_ks,
            best_ms[0],
            best_ms[1],
            extra_cons=[option["jump_con"]],
            excluded=b,
            jump_edges=(option["jump_edge"],),
            cutoff=None if winner is None else tuple(strains),
        )
        if candidate is not None:
            # 录入顺序靠前者同字典序时不被覆盖（cutoff 要求严格更小）。
            winner = option
            strains = candidate
    assert winner is not None and strains is not None
    b = winner["boundary"]
    smooth_ks = [k for k in range(1, n) if k != b]
    diffs = [strains[i] - strains[i - 1] for i in range(1, n)]
    best_m, best_s = best_ms

    result = _build_result(lengths, windows, strains, diffs, smooth_ks, best_m, best_s)
    result["slip"] = {
        "selected_boundary": b,
        "selected_boundary_entry_order": winner["entry_order"],
        "signed_jump": diffs[b - 1],
        "jump_direction": winner["direction"],
        "min_jump": min_jump,
        "excluded_from_smoothness": [b],
        "smoothness_boundaries": smooth_ks,
        "tie_break": "candidate entry order",
        "candidates": candidate_reports,
    }
    return result


def _build_result(
    lengths: list[int],
    windows: list[Window],
    strains: list[int],
    diffs: list[int],
    smooth_ks: list[int],
    best_m: int,
    best_s: int,
) -> dict:
    prefix = [0]
    for length in lengths:
        prefix.append(prefix[-1] + length)
    window_checks = []
    for idx, win in enumerate(windows):
        weighted_sum = 0
        for i in range(win.start, win.end + 1):
            weighted_sum += lengths[i] * strains[i]
        total_length = prefix[win.end + 1] - prefix[win.start]
        window_checks.append(
            {
                "index": idx,
                "start_segment": win.start + 1,
                "end_segment": win.end + 1,
                "total_length": total_length,
                "min_elongation": win.lo,
                "max_elongation": win.hi,
                "weighted_strain_sum": weighted_sum,
                "satisfied": win.lo <= weighted_sum <= win.hi,
            }
        )
    # 两级平滑指标仅由参与平滑的相邻差直接复核（重算以自证，不采用搜索内部值）；
    # 滑移模式下被选中的边界（不在 smooth_ks 中）被排除在外。
    smooth_diffs = [diffs[k - 1] for k in smooth_ks]
    recomputed_m = max(abs(d) for d in smooth_diffs) if smooth_diffs else 0
    recomputed_s = sum(abs(d) for d in smooth_diffs)
    assert recomputed_m == best_m and recomputed_s == best_s
    return {
        "segment_count": len(lengths),
        "strains": strains,
        "adjacent_diffs": diffs,
        "objectives": {
            "max_adjacent_diff": recomputed_m,
            "sum_adjacent_abs_diff": recomputed_s,
        },
        "window_checks": window_checks,
        "criteria_order": [
            "max_adjacent_diff",
            "sum_adjacent_abs_diff",
            "lexicographic",
        ],
    }
