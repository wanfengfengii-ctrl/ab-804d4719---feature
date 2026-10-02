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

可选滑移接头模式（slip_joint）
------------------------------
调用方给出 2..6 个候选段间边界 b（1 基，b 位于第 b 段与第 b+1 段之间，
即相邻差 adjacent_diffs[b-1]）与正整数最小突跳 J。启用后：

- 必须且只会选择一个候选边界，要求 |x_b - x_{b-1}| >= J（方向联合裁决）；
- 被选边界**不参与**两级平滑指标，其余相邻差全部参与：
      M' = max_{k != b} |x_k - x_{k-1}|
      S' = Σ_{k != b} |x_k - x_{k-1}|
- 联合裁决在（边界、突跳方向、逐段整数应变）上进行，依次最小化
  (M', S', 应变序列字典序, 候选录入顺序)；
- 每个候选边界单独给出可行性及其自身的最优 (M', S')；全部候选均无满足
  全部观测窗与最小突跳的整数解释时，整体判不可行。
未提供 slip_joint 时，请求、响应、裁决与错误行为与原接口完全一致。
"""

from __future__ import annotations

from dataclasses import dataclass

MIN_SEGMENTS = 6
MAX_SEGMENTS = 12
MIN_WINDOWS = 8
MAX_WINDOWS = 20
MIN_SLIP_CANDIDATES = 2
MAX_SLIP_CANDIDATES = 6


@dataclass(frozen=True)
class SlipConfig:
    """滑移接头候选配置：candidates 为 0 基段间边界（差分值索引），保持录入顺序。"""

    candidates: tuple[int, ...]
    min_jump: int


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


def _validate_slip_joint(
    payload: dict, n: int | None, fields: list[dict[str, str]]
) -> SlipConfig | None:
    """校验可选滑移接头配置；未提供返回 None，非法时登记字段错误。"""
    raw = payload.get("slip_joint")
    if raw is None:
        return None
    if not isinstance(raw, dict):
        _err(fields, "slip_joint", 'must be an object {"candidate_boundaries": [int], "min_jump": int}')
        return None
    for key in raw:
        if key not in ("candidate_boundaries", "min_jump"):
            _err(fields, f"slip_joint.{key}", "unknown field")
    ok = True

    raw_bounds = raw.get("candidate_boundaries")
    if raw_bounds is None:
        _err(fields, "slip_joint.candidate_boundaries", "field is required")
        ok = False
    elif not isinstance(raw_bounds, list):
        _err(
            fields,
            "slip_joint.candidate_boundaries",
            "must be an array of 2..6 segment boundaries (1-based, between two segments)",
        )
        ok = False
    else:
        count = len(raw_bounds)
        if not (MIN_SLIP_CANDIDATES <= count <= MAX_SLIP_CANDIDATES):
            _err(
                fields,
                "slip_joint.candidate_boundaries",
                f"must contain between {MIN_SLIP_CANDIDATES} and {MAX_SLIP_CANDIDATES}"
                f" candidate boundaries (got {count})",
            )
            ok = False
        candidates: list[int] = []
        seen: set[int] = set()
        for i, item in enumerate(raw_bounds):
            if not _is_int(item):
                _err(
                    fields,
                    f"slip_joint.candidate_boundaries[{i}]",
                    "must be an integer segment boundary",
                )
                ok = False
                continue
            # 段间边界 b 位于段 b 与段 b+1 之间（1 基），故 1 <= b <= n-1。
            if n is not None and not (1 <= item <= n - 1):
                _err(
                    fields,
                    f"slip_joint.candidate_boundaries[{i}]",
                    f"boundary must lie between two segments: 1..{n - 1} for {n} segments"
                    f" (got {item})",
                )
                ok = False
                continue
            if item in seen:
                _err(
                    fields,
                    f"slip_joint.candidate_boundaries[{i}]",
                    f"duplicate candidate boundary {item}",
                )
                ok = False
                continue
            seen.add(item)
            candidates.append(item - 1)  # 内部统一使用 0 基相邻差索引

    min_jump = raw.get("min_jump")
    if min_jump is None:
        _err(fields, "slip_joint.min_jump", "field is required")
        ok = False
    elif not _is_int(min_jump) or min_jump < 1:
        _err(fields, "slip_joint.min_jump", "must be a positive integer")
        ok = False

    if not ok:
        return None
    return SlipConfig(candidates=tuple(candidates), min_jump=min_jump)


_ALLOWED_TOP_LEVEL = {"segment_lengths", "strain_bounds", "windows", "slip_joint"}


def validate(payload: object) -> tuple[list[int], int, int, list[Window], SlipConfig | None]:
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
    slip = _validate_slip_joint(payload, n, fields)
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
    n: int, los: list[int], his: list[int], m: int, skip: int | None = None
) -> tuple[list[int], list[int]] | None:
    """并入一元域与 |x_k-x_{k-1}|<=M 后的 x 最长路闭包（锚点节点 n）。

    skip 给定时（0 基相邻差索引），该边界不并入平滑约束（滑移突跳所在）。
    """
    anchor = n
    edges: list[tuple[int, int, int]] = []
    for i in range(n):  # x_i >= anchor + lo_i；anchor >= x_i - hi_i
        edges.append((anchor, i, los[i]))
        edges.append((i, anchor, -his[i]))
    for k in range(1, n):
        if skip is not None and k - 1 == skip:
            continue  # 突跳边界不参与平滑指标
        edges.append((k - 1, k, -m))  # x_k >= x_{k-1} - M
        edges.append((k, k - 1, -m))  # x_{k-1} >= x_k - M
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
    """对所有线性区间约束反复做一元界传播；矛盾返回 None。"""
    los = list(los)
    his = list(his)
    while True:
        changed = False
        for terms, a, b in constraints:
            for j, w in terms:
                # 即使域已为单点也不能跳过：后加入的约束可能与其冲突。
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
                if b is not None:  # w*x_j <= b - others_min
                    c = b - others_min
                    if w > 0:
                        v = c // w
                        if v < his[j]:
                            his[j] = v
                            changed = True
                    else:
                        v = _ceil_div(c, w)
                        if v > los[j]:
                            los[j] = v
                            changed = True
                if a is not None:  # w*x_j >= a - others_max
                    c = a - others_max
                    if w > 0:
                        v = _ceil_div(c, w)
                        if v > los[j]:
                            los[j] = v
                            changed = True
                    else:
                        v = c // w
                        if v < his[j]:
                            his[j] = v
                            changed = True
                if los[j] > his[j]:
                    return None
        if not changed:
            return los, his


def _search(
    los: list[int], his: list[int], constraints: list[Constraint]
) -> tuple[int, ...] | None:
    """找一个可行赋值（MRV 变量序，域折半分支）；无可行解返回 None。"""
    narrowed = _propagate(los, his, constraints)
    if narrowed is None:
        return None
    los, his = narrowed
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
    result = _search(lo2, hi2, constraints)
    if result is not None:
        return result
    lo3, hi3 = list(los), list(his)
    lo3[j] = mid + 1
    return _search(lo3, hi3, constraints)


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


def _diff_constraints(n: int, m: int, skip: int | None = None) -> list[Constraint]:
    """|x_k - x_{k-1}| <= M（亦供通用传播，与差分闭包一致）；skip 边界除外。"""
    cons: list[Constraint] = []
    for k in range(1, n):
        if skip is not None and k - 1 == skip:
            continue
        cons.append((((k, 1), (k - 1, -1)), -m, m))
    return cons


def _jump_branches(jump_idx: int, min_jump: int) -> tuple[list[Constraint], list[Constraint]]:
    """滑移突跳 |x_{j+1} - x_j| >= J（方向由搜索自行裁决），拆为两个析取支。

    支 A：x_{j+1} - x_j >= J；支 B：x_j - x_{j+1} >= J。
    """
    j = jump_idx
    branch_up: list[Constraint] = [(((j + 1, 1), (j, -1)), min_jump, None)]
    branch_down: list[Constraint] = [(((j, 1), (j + 1, -1)), min_jump, None)]
    return branch_up, branch_down


def _abs_sum_constraints(
    n: int, m: int, s_lo: int | None, s_hi: int | None, skip: int | None = None
) -> list[Constraint]:
    """e_k >= |x_k-x_{k-1}|、0<=e_k<=M、Σe_k∈[s_lo,s_hi]；skip 边界不设 e。"""
    cons: list[Constraint] = []
    e_index: dict[int, int] = {}  # 相邻差索引 k-1 -> 辅助变量号
    next_e = n
    for k in range(1, n):
        if skip is not None and k - 1 == skip:
            continue
        e_var = next_e
        next_e += 1
        e_index[k - 1] = e_var
        cons.append((((e_var, 1), (k, -1), (k - 1, 1)), 0, None))
        cons.append((((e_var, 1), (k, 1), (k - 1, -1)), 0, None))
        cons.append((((e_var, 1),), 0, m))
    if e_index:
        cons.append((tuple((e, 1) for e in e_index.values()), s_lo, s_hi))
    # 无参与边界时 Σe 恒为 0：当前 n>=6 且至多排除一个边界，不会发生；
    # 若发生，调用方只会传 0 区间，平凡成立而无需加入约束。
    return cons


def _optimize_smooth(
    n: int,
    base_x_lo: list[int],
    base_x_hi: list[int],
    window_cons: list[Constraint],
    full_range: int,
    extra_cons: list[Constraint] | None = None,
    skip: int | None = None,
) -> tuple[int, int, list[int]] | None:
    """通用三级平滑裁决，返回 (M', S', 字典序最优 x)；无可行解返回 None。

    - extra_cons：始终并入的附加约束（滑移模式下为某一突跳方向支）；
    - skip：被排除在两级平滑指标之外的相邻差索引（含其闭包缩域与 e 变量）。
    三级依次最小化：参与边界相邻差最大绝对值、参与边界绝对值之和、x 字典序。
    """
    extra_cons = extra_cons or []
    e_count = (n - 1) if skip is None else (n - 2)

    def solve(
        m: int, s_hi: int | None = None, pin: tuple[int, int] | None = None
    ) -> tuple[int, ...] | None:
        """|Δ|<=m（skip 边界除外；若给 s_hi 则参与边界 Σ|Δ|<=s_hi），可钉压 x_var<=t。

        返回含 x（前 n 项）与可能的辅助 e 变量的完整赋值；无可行解返回 None。
        """
        bounds = _x_diff_closure(n, base_x_lo, base_x_hi, m, skip)
        if bounds is None:
            return None
        x_lo, x_hi = bounds
        constraints = list(window_cons)
        constraints += _diff_constraints(n, m, skip)
        constraints += extra_cons
        if s_hi is not None:
            x_lo = x_lo + [0] * e_count
            x_hi = x_hi + [m] * e_count
            constraints += _abs_sum_constraints(n, m, None, s_hi, skip)
        if pin is not None:
            var, t = pin
            constraints.append((((var, 1),), None, t))
        return _search(list(x_lo), list(x_hi), constraints)

    def feasible(
        m: int, s_hi: int | None = None, pin: tuple[int, int] | None = None
    ) -> bool:
        return solve(m, s_hi, pin) is not None

    # 阶段 0：无平滑约束（M = 全量程）下的整数可行性，并由找到的解释
    # 直接得到 M 的紧上界，避免从全量程（可能极大）开始二分。
    seed_assignment = solve(full_range)
    if seed_assignment is None:
        return None
    seed_m = max(
        abs(seed_assignment[i] - seed_assignment[i - 1])
        for i in range(1, n)
        if skip is None or i - 1 != skip
    )

    # 阶段 1：二分最小可行 M。
    m_lo, m_hi = 0, seed_m
    while m_lo < m_hi:
        mid = (m_lo + m_hi) // 2
        if feasible(mid):
            m_hi = mid
        else:
            m_lo = mid + 1
    best_m = m_lo

    # 阶段 2：二分最小可行 S = 参与边界 Σ|Δ_k|。
    s_lo, s_hi = 0, e_count * best_m
    while s_lo < s_hi:
        mid = (s_lo + s_hi) // 2
        if feasible(best_m, s_hi=mid):
            s_hi = mid
        else:
            s_lo = mid + 1
    best_s = s_lo

    # 阶段 3：逐段钉死字典序最小值（x_0, x_1, ... 顺序；可行性关于阈值单调）。
    bounds = _x_diff_closure(n, base_x_lo, base_x_hi, best_m, skip)
    assert bounds is not None
    x_lo, x_hi = bounds
    x_lo = x_lo + [0] * e_count
    x_hi = x_hi + [best_m] * e_count
    constraints = list(window_cons)
    constraints += _diff_constraints(n, best_m, skip)
    constraints += extra_cons
    constraints += _abs_sum_constraints(n, best_m, best_s, best_s, skip)

    strains: list[int] = []
    for var in range(n):
        t_lo, t_hi = x_lo[var], x_hi[var]
        while t_lo < t_hi:
            mid = (t_lo + t_hi) // 2
            trial = constraints + [(((var, 1),), None, mid)]
            if _search(list(x_lo), list(x_hi), trial) is not None:
                t_hi = mid
            else:
                t_lo = mid + 1
        strains.append(t_lo)
        x_lo[var] = x_hi[var] = t_lo
        narrowed = _propagate(x_lo, x_hi, constraints)
        assert narrowed is not None  # 已选最优值必然可行
        x_lo, x_hi = narrowed

    return best_m, best_s, strains


def invert_payload(payload: object) -> dict:
    """完整反演，返回可直接复核的结果字典；校验失败/不可行抛对应异常。"""
    lengths, strain_min, strain_max, windows, slip = validate(payload)
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

    if slip is None:
        outcome = _optimize_smooth(
            n, base_x_lo, base_x_hi, window_cons, full_range
        )
        if outcome is None:
            raise InfeasibleError("observation windows are mutually inconsistent")
        best_m, best_s, strains = outcome
        diffs = [strains[i] - strains[i - 1] for i in range(1, n)]
        return _build_result(lengths, windows, strains, diffs, best_m, best_s)

    # 滑移接头模式：逐候选（保持录入顺序）、逐突跳方向做三级裁决，
    # 再按 (M', S', x 字典序, 候选录入顺序) 联合选取唯一边界。
    candidate_reports: list[dict] = []
    best_overall: tuple[int, int, tuple[int, ...], int, list[int]] | None = None
    for order, boundary0 in enumerate(slip.candidates):
        branch_up, branch_down = _jump_branches(boundary0, slip.min_jump)
        local_best: tuple[int, int, tuple[int, ...], list[int]] | None = None
        for branch_cons in (branch_up, branch_down):
            outcome = _optimize_smooth(
                n,
                base_x_lo,
                base_x_hi,
                window_cons,
                full_range,
                extra_cons=branch_cons,
                skip=boundary0,
            )
            if outcome is None:
                continue
            m_val, s_val, xs = outcome
            key = (m_val, s_val, tuple(xs))
            if local_best is None or key < local_best[:3]:
                local_best = (m_val, s_val, tuple(xs), xs)
        report: dict = {
            "boundary": boundary0 + 1,  # 1 基段间边界
            "feasible": local_best is not None,
        }
        if local_best is not None:
            m_val, s_val, _, _ = local_best
            report["objectives"] = {
                "max_other_adjacent_diff": m_val,
                "sum_other_adjacent_abs_diff": s_val,
            }
            cand_key = (m_val, s_val, local_best[2], order)
            if best_overall is None or cand_key < (
                best_overall[0], best_overall[1], best_overall[2], best_overall[3]
            ):
                best_overall = (m_val, s_val, local_best[2], order, local_best[3])
        candidate_reports.append(report)

    if best_overall is None:
        raise InfeasibleError(
            "no candidate slip boundary admits an integer strain assignment satisfying"
            " all observation windows and min_jump"
        )

    best_m, best_s, _, order, strains = best_overall
    boundary0 = slip.candidates[order]
    diffs = [strains[i] - strains[i - 1] for i in range(1, n)]
    jump = diffs[boundary0]  # 带符号突跳 x_{b+1}-x_b
    assert abs(jump) >= slip.min_jump
    return _build_slip_result(
        lengths,
        windows,
        strains,
        diffs,
        best_m,
        best_s,
        boundary0,
        jump,
        slip.min_jump,
        candidate_reports,
    )


def _window_checks(
    lengths: list[int], windows: list[Window], strains: list[int]
) -> list[dict]:
    prefix = [0]
    for length in lengths:
        prefix.append(prefix[-1] + length)
    checks = []
    for idx, win in enumerate(windows):
        weighted_sum = 0
        for i in range(win.start, win.end + 1):
            weighted_sum += lengths[i] * strains[i]
        total_length = prefix[win.end + 1] - prefix[win.start]
        checks.append(
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
    return checks


def _build_result(
    lengths: list[int],
    windows: list[Window],
    strains: list[int],
    diffs: list[int],
    best_m: int,
    best_s: int,
) -> dict:
    # 两级平滑指标均由相邻差直接复核（重算以自证，不直接采用搜索内部值）。
    recomputed_m = max(abs(d) for d in diffs)
    recomputed_s = sum(abs(d) for d in diffs)
    assert recomputed_m == best_m and recomputed_s == best_s
    return {
        "segment_count": len(lengths),
        "strains": strains,
        "adjacent_diffs": diffs,
        "objectives": {
            "max_adjacent_diff": recomputed_m,
            "sum_adjacent_abs_diff": recomputed_s,
        },
        "window_checks": _window_checks(lengths, windows, strains),
        "criteria_order": [
            "max_adjacent_diff",
            "sum_adjacent_abs_diff",
            "lexicographic",
        ],
    }


def _build_slip_result(
    lengths: list[int],
    windows: list[Window],
    strains: list[int],
    diffs: list[int],
    best_m: int,
    best_s: int,
    boundary0: int,
    jump: int,
    min_jump: int,
    candidate_reports: list[dict],
) -> dict:
    """滑移模式结果：选中边界不参与平滑指标，其余相邻差全部参与并逐项可复核。"""
    n = len(lengths)
    other_abs = [abs(d) for k, d in enumerate(diffs) if k != boundary0]
    # 其余相邻差仍全部参与：数量恰为 n-2，且重算指标必须与裁决值逐位一致。
    assert len(other_abs) == n - 2
    recomputed_m = max(other_abs)
    recomputed_s = sum(other_abs)
    assert recomputed_m == best_m and recomputed_s == best_s
    assert abs(jump) == abs(diffs[boundary0]) and abs(jump) >= min_jump

    smooth_participation = [
        {
            "boundary": k + 1,  # 相邻差 adjacent_diffs[k] 两侧的段间边界（1 基）
            "adjacent_diff": diffs[k],
            "participates_in_smoothness": k != boundary0,
        }
        for k in range(n - 1)
    ]
    return {
        "segment_count": n,
        "strains": strains,
        "adjacent_diffs": diffs,
        "slip_joint": {
            "selected_boundary": boundary0 + 1,
            "signed_jump": jump,
            "min_jump": min_jump,
            "jump_satisfied": abs(jump) >= min_jump,
            "candidates": candidate_reports,
        },
        "objectives": {
            # 仅统计除选中边界外的其余相邻差。
            "max_other_adjacent_diff": recomputed_m,
            "sum_other_adjacent_abs_diff": recomputed_s,
        },
        "smooth_diff_participation": smooth_participation,
        "window_checks": _window_checks(lengths, windows, strains),
        "criteria_order": [
            "max_other_adjacent_diff",
            "sum_other_adjacent_abs_diff",
            "lexicographic",
            "candidate_input_order",
        ],
    }
