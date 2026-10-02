# 海缆连续缆段整数微应变联合反演服务

海缆检修组只能取得多个**重叠标距**的累计伸长量。本服务把 6–12 段缆段的
整数微应变作为未知量，对 8–20 个由「连续起止段 + 累计伸长量闭区间」组成的
观测窗做**精确整数联合反演**，避免逐窗均摊导致彼此矛盾的局部结论。

## 数学模型与优选准则

设第 i 段长度为 L_i、待求整数微应变为 x_i。每个观测窗 [s,e] 要求

    Σ_{i=s..e} L_i · x_i ∈ [min_elongation, max_elongation]

且每段都满足统一应变闭区间 `strain_bounds.min ≤ x_i ≤ strain_bounds.max`。
所有运算与比较均为 Python 任意精度**整数**，无任何浮点参与。

可行解依次最小化（三级字典序，前一级最优后才比较下一级）：

1. `max_adjacent_diff` = max_k |x_k − x_{k−1}|
2. `sum_adjacent_abs_diff` = Σ_k |x_k − x_{k−1}|
3. 应变序列 (x_0, …, x_{n−1}) 的字典序

### 可选：护套滑移接头定位（`slip_joint`）

护套滑移会在某个接头两侧形成真实应变突跳；全局平滑反演会把突跳摊向邻段。
请求体中可选提供 2–6 个**候选段间边界**与正整数**最小突跳**：

- 段间边界 `b` 为 1 基，位于段 `b` 与段 `b+1` 之间，取值 `1..n-1`，
  对应 `adjacent_diffs[b-1]`；候选必须互异。
- 启用后**必须且只会选择一个**候选边界，其带符号应变差
  `x_{b+1} − x_b` 的绝对值不得低于 `min_jump`。
- 被选边界**不参与**两级平滑指标，其余相邻差仍**全部**参与：
  - `max_other_adjacent_diff` = max_{k ≠ b} |x_k − x_{k−1}|
  - `sum_other_adjacent_abs_diff` = Σ_{k ≠ b} |x_k − x_{k−1}|
- 服务在**边界、突跳方向、逐段整数应变**上联合裁决，依次最小化
  其余相邻差的最大值、绝对值之和与应变序列字典序；三者完全并列时
  按候选**录入顺序**确定边界。
- 响应给出选中边界、带符号突跳、每个候选的可行性及其自身最优指标，
  并以 `smooth_diff_participation` 逐行列出各边界是否参与平滑指标，
  检修组可直接复核「只有选中边界被排除」。
- 合法配置下若**没有任何**候选边界存在满足全部观测窗与最小突跳的
  整数解释，返回 `409 INFEASIBLE`；候选越界或重复返回 `422` 字段错误。
- 不提供 `slip_joint` 时，请求、响应、三级裁决与错误行为完全不变。

实现方法（`app/solver.py`，仅标准库）：

- 引入前缀和 P_i = Σ_{j<i} L_j·x_j，观测窗与应变界全部化为 P 上的
  **差分约束**，以 Floyd–Warshall 最长路闭包检测正环（冲突）并导出
  每段 x_i 的紧整数域；
- |x_k−x_{k−1}|≤M 同样是差分约束，并入闭包缩域；
- 对线性窗约束做整数界传播 + MRV/折半回溯判定可行性；
- 最小 M、最小 S（辅助变量 e_k≥|Δ_k| 精确松弛）与字典序最优序列
  均利用「可行性关于阈值单调」二分钉死。

## 运行（Docker Compose）

```bash
# 宿主机端口可配置（默认 8000）
HOST_PORT=9000 docker compose up --build -d api
curl -s http://localhost:9000/health
```

### 一次性验证服务 verify

`verify` 服务会等待 `api` 健康检查通过，然后依次运行：

1. 单元测试（pytest，含与全枚举暴力解的三级最优性对照）
2. 构建检查（`compileall`）与 `pip check`
3. 一组反演 API 冒烟，覆盖**新旧两类请求**：默认模式（成功回算、
   `INFEASIBLE`、`INVALID_INPUT`）与滑移接头模式（成功定位、仅选中边界
   排除平滑指标的逐项复核、候选越界/重复 422、全部候选不可行 409）

完成后**自行退出**，退出码即结论（0 为全部通过）：

```bash
docker compose build
docker compose up --abort-on-container-exit --exit-code-from verify verify
echo "verify exit code: $?"
```

## API

### `GET /health`

返回 `{"status": "ok"}`，供容器与 Compose 健康检查使用。

### `GET /api/v1/schema`

返回请求/响应/错误码的字段约定。

### `POST /api/v1/invert`

请求体：

```json
{
  "segment_lengths": [10, 12, 11, 13, 10, 14],
  "strain_bounds": {"min": -100, "max": 100},
  "windows": [
    {"start_segment": 1, "end_segment": 6,
     "min_elongation": 100, "max_elongation": 400}
  ]
}
```

- `segment_lengths`：6–12 个正整数，按顺序排列。
- `strain_bounds`：统一应变闭区间（整数微应变）。
- `windows`：8–20 个观测窗；段号 1 基且含端点，`start ≤ end`，
  `min_elongation ≤ max_elongation`。

成功（200）：

```json
{
  "code": "OK",
  "result": {
    "segment_count": 6,
    "strains": [3, 5, 4, 2, 6, 1],
    "adjacent_diffs": [2, -1, -2, 4, -5],
    "objectives": {
      "max_adjacent_diff": 5,
      "sum_adjacent_abs_diff": 14
    },
    "window_checks": [
      {
        "index": 0,
        "start_segment": 1,
        "end_segment": 6,
        "total_length": 70,
        "min_elongation": 100,
        "max_elongation": 400,
        "weighted_strain_sum": 234,
        "satisfied": true
      }
    ],
    "criteria_order": ["max_adjacent_diff",
                       "sum_adjacent_abs_diff", "lexicographic"]
  }
}
```

每个 `weighted_strain_sum` 都能用提交的 `segment_lengths` 与返回的
`strains` 直接复核并确认落在提交闭区间内；两级平滑指标可用
`adjacent_diffs` 直接复核。

#### 滑移接头模式请求

在默认请求体中追加 `slip_joint`：

```json
"slip_joint": {
  "candidate_boundaries": [2, 5],
  "min_jump": 4
}
```

成功（200）响应的 `result` 在原有字段基础上改为：

```json
{
  "segment_count": 6,
  "strains": [3, 5, 4, 2, 6, 1],
  "adjacent_diffs": [2, -1, -2, 4, -5],
  "slip_joint": {
    "selected_boundary": 5,
    "signed_jump": -5,
    "min_jump": 4,
    "jump_satisfied": true,
    "candidates": [
      {"boundary": 2, "feasible": false},
      {"boundary": 5, "feasible": true,
       "objectives": {"max_other_adjacent_diff": 4,
                      "sum_other_adjacent_abs_diff": 9}}
    ]
  },
  "objectives": {
    "max_other_adjacent_diff": 4,
    "sum_other_adjacent_abs_diff": 9
  },
  "smooth_diff_participation": [
    {"boundary": 1, "adjacent_diff": 2,  "participates_in_smoothness": true},
    {"boundary": 2, "adjacent_diff": -1, "participates_in_smoothness": true},
    {"boundary": 3, "adjacent_diff": -2, "participates_in_smoothness": true},
    {"boundary": 4, "adjacent_diff": 4,  "participates_in_smoothness": true},
    {"boundary": 5, "adjacent_diff": -5, "participates_in_smoothness": false}
  ],
  "window_checks": [ /* 同默认模式，逐条可复核 */ ],
  "criteria_order": ["max_other_adjacent_diff",
                     "sum_other_adjacent_abs_diff",
                     "lexicographic", "candidate_input_order"]
}
```

复核要点：`smooth_diff_participation` 中恰有选中边界一行
`participates_in_smoothness=false`；`objectives` 两项恰好等于其余
`n-2` 个相邻差的最大绝对值与绝对值之和；`signed_jump` 等于
`adjacent_diffs[selected_boundary-1]` 且绝对值 ≥ `min_jump`。

错误：

- `400`：请求体不是合法 JSON。
- `422 INVALID_INPUT`：字段错误，`fields[]` 逐条给出 `field` 与 `message`。
  滑移模式下候选数量不是 2–6 个、候选越界（不在 `1..n-1`）、候选重复、
  `min_jump` 不是正整数等均在此返回。
- `409 INFEASIBLE`：输入合法但不存在满足全部闭区间的整数应变序列；
  滑移模式下表示**所有**候选边界都无法同时满足全部观测窗与最小突跳。

## 本地开发（无 Docker）

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q
.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000
API_BASE_URL=http://127.0.0.1:8000 .venv/bin/python scripts/smoke.py
```
