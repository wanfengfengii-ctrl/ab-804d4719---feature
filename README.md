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
3. 一组反演 API 冒烟（成功回算、`INFEASIBLE`、`INVALID_INPUT`）

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
- `slip`（**可选**）：护套滑移接头配置。启用后在同一批标距中联合定位
  **唯一一个**滑移接头：
  - `candidate_boundaries`：2–6 个**互异**段间边界（边界 k = 第 k 段与
    第 k+1 段之间，1 基，取值 1..段数-1）；
  - `min_jump`：正整数；选中边界上的应变差绝对值不得低于该阈值
    （闭区间，恰好等于也合法）。

  启用后的裁决模型：必须选择且只选择一个候选边界，该边界的应变差
  满足 |x_k − x_{k−1}| ≥ `min_jump`，且**不参与**两级平滑指标；其余
  相邻差仍全部参与。在「边界 × 突跳方向（上升/下降）× 逐段整数应变」
  上联合裁决，依次最小化

  1. 其余相邻差最大绝对值 `max_adjacent_diff`；
  2. 其余相邻差绝对值之和 `sum_adjacent_abs_diff`；
  3. 应变序列 (x_0, …, x_{n−1}) 的字典序；

  三者完全并列时按候选**录入顺序**取靠前者。未提供 `slip` 时，请求、
  响应、三级裁决与错误行为与旧版完全一致（响应中不出现 `slip` 字段）。
  候选越界或重复返回 422 字段错误；合法配置下若不存在满足全部观测窗
  及最小突跳的解释，返回 409 `INFEASIBLE`。

  启用滑移时成功响应额外包含 `result.slip`：

  ```json
  {
    "selected_boundary": 3,
    "selected_boundary_entry_order": 1,
    "signed_jump": 5,
    "jump_direction": 1,
    "min_jump": 5,
    "excluded_from_smoothness": [3],
    "smoothness_boundaries": [1, 2, 4, 5],
    "tie_break": "candidate entry order",
    "candidates": [
      {"boundary": 1, "entry_order": 0, "feasible": false,
       "directions": [{"direction": 1, "feasible": false},
                      {"direction": -1, "feasible": false}]},
      {"boundary": 3, "entry_order": 1, "feasible": true,
       "max_adjacent_diff": 0, "sum_adjacent_abs_diff": 0,
       "optimal_directions": [1],
       "directions": [
         {"direction": 1, "feasible": true,
          "max_adjacent_diff": 0, "sum_adjacent_abs_diff": 0},
         {"direction": -1, "feasible": false}]}
    ]
  }
  ```

  其中 `signed_jump = x_k − x_{k−1}`（带符号，可直接由 `strains`
  复核），`excluded_from_smoothness` 恰为选中边界，
  `smoothness_boundaries` 为参与两级平滑指标的其余全部边界——检修组
  可直接据此复核只有选中边界被排除。

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

错误：

- `400`：请求体不是合法 JSON。
- `422 INVALID_INPUT`：字段错误，`fields[]` 逐条给出 `field` 与 `message`。
- `409 INFEASIBLE`：输入合法但观测窗彼此冲突（含与统一应变界冲突），
  不存在满足全部闭区间的整数应变序列；启用 `slip` 时还包括没有任何
  候选边界能同时满足全部观测窗与最小突跳。

## 本地开发（无 Docker）

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q
.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000
API_BASE_URL=http://127.0.0.1:8000 .venv/bin/python scripts/smoke.py
```
