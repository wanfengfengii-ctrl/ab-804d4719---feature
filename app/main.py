"""海缆整数微应变联合反演 HTTP API。"""

from __future__ import annotations

import os

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .solver import InfeasibleError, ValidationErrors, invert_payload

app = FastAPI(
    title="Submarine Cable Integer Strain Inversion API",
    version="1.0.0",
    description=(
        "对多个重叠标距的累计伸长量观测窗做精确整数联合反演，"
        "依次最小化相邻段最大应变差、相邻差绝对值之和与应变序列字典序。"
    ),
)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.get("/api/v1/schema")
async def schema_hint() -> dict:
    """返回输入输出字段约定，便于调用方自查。"""
    return {
        "request": {
            "segment_lengths": "6..12 个正整数，按顺序排列的缆段长度",
            "strain_bounds": {"min": "整数微应变闭区间下端", "max": "上端"},
            "windows": [
                {
                    "start_segment": "起始段（1 基，含）",
                    "end_segment": "结束段（1 基，含，不小于 start_segment）",
                    "min_elongation": "累计伸长量闭区间下端（整数）",
                    "max_elongation": "上端（整数，不小于下端）",
                }
            ],
            "window_count": "8..20",
            "slip (可选)": {
                "candidate_boundaries": "2..6 个互异段间边界（边界 k = 第 k 段与第 k+1 段之间，1 基，取值 1..段数-1）",
                "min_jump": "正整数；选中边界的应变差绝对值不得低于该阈值",
                "_说明": (
                    "启用后恰选一个候选边界作为护套滑移接头：该边界不参与两级平滑指标，"
                    "其余相邻差全部参与；在边界、突跳方向与逐段整数应变间联合裁决，"
                    "依次最小化其余相邻差的最大绝对值、绝对值之和与应变序列字典序，"
                    "完全并列按候选录入顺序"
                ),
            },
        },
        "success_response": {
            "segment_count": "int",
            "strains": "逐段整数微应变（字典序最优）",
            "adjacent_diffs": "逐相邻段应变差，可直接复核两级平滑指标",
            "objectives": {
                "max_adjacent_diff": "第一级指标 = max(|其余相邻差|)；滑移模式排除选中边界",
                "sum_adjacent_abs_diff": "第二级指标 = sum(|其余相邻差|)；滑移模式排除选中边界",
            },
            "slip (仅启用滑移时出现)": {
                "selected_boundary": "选中的段间边界（1 基间隙号）",
                "signed_jump": "选中边界上的带符号突跳 = x_k - x_{k-1}，|signed_jump| >= min_jump",
                "jump_direction": "突跳方向：+1 上升 / -1 下降",
                "excluded_from_smoothness": "被排除在两级平滑指标之外的边界（恰为选中边界）",
                "smoothness_boundaries": "参与两级平滑指标的其余全部边界",
                "candidates": [
                    {
                        "boundary": "候选边界（1 基，第 k 段与第 k+1 段之间）",
                        "entry_order": "候选录入顺序（从 0 起，完全并列时的裁决依据）",
                        "feasible": "该边界是否存在满足全部观测窗与最小突跳的解释",
                        "max_adjacent_diff": "该边界可行解释的第一级最优指标（可行时）",
                        "sum_adjacent_abs_diff": "该边界可行解释的第二级最优指标（可行时）",
                        "optimal_directions": "达到该边界最优两级指标的突跳方向列表",
                        "directions": [
                            {
                                "direction": "突跳方向 +1（x_k-x_{k-1}>=min_jump）/ -1",
                                "feasible": "该方向是否可行",
                                "max_adjacent_diff": "该方向可行时的第一级最优指标",
                                "sum_adjacent_abs_diff": "该方向可行时的第二级最优指标",
                            }
                        ],
                    }
                ],
            },
            "window_checks": [
                {
                    "weighted_strain_sum": "= Σ 窗内长度×应变，须落入提交的闭区间",
                    "total_length": "= Σ 窗内段长",
                    "satisfied": "min<=weighted_strain_sum<=max",
                }
            ],
            "criteria_order": [
                "max_adjacent_diff",
                "sum_adjacent_abs_diff",
                "lexicographic",
            ],
        },
        "errors": {
            "INVALID_INPUT": {"fields": [{"field": "字段路径", "message": "原因"}]},
            "INFEASIBLE": (
                "观测窗彼此冲突，不存在满足全部闭区间的整数应变序列；"
                "启用滑移时还包括：没有任何候选边界能同时满足全部观测窗与最小突跳"
            ),
        },
    }


@app.post("/api/v1/invert")
async def invert(request: Request) -> JSONResponse:
    try:
        payload = await request.json()
    except Exception:
        return JSONResponse(
            status_code=400,
            content={
                "code": "INVALID_INPUT",
                "message": "request body must be valid JSON",
                "fields": [{"field": ".", "message": "invalid JSON body"}],
            },
        )
    try:
        result = invert_payload(payload)
    except ValidationErrors as exc:
        return JSONResponse(
            status_code=422,
            content={
                "code": "INVALID_INPUT",
                "message": "input validation failed",
                "fields": exc.fields,
            },
        )
    except InfeasibleError as exc:
        return JSONResponse(
            status_code=409,
            content={"code": "INFEASIBLE", "message": str(exc)},
        )
    return JSONResponse(status_code=200, content={"code": "OK", "result": result})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=os.environ.get("APP_HOST", "0.0.0.0"),
        port=int(os.environ.get("APP_PORT", "8000")),
    )
