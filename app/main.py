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
            "slip_joint": "(可选) 护套滑移接头定位，缺省时行为与原接口完全一致",
            "slip_joint_fields": {
                "candidate_boundaries": "2..6 个互异段间边界（1 基：b 位于段 b 与段 b+1 之间，取值 1..n-1），按录入顺序裁决并列",
                "min_jump": "正整数；选中边界的带符号应变差绝对值必须不小于该值",
            },
        },
        "success_response": {
            "segment_count": "int",
            "strains": "逐段整数微应变（字典序最优）",
            "adjacent_diffs": "逐相邻段应变差，可直接复核两级平滑指标",
            "objectives": {
                "max_adjacent_diff": "（默认模式）第一级指标 = max(|adjacent_diffs|)",
                "sum_adjacent_abs_diff": "（默认模式）第二级指标 = sum(|adjacent_diffs|)",
                "max_other_adjacent_diff": "（滑移模式）排除选中边界后其余相邻差的最大绝对值",
                "sum_other_adjacent_abs_diff": "（滑移模式）排除选中边界后其余相邻差绝对值之和",
            },
            "slip_joint": {
                "selected_boundary": "唯一选中的候选段间边界（1 基）",
                "signed_jump": "选中边界带符号突跳 = x_{b+1}-x_b，绝对值 >= min_jump",
                "min_jump": "回显最小突跳阈值",
                "jump_satisfied": "带符号突跳绝对值是否达标（必为 true）",
                "candidates": [
                    {
                        "boundary": "候选边界（1 基，按录入顺序）",
                        "feasible": "该边界是否存在满足全部观测窗与最小突跳的解释",
                        "objectives": "feasible=true 时给出该边界自身最优 (max_other_adjacent_diff, sum_other_adjacent_abs_diff)",
                    }
                ],
            },
            "smooth_diff_participation": [
                {
                    "boundary": "段间边界（1 基）",
                    "adjacent_diff": "该边界相邻差",
                    "participates_in_smoothness": "仅选中边界为 false，其余全部为 true，可直接复核排除范围",
                }
            ],
            "window_checks": [
                {
                    "weighted_strain_sum": "= Σ 窗内长度×应变，须落入提交的闭区间",
                    "total_length": "= Σ 窗内段长",
                    "satisfied": "min<=weighted_strain_sum<=max",
                }
            ],
            "criteria_order_default": [
                "max_adjacent_diff",
                "sum_adjacent_abs_diff",
                "lexicographic",
            ],
            "criteria_order_slip": [
                "max_other_adjacent_diff",
                "sum_other_adjacent_abs_diff",
                "lexicographic",
                "candidate_input_order",
            ],
        },
        "errors": {
            "INVALID_INPUT": {"fields": [{"field": "字段路径", "message": "原因"}]},
            "INFEASIBLE": {
                "default": "观测窗彼此冲突，不存在满足全部闭区间的整数应变序列",
                "slip_joint": "合法配置下没有任何候选边界存在满足全部观测窗与最小突跳的解释",
            },
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
