"""API 层测试：健康检查、成功反演、422 字段错误、409 不可行、坏 JSON。"""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def sample_payload():
    return {
        "segment_lengths": [10, 12, 11, 13, 10, 14],
        "strain_bounds": {"min": -100, "max": 100},
        "windows": [
            {"start_segment": 1, "end_segment": 6, "min_elongation": 60, "max_elongation": 700},
            {"start_segment": 1, "end_segment": 1, "min_elongation": 0, "max_elongation": 1000},
            {"start_segment": 2, "end_segment": 2, "min_elongation": 0, "max_elongation": 1000},
            {"start_segment": 3, "end_segment": 3, "min_elongation": 0, "max_elongation": 1000},
            {"start_segment": 4, "end_segment": 4, "min_elongation": 0, "max_elongation": 1000},
            {"start_segment": 5, "end_segment": 5, "min_elongation": 0, "max_elongation": 1000},
            {"start_segment": 6, "end_segment": 6, "min_elongation": 0, "max_elongation": 1000},
            {"start_segment": 1, "end_segment": 3, "min_elongation": 0, "max_elongation": 2000},
        ],
    }


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_schema_hint_documents_fields():
    resp = client.get("/api/v1/schema")
    assert resp.status_code == 200
    body = resp.json()
    assert "window_checks" in body["success_response"]
    assert "INFEASIBLE" in body["errors"]


def test_invert_success_and_recompute():
    resp = client.post("/api/v1/invert", json=sample_payload())
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["code"] == "OK"
    result = body["result"]
    assert len(result["strains"]) == 6
    assert all(-100 <= v <= 100 for v in result["strains"])
    # 全部回算和均可由提交数据复核且落在提交区间内。
    payload = sample_payload()
    lengths = payload["segment_lengths"]
    strains = result["strains"]
    for check, win in zip(result["window_checks"], payload["windows"]):
        total = sum(
            lengths[i] * strains[i]
            for i in range(win["start_segment"] - 1, win["end_segment"])
        )
        assert check["weighted_strain_sum"] == total
        assert win["min_elongation"] <= total <= win["max_elongation"]
        assert check["satisfied"] is True
    diffs = [strains[i] - strains[i - 1] for i in range(1, 6)]
    assert result["objectives"]["max_adjacent_diff"] == max(abs(d) for d in diffs)
    assert result["objectives"]["sum_adjacent_abs_diff"] == sum(abs(d) for d in diffs)


def test_invert_invalid_fields_422():
    payload = sample_payload()
    payload["segment_lengths"] = [1, 2]  # 少于 6 段
    resp = client.post("/api/v1/invert", json=payload)
    assert resp.status_code == 422
    body = resp.json()
    assert body["code"] == "INVALID_INPUT"
    assert any(f["field"] == "segment_lengths" for f in body["fields"])


def test_invert_infeasible_409():
    payload = sample_payload()
    payload["windows"][0] = {
        "start_segment": 1,
        "end_segment": 6,
        "min_elongation": 10**9,
        "max_elongation": 10**9,
    }
    resp = client.post("/api/v1/invert", json=payload)
    assert resp.status_code == 409
    assert resp.json()["code"] == "INFEASIBLE"


def test_bad_json_400():
    resp = client.post(
        "/api/v1/invert",
        content=b"{not json",
        headers={"content-type": "application/json"},
    )
    assert resp.status_code == 400
    assert resp.json()["code"] == "INVALID_INPUT"


# --------------------------------------------------------------------------- #
# 滑移接头模式
# --------------------------------------------------------------------------- #
def slip_payload():
    lengths = [10, 12, 11, 13, 10, 14]
    truth = [3, 5, 4, 2, 6, 1]  # diffs = [2,-1,-2,4,-5]
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
    }


def test_slip_success_and_only_selected_boundary_excluded():
    payload = slip_payload()
    payload["slip_joint"] = {
        "candidate_boundaries": [2, 5],
        "min_jump": 4,
    }
    resp = client.post("/api/v1/invert", json=payload)
    assert resp.status_code == 200, resp.text
    result = resp.json()["result"]
    slip = result["slip_joint"]
    assert slip["selected_boundary"] == 5
    assert slip["signed_jump"] == -5
    assert slip["jump_satisfied"] is True
    assert slip["min_jump"] == 4

    # 必须且只选择一个候选边界。
    selected = slip["selected_boundary"]
    assert selected in (2, 5)
    reports = slip["candidates"]
    assert [r["boundary"] for r in reports] == [2, 5]  # 保持录入顺序
    by_b = {r["boundary"]: r for r in reports}
    assert by_b[2]["feasible"] is False
    assert "objectives" not in by_b[2]
    assert by_b[5]["feasible"] is True

    # 检修组直接复核：只有选中边界被排除在平滑指标之外。
    participation = result["smooth_diff_participation"]
    excluded = [p["boundary"] for p in participation if not p["participates_in_smoothness"]]
    assert excluded == [selected]
    strains = result["strains"]
    diffs = [strains[i] - strains[i - 1] for i in range(1, 6)]
    other_abs = [abs(d) for k, d in enumerate(diffs) if k != selected - 1]
    assert result["objectives"]["max_other_adjacent_diff"] == max(other_abs)
    assert result["objectives"]["sum_other_adjacent_abs_diff"] == sum(other_abs)

    # 观测窗仍全部满足。
    for check in result["window_checks"]:
        assert check["satisfied"] is True


def test_slip_invalid_boundary_422():
    payload = slip_payload()
    payload["slip_joint"] = {"candidate_boundaries": [0, 6], "min_jump": 1}
    resp = client.post("/api/v1/invert", json=payload)
    assert resp.status_code == 422
    body = resp.json()
    assert body["code"] == "INVALID_INPUT"
    fields = {f["field"] for f in body["fields"]}
    assert any("candidate_boundaries[0]" in f for f in fields)
    assert any("candidate_boundaries[1]" in f for f in fields)


def test_slip_duplicate_boundary_422():
    payload = slip_payload()
    payload["slip_joint"] = {"candidate_boundaries": [3, 3], "min_jump": 2}
    resp = client.post("/api/v1/invert", json=payload)
    assert resp.status_code == 422
    assert any(
        "duplicate" in f["message"] for f in resp.json()["fields"]
    )


def test_slip_wrong_count_422():
    payload = slip_payload()
    payload["slip_joint"] = {"candidate_boundaries": [1], "min_jump": 1}
    resp = client.post("/api/v1/invert", json=payload)
    assert resp.status_code == 422
    assert any("candidate_boundaries" in f["field"] for f in resp.json()["fields"])


def test_slip_nonpositive_jump_422():
    payload = slip_payload()
    payload["slip_joint"] = {"candidate_boundaries": [1, 2], "min_jump": 0}
    resp = client.post("/api/v1/invert", json=payload)
    assert resp.status_code == 422
    assert any("min_jump" in f["field"] for f in resp.json()["fields"])


def test_slip_all_candidates_infeasible_409():
    payload = slip_payload()
    # 序列被单段窗钉死，最大相邻差 5；阈值 6 使两个候选都不可行。
    payload["slip_joint"] = {"candidate_boundaries": [2, 5], "min_jump": 6}
    resp = client.post("/api/v1/invert", json=payload)
    assert resp.status_code == 409
    assert resp.json()["code"] == "INFEASIBLE"


def test_default_response_unchanged_without_slip():
    """未提供 slip_joint 时响应结构保持原样（无新增键）。"""
    resp = client.post("/api/v1/invert", json=sample_payload())
    assert resp.status_code == 200
    result = resp.json()["result"]
    assert "slip_joint" not in result
    assert "smooth_diff_participation" not in result
    assert set(result["objectives"]) == {
        "max_adjacent_diff",
        "sum_adjacent_abs_diff",
    }
