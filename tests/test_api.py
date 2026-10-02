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


def slip_payload():
    # 真值 [0,0,0,5,5,5]：边界 3 为 +5 突跳，其余相邻差为 0。
    truth = [0, 0, 0, 5, 5, 5]
    lengths = [10, 12, 11, 13, 10, 14]
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
        "slip": {"candidate_boundaries": [1, 3, 5], "min_jump": 5},
    }


def test_invert_slip_success_and_recompute():
    payload = slip_payload()
    resp = client.post("/api/v1/invert", json=payload)
    assert resp.status_code == 200, resp.text
    result = resp.json()["result"]
    slip = result["slip"]
    assert slip["selected_boundary"] == 3
    assert slip["signed_jump"] == 5
    assert slip["jump_direction"] == 1
    assert abs(slip["signed_jump"]) >= payload["slip"]["min_jump"]
    # 只有选中边界被排除在平滑指标之外。
    assert slip["excluded_from_smoothness"] == [3]
    assert slip["smoothness_boundaries"] == [1, 2, 4, 5]
    strains = result["strains"]
    diffs = [strains[i] - strains[i - 1] for i in range(1, 6)]
    kept = [diffs[k - 1] for k in (1, 2, 4, 5)]
    assert result["objectives"]["max_adjacent_diff"] == max(abs(d) for d in kept)
    assert result["objectives"]["sum_adjacent_abs_diff"] == sum(abs(d) for d in kept)
    assert result["objectives"] == {"max_adjacent_diff": 0, "sum_adjacent_abs_diff": 0}
    # 每个观测窗仍可由提交数据复核。
    for check, win in zip(result["window_checks"], payload["windows"]):
        total = sum(
            payload["segment_lengths"][i] * strains[i]
            for i in range(win["start_segment"] - 1, win["end_segment"])
        )
        assert check["weighted_strain_sum"] == total
        assert win["min_elongation"] <= total <= win["max_elongation"]
    # 候选报告：只有边界 3 可行。
    reports = {c["boundary"]: c for c in slip["candidates"]}
    assert sorted(reports) == [1, 3, 5]
    assert reports[3]["feasible"] is True
    assert reports[1]["feasible"] is False
    assert reports[5]["feasible"] is False


def test_invert_slip_invalid_boundary_422():
    payload = slip_payload()
    payload["slip"] = {"candidate_boundaries": [0, 6], "min_jump": 1}
    resp = client.post("/api/v1/invert", json=payload)
    assert resp.status_code == 422
    fields = {f["field"] for f in resp.json()["fields"]}
    assert "slip.candidate_boundaries[0]" in fields
    assert "slip.candidate_boundaries[1]" in fields


def test_invert_slip_duplicate_boundary_422():
    payload = slip_payload()
    payload["slip"] = {"candidate_boundaries": [3, 3], "min_jump": 1}
    resp = client.post("/api/v1/invert", json=payload)
    assert resp.status_code == 422
    assert "slip.candidate_boundaries[1]" in {
        f["field"] for f in resp.json()["fields"]
    }


def test_invert_slip_nonpositive_jump_422():
    payload = slip_payload()
    payload["slip"] = {"candidate_boundaries": [1, 3], "min_jump": 0}
    resp = client.post("/api/v1/invert", json=payload)
    assert resp.status_code == 422
    assert "slip.min_jump" in {f["field"] for f in resp.json()["fields"]}


def test_invert_slip_infeasible_409():
    payload = slip_payload()
    payload["slip"]["min_jump"] = 6  # 唯一突跳只有 5，无任何候选可行
    resp = client.post("/api/v1/invert", json=payload)
    assert resp.status_code == 409
    assert resp.json()["code"] == "INFEASIBLE"


def test_legacy_request_without_slip_unchanged():
    payload = slip_payload()
    del payload["slip"]
    resp = client.post("/api/v1/invert", json=payload)
    assert resp.status_code == 200
    result = resp.json()["result"]
    assert "slip" not in result
    # 旧行为：突跳边界也参与平滑指标。
    assert result["objectives"]["max_adjacent_diff"] == 5
