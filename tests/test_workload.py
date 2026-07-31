"""Workload manifest schema, result classification, and launcher bridge."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from autokernel.workload import (
    WORKLOAD_SCHEMA_VERSION,
    WorkloadError,
    WorkloadManifest,
    load_workload,
)
from autokernel.workload.launcher import (
    build_launcher_command,
    resolve_launcher,
    run_ab,
)
from autokernel.workload.result import (
    GenerationRunResult,
    classify_end_to_end,
    load_generation_result,
    write_generation_result,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKLOADS = REPO_ROOT / "workloads"


def test_load_wan_and_ltx_manifests():
    wan = load_workload(WORKLOADS / "wan_t2v_1.3b_480p.yaml")
    ltx = load_workload(WORKLOADS / "ltx_480p.yaml")
    assert wan.schema_version == WORKLOAD_SCHEMA_VERSION
    assert wan.workload_id == "wan-t2v-1.3b-480p"
    assert wan.model.model_id == "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    assert wan.sampling.height == 480
    assert wan.sampling.width == 832
    assert wan.mode_env is not None
    assert wan.mode_env.for_mode("native")["FASTVIDEO_WAN_FUSIONS"] == "0"
    assert wan.mode_env.for_mode("optimized")["FASTVIDEO_WAN_FUSIONS"] == "1"

    assert ltx.workload_id == "ltx-t2v-480p"
    assert "LTX" in ltx.model.model_id or "ltx" in ltx.model.model_id.lower()
    assert ltx.task == "t2v"
    assert ltx.performance is not None
    assert ltx.performance.min_end_to_end_speedup == pytest.approx(1.01)


def test_workload_roundtrip_json(tmp_path):
    wan = load_workload(WORKLOADS / "wan_t2v_1.3b_480p.yaml")
    path = tmp_path / "wan.json"
    path.write_text(json.dumps(wan.as_dict(), indent=2), encoding="utf-8")
    again = load_workload(path)
    assert again.as_dict() == wan.as_dict()


def test_generation_request_matches_wan_ab_shape():
    wan = load_workload(WORKLOADS / "wan_t2v_1.3b_480p.yaml")
    request = wan.generation_request()
    assert "prompt" in request
    assert request["sampling"]["height"] == 480
    assert request["sampling"]["width"] == 832
    assert request["sampling"]["num_frames"] == 49
    assert request["sampling"]["num_inference_steps"] == 4
    assert request["sampling"]["guidance_scale"] == 5.0
    assert request["sampling"]["seed"] == 1024
    assert request["output"]["return_frames"] is True
    kwargs = wan.generator_kwargs()
    assert kwargs["num_gpus"] == 1
    assert kwargs["text_encoder_cpu_offload"] is True


def test_rejects_secret_fields():
    payload = load_workload(WORKLOADS / "ltx_480p.yaml").as_dict()
    payload["runtime"]["password"] = "nope"
    with pytest.raises(WorkloadError, match="secret fields"):
        WorkloadManifest.from_dict(payload)


def test_requires_prompt_or_prompt_file():
    payload = load_workload(WORKLOADS / "ltx_480p.yaml").as_dict()
    del payload["prompt"]
    with pytest.raises(WorkloadError, match="prompt or prompt_file"):
        WorkloadManifest.from_dict(payload)


def test_prompt_file_resolution(tmp_path):
    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("hello from file\n", encoding="utf-8")
    payload = load_workload(WORKLOADS / "ltx_480p.yaml").as_dict()
    del payload["prompt"]
    payload["prompt_file"] = "prompt.txt"
    manifest = WorkloadManifest.from_dict(payload, source=str(tmp_path / "w.yaml"))
    assert manifest.resolve_prompt(base_dir=tmp_path) == "hello from file"


def test_rejects_unknown_top_level_field():
    payload = load_workload(WORKLOADS / "ltx_480p.yaml").as_dict()
    payload["callable"] = "models.foo:bar"
    with pytest.raises(WorkloadError, match="unknown field"):
        WorkloadManifest.from_dict(payload)


def test_rejects_bad_schema_version():
    payload = load_workload(WORKLOADS / "ltx_480p.yaml").as_dict()
    payload["schema_version"] = 99
    with pytest.raises(WorkloadError, match="unsupported version"):
        WorkloadManifest.from_dict(payload)


def test_result_schema_and_classification(tmp_path):
    native = GenerationRunResult.from_dict(
        {
            "schema_version": 1,
            "mode": "native",
            "status": "ok",
            "workload_id": "wan-t2v-1.3b-480p",
            "model_id": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
            "request": {"prompt": "x"},
            "warmups": 1,
            "runs": 2,
            "wall_seconds": [36.7, 36.6],
            "median_wall_seconds": 36.65,
            "generation_seconds": [30.0, 30.1],
            "peak_memory_mb": [20000.0, 20010.0],
            "environment": {"cuda": "12.8"},
        }
    )
    optimized = GenerationRunResult.from_dict(
        {
            "schema_version": 1,
            "mode": "optimized",
            "status": "ok",
            "workload_id": "wan-t2v-1.3b-480p",
            "model_id": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
            "request": {"prompt": "x"},
            "warmups": 1,
            "runs": 2,
            "wall_seconds": [36.8, 36.5],
            "median_wall_seconds": 36.65,
            "generation_seconds": [30.0, 30.0],
            "peak_memory_mb": [20000.0, 20000.0],
            "environment": {"cuda": "12.8"},
        }
    )
    verdict = classify_end_to_end(native, optimized)
    assert verdict["classification"] == "neutral"

    improved = GenerationRunResult.from_dict(
        {
            **optimized.as_dict(),
            "wall_seconds": [30.0, 30.2],
            "median_wall_seconds": 30.1,
        }
    )
    assert classify_end_to_end(native, improved)["classification"] == "improved"

    path = tmp_path / "native_result.json"
    write_generation_result(native, path)
    loaded = load_generation_result(path)
    assert loaded.median_wall_seconds == pytest.approx(36.65)


def test_build_launcher_command_and_resolve(tmp_path):
    checkout = tmp_path / "FastVideo"
    script = (
        checkout
        / "examples"
        / "inference"
        / "optimizations"
        / "generation_launcher.py"
    )
    script.parent.mkdir(parents=True)
    script.write_text("# stub\n", encoding="utf-8")
    resolved = resolve_launcher(checkout)
    assert resolved == script
    command = build_launcher_command(
        python="python3",
        launcher=script,
        workload=WORKLOADS / "wan_t2v_1.3b_480p.yaml",
        mode="native",
        output_dir=tmp_path / "out",
    )
    assert command[:2] == ["python3", str(script)]
    assert "--mode" in command and "native" in command


def test_run_ab_resume(tmp_path, monkeypatch):
    checkout = tmp_path / "FastVideo"
    script = (
        checkout
        / "examples"
        / "inference"
        / "optimizations"
        / "generation_launcher.py"
    )
    script.parent.mkdir(parents=True)
    script.write_text("# stub\n", encoding="utf-8")
    out = tmp_path / "ab"
    out.mkdir()

    calls: list[str] = []

    def fake_run_mode(**kwargs):
        mode = kwargs["mode"]
        calls.append(mode)
        result = GenerationRunResult.from_dict(
            {
                "schema_version": 1,
                "mode": mode if mode != "fused" else "optimized",
                "status": "ok",
                "workload_id": "wan-t2v-1.3b-480p",
                "model_id": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
                "request": {},
                "warmups": 1,
                "runs": 1,
                "wall_seconds": [10.0 if mode == "native" else 9.0],
                "median_wall_seconds": 10.0 if mode == "native" else 9.0,
                "generation_seconds": [8.0],
                "peak_memory_mb": [1000.0],
                "environment": {},
            }
        )
        write_generation_result(result, out / f"{mode}_result.json")

        class _Done:
            returncode = 0
            stdout = ""
            stderr = ""

        return _Done()

    monkeypatch.setattr(
        "autokernel.workload.launcher.run_mode", fake_run_mode
    )
    first = run_ab(
        fastvideo_checkout=checkout,
        workload=WORKLOADS / "wan_t2v_1.3b_480p.yaml",
        output_dir=out,
        resume=True,
    )
    assert first["comparison"]["classification"] in {"improved", "neutral"}
    assert calls == ["native", "optimized"]

    second = run_ab(
        fastvideo_checkout=checkout,
        workload=WORKLOADS / "wan_t2v_1.3b_480p.yaml",
        output_dir=out,
        resume=True,
    )
    assert calls == ["native", "optimized"]  # no re-run
    assert second["comparison"] is not None


def test_cli_validate(monkeypatch):
    from workload import main

    assert main(["validate", str(WORKLOADS / "ltx_480p.yaml")]) == 0
    assert main(["show", str(WORKLOADS / "wan_t2v_1.3b_480p.yaml")]) == 0


def test_read_state_rejects_corrupt_json(tmp_path):
    from autokernel.workload.launcher import _read_state

    path = tmp_path / "launcher_state.json"
    path.write_text("{not-json", encoding="utf-8")
    with pytest.raises(WorkloadError, match="invalid JSON"):
        _read_state(path)


def test_run_ab_rejects_unknown_modes(tmp_path, monkeypatch):
    from autokernel.workload.launcher import run_ab

    checkout = tmp_path / "FastVideo"
    script = (
        checkout
        / "examples"
        / "inference"
        / "optimizations"
        / "generation_launcher.py"
    )
    script.parent.mkdir(parents=True)
    script.write_text("# stub\n", encoding="utf-8")
    with pytest.raises(WorkloadError, match="unsupported launcher mode"):
        run_ab(
            fastvideo_checkout=checkout,
            workload=WORKLOADS / "wan_t2v_1.3b_480p.yaml",
            output_dir=tmp_path / "out",
            modes=("native", "candidate"),
        )


def test_classify_end_to_end_zero_optimized_median():
    from autokernel.workload.result import GenerationRunResult, classify_end_to_end

    native = GenerationRunResult.from_dict(
        {
            "schema_version": 1,
            "mode": "native",
            "status": "ok",
            "workload_id": "w",
            "model_id": "m",
            "request": {},
            "warmups": 0,
            "runs": 1,
            "wall_seconds": [1.0],
            "median_wall_seconds": 1.0,
            "generation_seconds": [1.0],
            "peak_memory_mb": [1.0],
            "environment": {},
        }
    )
    optimized = GenerationRunResult.from_dict(
        {
            "schema_version": 1,
            "mode": "optimized",
            "status": "ok",
            "workload_id": "w",
            "model_id": "m",
            "request": {},
            "warmups": 0,
            "runs": 1,
            "wall_seconds": [0.0],
            "median_wall_seconds": 0.0,
            "generation_seconds": [0.0],
            "peak_memory_mb": [1.0],
            "environment": {},
        }
    )
    assert classify_end_to_end(native, optimized)["classification"] == "failed"
