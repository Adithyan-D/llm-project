"""Phase 0 acceptance tests: the toolchain must work on this machine.

These tests intentionally assert *capabilities we rely on later* (CPU-only
torch, 3.12 runtime, gradient flow, cache containment), not just imports.
"""

from __future__ import annotations

import json
import shutil
import sys
from dataclasses import asdict
from pathlib import Path

import check_env
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_python_runtime_is_312() -> None:
    assert sys.version_info[:2] == (3, 12), f"unexpected runtime: {sys.version}"


def test_torch_is_cpu_build() -> None:
    assert torch.cuda.is_available() is False
    assert "+cpu" in torch.__version__


def test_thread_policy_applies_all_logical_cpus(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LLM_FOUNDRY_NUM_THREADS", raising=False)
    default_threads, applied = check_env.apply_thread_policy()
    try:
        assert default_threads >= 1
        assert applied == max(1, __import__("os").cpu_count() or 1)
        assert torch.get_num_threads() == applied
    finally:
        torch.set_num_threads(default_threads)


def test_thread_policy_respects_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_FOUNDRY_NUM_THREADS", "2")
    default_threads, applied = check_env.apply_thread_policy()
    try:
        assert applied == 2
    finally:
        torch.set_num_threads(default_threads)


def test_cache_paths_stay_inside_project(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in ("HF_HOME", "TORCH_HOME", "HF_DATASETS_CACHE"):
        monkeypatch.delenv(key, raising=False)
    paths = check_env.configure_cache_env()
    for key, value in paths.items():
        path = Path(value).resolve()
        assert PROJECT_ROOT in path.parents or path == PROJECT_ROOT, f"{key} escapes project"
        assert path.is_dir()


def test_disk_headroom_for_models() -> None:
    free_gb = shutil.disk_usage(PROJECT_ROOT).free / 1024**3
    assert free_gb > 20, f"only {free_gb:.1f} GiB free"


def test_physical_memory_reported() -> None:
    mem = check_env.system_memory_gb()
    if mem is None:
        pytest.skip("memory query unsupported on this platform")
    total_gb, available_gb = mem
    assert total_gb >= 8
    assert 0 <= available_gb <= total_gb


def test_matmul_is_correct_and_finite() -> None:
    a = torch.eye(4)
    b = torch.ones(4, 3)
    out = a @ b
    assert out.shape == (4, 3)
    assert torch.allclose(out, torch.ones(4, 3))
    assert torch.isfinite(out).all()


def test_gradient_flow_through_layernorm_and_linear() -> None:
    torch.manual_seed(0)
    x = torch.randn(2, 5, 8)
    layer = torch.nn.Sequential(torch.nn.LayerNorm(8), torch.nn.Linear(8, 16))
    out = layer(x)
    assert out.shape == (2, 5, 16)
    out.sum().backward()
    for param in layer.parameters():
        assert param.grad is not None
        assert torch.isfinite(param.grad).all()


def test_provisional_benchmark_model_is_in_target_range() -> None:
    model = check_env._build_provisional_gpt()
    n_params = sum(p.numel() for p in model.parameters())
    assert 1_000_000 < n_params < 20_000_000

    idx = torch.randint(0, 8192, (2, 16))
    logits = model(idx)
    assert logits.shape == (2, 16, 8192)


def test_environment_report_is_serializable() -> None:
    report = check_env.build_report(run_benchmarks=False)
    payload = json.dumps(asdict(report))
    assert "python_version" in payload
    assert report.project_root == str(PROJECT_ROOT)
    assert report.benchmarks == {}
