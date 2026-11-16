"""Phase 0 environment audit and CPU micro-benchmark.

Run with the project virtual environment:

    .venv\\Scripts\\python.exe scripts/check_env.py
    .venv\\Scripts\\python.exe scripts/check_env.py --json

The benchmark is deliberately marked *provisional*: the real from-scratch
transformer arrives in Phase 2. Here we only measure raw CPU capability
(matmul throughput) and a GPT-shaped forward/backward step so the Phase 2/3
budgets are based on measured numbers rather than assumptions.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import platform
import shutil
import sys
import time
from ctypes import wintypes
from dataclasses import asdict, dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class _MemoryStatusEx(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def system_memory_gb() -> tuple[float, float] | None:
    """Return (total, available) physical memory in GiB, or None if unavailable."""
    if sys.platform != "win32":
        return None
    status = _MemoryStatusEx()
    status.dwLength = ctypes.sizeof(_MemoryStatusEx)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return None
    gib = 1024**3
    return status.ullTotalPhys / gib, status.ullAvailPhys / gib


def process_rss_mb() -> float | None:
    """Current process working set in MiB (Windows only)."""
    if sys.platform != "win32":
        return None

    class _ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong),
            ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    handle = kernel32.GetCurrentProcess()
    counters = _ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(_ProcessMemoryCounters)
    ok = 0
    for dll_name, fn_name in (
        ("kernel32", "K32GetProcessMemoryInfo"),
        ("psapi", "GetProcessMemoryInfo"),
    ):
        try:
            dll = ctypes.WinDLL(dll_name, use_last_error=True)
        except OSError:
            continue
        fn = getattr(dll, fn_name, None)
        if fn is None:
            continue
        fn.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(_ProcessMemoryCounters),
            wintypes.DWORD,
        ]
        fn.restype = wintypes.BOOL
        ok = fn(handle, ctypes.byref(counters), counters.cb)
        if ok:
            break
    if not ok:
        return None
    return counters.WorkingSetSize / (1024**2)


@dataclass
class EnvironmentReport:
    python_version: str
    python_executable: str
    platform: str
    torch_version: str
    torch_cuda_available: bool
    torch_threads: int
    torch_interop_threads: int
    cpu_count_logical: int
    cpu_count_physical: int | None
    ram_total_gb: float | None
    ram_available_gb: float | None
    disk_total_gb: float
    disk_free_gb: float
    project_root: str
    cache_paths: dict[str, str] = field(default_factory=dict)
    benchmarks: dict[str, float | str] = field(default_factory=dict)


def physical_core_count() -> int | None:
    """Count physical cores via GetLogicalProcessorInformationEx (Windows)."""
    if sys.platform != "win32":
        return None
    relation_processor_core = 0
    length = ctypes.c_ulong(0)
    kernel32 = ctypes.windll.kernel32
    kernel32.GetLogicalProcessorInformationEx(relation_processor_core, None, ctypes.byref(length))
    if length.value == 0:
        return None
    buffer = ctypes.create_string_buffer(length.value)
    if not kernel32.GetLogicalProcessorInformationEx(
        relation_processor_core, buffer, ctypes.byref(length)
    ):
        return None
    count = 0
    offset = 0
    raw = bytes(buffer[: length.value])
    while offset < len(raw):
        relationship, size = (
            int.from_bytes(raw[offset : offset + 4], "little"),
            int.from_bytes(raw[offset + 4 : offset + 8], "little"),
        )
        if size == 0:
            break
        if relationship == relation_processor_core:
            count += 1
        offset += size
    return count or None


def cpu_counts() -> tuple[int, int | None]:
    return os.cpu_count() or 0, physical_core_count()


def apply_thread_policy() -> tuple[int, int]:
    """Return (torch_default_threads, applied_threads).

    Torch on Windows defaults to physical-core count. On a 4C/8T laptop the
    SMT threads still help GEMM throughput, so we default to all logical CPUs
    unless LLM_FOUNDRY_NUM_THREADS overrides it.
    """
    import torch

    default_threads = torch.get_num_threads()
    requested = os.environ.get("LLM_FOUNDRY_NUM_THREADS")
    default_choice = max(1, os.cpu_count() or 1)
    applied = max(1, int(requested)) if requested is not None else default_choice
    torch.set_num_threads(applied)
    return default_threads, torch.get_num_threads()


def configure_cache_env() -> dict[str, str]:
    """Point ML caches into the project so disk usage stays accountable.

    These must be set *before* torch/transformers are imported to take effect.
    """
    paths = {
        "HF_HOME": PROJECT_ROOT / "model_registry" / "hf_cache",
        "TORCH_HOME": PROJECT_ROOT / "model_registry" / "torch_cache",
        "HF_DATASETS_CACHE": PROJECT_ROOT / "model_registry" / "hf_cache" / "datasets",
    }
    for key, path in paths.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault(key, str(path))
    return {key: os.environ[key] for key in paths}


def bench_matmul(size: int = 2048, iters: int = 10, warmup: int = 3) -> dict[str, float]:
    """Measure dense FP32 matmul throughput (GFLOPS) on CPU."""
    import torch

    a = torch.randn(size, size)
    b = torch.randn(size, size)
    for _ in range(warmup):
        a @ b
    start = time.perf_counter()
    for _ in range(iters):
        a @ b
    elapsed = time.perf_counter() - start
    flops = 2.0 * size**3 * iters
    return {
        "size": float(size),
        "iters": float(iters),
        "seconds": elapsed,
        "gflops": flops / elapsed / 1e9,
    }


def _build_provisional_gpt(
    vocab_size: int = 8192,
    d_model: int = 256,
    n_heads: int = 4,
    n_layers: int = 4,
    ctx: int = 128,
):
    """GPT-shaped module used only to size Phase 2/3 budgets.

    Phase 2 replaces this with the tested, from-scratch implementation.
    """
    import torch
    from torch import nn

    class Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.ln1 = nn.LayerNorm(d_model)
            self.qkv = nn.Linear(d_model, 3 * d_model)
            self.proj = nn.Linear(d_model, d_model)
            self.ln2 = nn.LayerNorm(d_model)
            self.mlp = nn.Sequential(
                nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model)
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            b, t, c = x.shape
            q, k, v = self.qkv(self.ln1(x)).chunk(3, dim=-1)
            q, k, v = (z.view(b, t, n_heads, c // n_heads).transpose(1, 2) for z in (q, k, v))
            att = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
            att = att.transpose(1, 2).reshape(b, t, c)
            x = x + self.proj(att)
            return x + self.mlp(self.ln2(x))

    class GPT(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.ctx = ctx
            self.tok = nn.Embedding(vocab_size, d_model)
            self.pos = nn.Embedding(ctx, d_model)
            self.blocks = nn.ModuleList(Block() for _ in range(n_layers))
            self.ln_f = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, vocab_size, bias=False)
            self.head.weight = self.tok.weight

        def forward(self, idx: torch.Tensor) -> torch.Tensor:
            b, t = idx.shape
            x = self.tok(idx) + self.pos(torch.arange(t))[None, :, :]
            for block in self.blocks:
                x = block(x)
            return self.head(self.ln_f(x))

    return GPT()


def bench_train_step(
    batch: int = 8, seq: int = 128, iters: int = 5, warmup: int = 2
) -> dict[str, float | str]:
    """Measure a provisional 5M-param GPT fwd+bwd step (tokens/s, ms/step)."""
    import torch

    torch.manual_seed(0)
    model = _build_provisional_gpt()
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    x = torch.randint(0, 8192, (batch, seq))
    y = torch.randint(0, 8192, (batch, seq))
    loss_fn = torch.nn.CrossEntropyLoss()

    def step() -> float:
        opt.zero_grad(set_to_none=True)
        logits = model(x)
        loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()
        opt.step()
        return loss.detach().item()

    for _ in range(warmup):
        step()
    start = time.perf_counter()
    loss = 0.0
    for _ in range(iters):
        loss = step()
    elapsed = time.perf_counter() - start
    tokens = batch * seq * iters
    return {
        "params_millions": n_params / 1e6,
        "batch": float(batch),
        "seq": float(seq),
        "iters": float(iters),
        "seconds": elapsed,
        "ms_per_step": elapsed / iters * 1000,
        "tokens_per_second": tokens / elapsed,
        "initial_loss": loss,
        "note": "provisional architecture; replaced by Phase 2 implementation",
    }


def build_report(run_benchmarks: bool = True) -> EnvironmentReport:
    cache_paths = configure_cache_env()
    import torch

    logical, physical = cpu_counts()
    default_threads, applied_threads = apply_thread_policy()
    mem = system_memory_gb()
    disk = shutil.disk_usage(PROJECT_ROOT)

    report = EnvironmentReport(
        python_version=sys.version.split()[0],
        python_executable=sys.executable,
        platform=platform.platform(),
        torch_version=torch.__version__,
        torch_cuda_available=torch.cuda.is_available(),
        torch_threads=applied_threads,
        torch_interop_threads=torch.get_num_interop_threads(),
        cpu_count_logical=logical,
        cpu_count_physical=physical,
        ram_total_gb=round(mem[0], 2) if mem else None,
        ram_available_gb=round(mem[1], 2) if mem else None,
        disk_total_gb=round(disk.total / 1024**3, 2),
        disk_free_gb=round(disk.free / 1024**3, 2),
        project_root=str(PROJECT_ROOT),
        cache_paths=cache_paths,
    )

    if run_benchmarks:
        mat = bench_matmul()
        train = bench_train_step()
        report.benchmarks = {
            "torch_default_threads": float(default_threads),
            "torch_applied_threads": float(applied_threads),
            "matmul_gflops": round(mat["gflops"], 2),
            "matmul_seconds": round(mat["seconds"], 4),
            "train_step_ms": round(float(train["ms_per_step"]), 1),
            "train_tokens_per_second": round(float(train["tokens_per_second"]), 1),
            "provisional_params_millions": round(float(train["params_millions"]), 2),
            "process_rss_mb": round(process_rss_mb() or 0.0, 1),
        }
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.add_argument("--no-bench", action="store_true", help="skip benchmarks")
    args = parser.parse_args(argv)

    torch_available = True
    try:
        import torch  # noqa: F401
    except ImportError:
        torch_available = False

    if not torch_available:
        print("FAIL: torch is not importable from this interpreter.", file=sys.stderr)
        print(f"interpreter: {sys.executable}", file=sys.stderr)
        return 1

    report = build_report(run_benchmarks=not args.no_bench)

    if args.json:
        print(json.dumps(asdict(report), indent=2))
        return 0

    b = report.benchmarks
    print("LLM Foundry - Phase 0 environment report")
    print(f"  python            : {report.python_version}")
    print(f"  interpreter       : {report.python_executable}")
    print(f"  platform          : {report.platform}")
    print(f"  torch             : {report.torch_version} (cuda={report.torch_cuda_available})")
    print(f"  torch threads     : {report.torch_threads} (interop {report.torch_interop_threads})")
    print(f"  cpu logical/physical: {report.cpu_count_logical}/{report.cpu_count_physical}")
    print(f"  ram total/avail   : {report.ram_total_gb} / {report.ram_available_gb} GiB")
    print(f"  disk total/free   : {report.disk_total_gb} / {report.disk_free_gb} GiB")
    print(f"  project root      : {report.project_root}")
    print("  caches:")
    for key, value in report.cache_paths.items():
        print(f"    {key} = {value}")
    if b:
        print("  benchmarks (measured):")
        print(f"    matmul 2048x2048 fp32 : {b['matmul_gflops']} GFLOPS")
        print(
            f"    provisional GPT step  : {b['train_step_ms']} ms "
            f"({b['train_tokens_per_second']} tokens/s, "
            f"{b['provisional_params_millions']}M params)"
        )
        print(f"    process RSS           : {b['process_rss_mb']:.0f} MiB")
    print(f"  cores usable for math : {math.floor(report.torch_threads)} threads")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
