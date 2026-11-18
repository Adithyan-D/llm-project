"""Pretraining engine: token streams, packed batching, schedule, checkpoints."""

from .checkpoint import (
    CHECKPOINT_VERSION,
    LATEST_FILENAME,
    load_checkpoint,
    resolve_checkpoint,
    save_checkpoint,
)
from .config import SCHEDULES, TrainingConfig
from .data import (
    DEFAULT_SEPARATOR,
    TOKEN_DTYPE,
    TOKEN_STREAM_VERSION,
    Batch,
    BatchCursor,
    PackedSampler,
    SplitStream,
    TokenStreams,
    build_token_streams,
    evaluation_batches,
    sha256_file,
)
from .safety import (
    InsufficientMemoryError,
    TrainingOOMError,
    available_memory_bytes,
    ensure_memory_budget,
    is_oom_error,
)
from .scheduler import LRSchedule
from .trainer import Trainer

__all__ = [
    "CHECKPOINT_VERSION",
    "DEFAULT_SEPARATOR",
    "LATEST_FILENAME",
    "SCHEDULES",
    "TOKEN_DTYPE",
    "TOKEN_STREAM_VERSION",
    "Batch",
    "BatchCursor",
    "InsufficientMemoryError",
    "LRSchedule",
    "PackedSampler",
    "SplitStream",
    "TokenStreams",
    "Trainer",
    "TrainingConfig",
    "TrainingOOMError",
    "available_memory_bytes",
    "build_token_streams",
    "ensure_memory_budget",
    "evaluation_batches",
    "is_oom_error",
    "load_checkpoint",
    "resolve_checkpoint",
    "save_checkpoint",
    "sha256_file",
]
