"""Dataset pipeline: loaders, preprocessing, statistics."""

from .dataset import (
    SPLITS,
    DatasetConfig,
    ProcessingResult,
    iter_split,
    process_dataset,
    read_meta,
)
from .loaders import (
    DatasetFormatError,
    LoadIssue,
    LoadStats,
    detect_format,
    iter_sources,
    iter_text_records,
    load_texts,
)
from .preprocess import (
    Deduplicator,
    NormalizationConfig,
    normalize_text,
    text_fingerprint,
    validate_text,
)
from .stats import SplitStats, merge_stats

__all__ = [
    "SPLITS",
    "DatasetConfig",
    "DatasetFormatError",
    "Deduplicator",
    "LoadIssue",
    "LoadStats",
    "NormalizationConfig",
    "ProcessingResult",
    "SplitStats",
    "detect_format",
    "iter_sources",
    "iter_split",
    "iter_text_records",
    "load_texts",
    "merge_stats",
    "normalize_text",
    "process_dataset",
    "read_meta",
    "text_fingerprint",
    "validate_text",
]
