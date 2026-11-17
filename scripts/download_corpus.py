"""Download a small public corpus into datasets/raw with checksum verification.

Corpora are not committed (see .gitignore). Reproduce the Phase 1 datasets with:

    .venv\\Scripts\\python.exe scripts\\download_corpus.py --name tinyshakespeare

Why checksums: dataset versions must be pinned, otherwise tokenizer/dataset hashes
in IMPLEMENTATION_PLAN.md cannot be reproduced later.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CORPORA: dict[str, dict[str, str]] = {
    "tinyshakespeare": {
        "url": "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        "sha256": "86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed",
        "filename": "input.txt",
        "license": "public domain (Shakespeare)",
        "description": "~1.1 MB of Shakespeare plays, 32,777 lines; Tiny Shakespeare corpus",
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", choices=sorted(CORPORA), required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    spec = CORPORA[args.name]
    target_dir = PROJECT_ROOT / "datasets" / "raw" / args.name
    target = target_dir / spec["filename"]

    if target.exists() and not args.force:
        actual = sha256_file(target)
        status = "ok" if actual == spec["sha256"] else "MISMATCH"
        print(f"{target} already exists (sha256 {status})")
        return 0 if status == "ok" else 1

    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"downloading {spec['url']}")
    urllib.request.urlretrieve(spec["url"], target)  # noqa: S310 - fixed https registry

    actual = sha256_file(target)
    if actual != spec["sha256"]:
        target.unlink(missing_ok=True)
        print(f"checksum mismatch: expected {spec['sha256']}, got {actual}", file=sys.stderr)
        return 1

    size = target.stat().st_size
    print(f"wrote {target} ({size:,} bytes, sha256 verified)")
    print(f"license: {spec['license']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
