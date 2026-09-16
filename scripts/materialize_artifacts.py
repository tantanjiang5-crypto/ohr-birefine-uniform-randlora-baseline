#!/usr/bin/env python3
"""Reassemble and verify large experiment artifacts downloaded through Git LFS."""
from __future__ import annotations

import hashlib
from pathlib import Path

EXPECTED_SHA256 = "40467b766d2a98d6937a17029b73507a7264fc54fa9b80bf28442b29707f03e5"
EXPECTED_BYTES = 381_792_348


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    artifact_dir = repo / "artifacts"
    parts = sorted(artifact_dir.glob("seed2026_59cls.pth.part-*"))
    if not parts:
        raise SystemExit("no seed2026_59cls.pth parts found; run `git lfs pull`")
    target = artifact_dir / "seed2026_59cls.pth"
    temporary = target.with_suffix(".pth.incomplete")
    digest = hashlib.sha256()
    size = 0
    with temporary.open("wb") as output:
        for part in parts:
            with part.open("rb") as source:
                for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
                    output.write(block)
                    digest.update(block)
                    size += len(block)
    actual = digest.hexdigest()
    if size != EXPECTED_BYTES or actual != EXPECTED_SHA256:
        temporary.unlink(missing_ok=True)
        raise SystemExit(
            f"artifact verification failed: bytes={size}, sha256={actual}"
        )
    temporary.replace(target)
    print(f"PASS {target} bytes={size} sha256={actual}")


if __name__ == "__main__":
    main()
