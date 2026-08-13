"""Normalize Git-tracked UTF-8 text files to LF line endings."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def tracked_files(root: Path) -> tuple[Path, ...]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    return tuple(root / Path(value.decode("utf-8")) for value in result.stdout.split(b"\0") if value)


def normalized_bytes(path: Path) -> bytes | None:
    value = path.read_bytes()
    if b"\0" in value:
        return None
    try:
        value.decode("utf-8")
    except UnicodeDecodeError:
        return None
    return value.replace(b"\r\n", b"\n").replace(b"\r", b"\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="exit nonzero when any file needs normalization")
    parser.add_argument("--dry-run", action="store_true", help="report files without modifying them")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    changed = []
    for path in tracked_files(root):
        replacement = normalized_bytes(path)
        if replacement is None or replacement == path.read_bytes():
            continue
        changed.append(path.relative_to(root))
        if not args.check and not args.dry_run:
            path.write_bytes(replacement)
    for path in changed:
        print(path.as_posix())
    if args.check and changed:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
