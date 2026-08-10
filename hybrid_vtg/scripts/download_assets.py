"""Download Hybrid VTG datasets and checkpoints into one assets tree."""

from __future__ import annotations

import sys

from hybrid_vtg.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["download", *sys.argv[1:]]))
