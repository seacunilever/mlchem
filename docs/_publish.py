"""Mirror ``docs/build/html/`` into ``docs/`` so GitHub Pages serves the latest build.

Run automatically by the ``html`` target in ``docs/Makefile`` and ``docs/make.bat``.

Behaviour:
* Removes every entry currently sitting at the root of ``docs/`` *except* the
  files/folders listed in ``KEEP`` (build inputs and tooling).
* Copies the freshly built site from ``docs/build/html/`` to ``docs/``.
* Keeps the ``.nojekyll`` marker so GitHub Pages does not strip ``_static`` /
  ``_sources`` directories.

The script is idempotent: re-running it after a build always yields the same
mirror.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

DOCS = Path(__file__).resolve().parent
SRC = DOCS / "build" / "html"

# Files/directories at the root of ``docs/`` that must never be deleted by the
# publish step. Anything not in this set is treated as a stale published asset
# and replaced from the fresh build.
KEEP = {
    "source",
    "build",
    "Makefile",
    "make.bat",
    "_publish.py",
    ".nojekyll",
    ".gitignore",
}


def main() -> int:
    if not SRC.exists():
        print(
            f"[publish] No HTML build found at {SRC}.\n"
            "          Run `make html` (or the equivalent sphinx-build command) first.",
            file=sys.stderr,
        )
        return 1

    # 1) Wipe stale published assets at docs/ root.
    for entry in DOCS.iterdir():
        if entry.name in KEEP:
            continue
        if entry.is_dir():
            shutil.rmtree(entry)
        else:
            entry.unlink()

    # 2) Copy the fresh build over.
    for entry in SRC.iterdir():
        target = DOCS / entry.name
        if entry.is_dir():
            shutil.copytree(entry, target)
        else:
            shutil.copy2(entry, target)

    # 3) Ensure the GitHub Pages marker is present.
    nojekyll = DOCS / ".nojekyll"
    if not nojekyll.exists():
        nojekyll.touch()

    print(f"[publish] Mirrored {SRC} -> {DOCS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
