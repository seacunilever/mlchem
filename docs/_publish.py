"""Mirror ``docs/build/html/`` into ``docs/`` so GitHub Pages serves the latest build.

Run automatically by the ``html`` target in ``docs/Makefile`` and ``docs/make.bat``.

Behaviour:
* Synchronizes the freshly built site from ``docs/build/html/`` into ``docs/``.
* Skips unchanged files and removes stale generated assets while preserving
  the files/folders listed in ``KEEP`` (build inputs and tooling).
* Keeps the ``.nojekyll`` marker so GitHub Pages does not strip ``_static`` /
  ``_sources`` / ``_modules`` directories.

The script is idempotent: re-running it after a build always yields the same
mirror.
"""
from __future__ import annotations

import filecmp
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


def _remove_path(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()


def _sync_tree(source: Path, destination: Path, preserve: set[str] | None = None) -> None:
    if destination.is_symlink():
        _remove_path(destination)
    elif destination.exists() and not destination.is_dir():
        _remove_path(destination)

    destination.mkdir(parents=True, exist_ok=True)
    source_entries = {entry.name: entry for entry in source.iterdir()}

    for source_entry in source_entries.values():
        destination_entry = destination / source_entry.name
        if source_entry.is_dir():
            if destination_entry.is_symlink() or (
                destination_entry.exists() and not destination_entry.is_dir()
            ):
                _remove_path(destination_entry)
            _sync_tree(source_entry, destination_entry)
            continue

        if destination_entry.is_symlink() or destination_entry.is_dir():
            _remove_path(destination_entry)
        if destination_entry.is_file() and filecmp.cmp(
            source_entry, destination_entry, shallow=False
        ):
            continue
        shutil.copy2(source_entry, destination_entry)

    preserved_names = preserve or set()
    for destination_entry in destination.iterdir():
        if (
            destination_entry.name not in source_entries
            and destination_entry.name not in preserved_names
        ):
            _remove_path(destination_entry)


def main() -> int:
    if not SRC.exists():
        print(
            f"[publish] No HTML build found at {SRC}.\n"
            "          Run `make html` (or the equivalent sphinx-build command) first.",
            file=sys.stderr,
        )
        return 1

    try:
        _sync_tree(SRC, DOCS, preserve=KEEP)
        (DOCS / ".nojekyll").touch(exist_ok=True)
    except PermissionError as exc:
        blocked_path = exc.filename or "a published documentation path"
        print(
            f"[publish] Could not update {blocked_path}: access was denied or the "
            "file is in use. Close applications using the published documentation, "
            "then rerun `make html`.\n"
            f"          {exc}",
            file=sys.stderr,
        )
        return 1

    print(f"[publish] Mirrored {SRC} -> {DOCS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
