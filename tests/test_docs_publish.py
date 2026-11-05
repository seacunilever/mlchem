import importlib.util
from pathlib import Path

import pytest


_PUBLISH_PATH = Path(__file__).resolve().parents[1] / "docs" / "_publish.py"
_PUBLISH_SPEC = importlib.util.spec_from_file_location("docs_publish", _PUBLISH_PATH)
assert _PUBLISH_SPEC is not None
assert _PUBLISH_SPEC.loader is not None
publish = importlib.util.module_from_spec(_PUBLISH_SPEC)
_PUBLISH_SPEC.loader.exec_module(publish)


def _configure_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    docs = tmp_path / "docs"
    source = docs / "build" / "html"
    source.mkdir(parents=True)
    monkeypatch.setattr(publish, "DOCS", docs)
    monkeypatch.setattr(publish, "SRC", source)
    return docs, source


def test_main_skips_unchanged_files_without_removing_published_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    docs, source = _configure_paths(tmp_path, monkeypatch)
    relative_path = Path("_modules") / "mlchem" / "module.html"
    source_file = source / relative_path
    published_file = docs / relative_path
    source_file.parent.mkdir(parents=True)
    published_file.parent.mkdir(parents=True)
    source_file.write_text("unchanged", encoding="utf-8")
    published_file.write_text("unchanged", encoding="utf-8")

    def fail_if_removed(path: Path) -> None:
        raise PermissionError(f"simulated lock: {path}")

    monkeypatch.setattr(publish.shutil, "rmtree", fail_if_removed)

    assert publish.main() == 0
    assert published_file.read_text(encoding="utf-8") == "unchanged"
    assert (docs / ".nojekyll").is_file()


def test_main_updates_files_and_removes_stale_assets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    docs, source = _configure_paths(tmp_path, monkeypatch)
    (source / "index.html").write_text("new page", encoding="utf-8")
    (docs / "index.html").write_text("old page", encoding="utf-8")
    (docs / "stale.html").write_text("stale", encoding="utf-8")

    assert publish.main() == 0
    assert (docs / "index.html").read_text(encoding="utf-8") == "new page"
    assert not (docs / "stale.html").exists()


def test_main_reports_permission_errors_with_recovery_guidance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    docs, source = _configure_paths(tmp_path, monkeypatch)
    source_file = source / "index.html"
    published_file = docs / "index.html"
    source_file.write_text("new page", encoding="utf-8")
    published_file.write_text("old page", encoding="utf-8")

    def fail_to_copy(source_path: Path, destination_path: Path) -> None:
        raise PermissionError(32, "file is in use", str(destination_path))

    monkeypatch.setattr(publish.shutil, "copy2", fail_to_copy)

    assert publish.main() == 1
    assert "Close applications using the published documentation" in capsys.readouterr().err
    assert published_file.read_text(encoding="utf-8") == "old page"
