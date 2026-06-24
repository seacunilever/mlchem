#!/usr/bin/env python
"""Run reproducible install+test checks across local Python envs.

Default env roots are under C:/Users/Leonardo.Contreas/Envs.
This script is intentionally conservative:
- 3.12 and 3.13 are required to pass.
- 3.14 is reported as experimental by default.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENVS = {
    "3.12": Path(r"C:/Users/Leonardo.Contreas/Envs/mlchemenv312"),
    "3.13": Path(r"C:/Users/Leonardo.Contreas/Envs/mlchemenv313"),
    "3.14": Path(r"C:/Users/Leonardo.Contreas/Envs/mlchemenv314"),
}


@dataclass
class EnvResult:
    version: str
    status: str
    detail: str


def _python_path(env_root: Path) -> Path:
    return env_root / "Scripts" / "python.exe"


def _run(
    cmd: list[str],
    cwd: Path,
    live_output: bool,
) -> subprocess.CompletedProcess[str]:
    if not live_output:
        return subprocess.run(cmd, cwd=str(cwd), text=True, capture_output=True)

    process = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )

    output_lines: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        output_lines.append(line)
        print(line, end="")

    returncode = process.wait()
    output = "".join(output_lines)
    return subprocess.CompletedProcess(args=cmd, returncode=returncode, stdout=output, stderr="")


def _run_and_check(
    cmd: list[str],
    cwd: Path,
    label: str,
    live_output: bool,
    version: str,
) -> tuple[bool, str]:
    if live_output:
        print(f"\n[{version}] {label}: {' '.join(cmd)}")

    proc = _run(cmd, cwd, live_output=live_output)
    if proc.returncode == 0:
        return True, ""

    tail = (proc.stderr or proc.stdout or "").strip().splitlines()
    detail = "\n".join(tail[-12:]) if tail else f"{label} failed with exit code {proc.returncode}"
    return False, detail


def run_matrix(
    versions: Iterable[str],
    pytest_args: list[str],
    allow_314_failure: bool,
    skip_install: bool,
    live_output: bool,
) -> list[EnvResult]:
    results: list[EnvResult] = []

    for version in versions:
        env_root = DEFAULT_ENVS[version]
        py = _python_path(env_root)

        if not py.exists():
            results.append(EnvResult(version, "missing", f"Interpreter not found: {py}"))
            continue

        if live_output:
            mode = "skip-install" if skip_install else "full-install"
            print(f"\n===== Python {version} ({mode}) =====")

        if not skip_install:
            ok, detail = _run_and_check(
                [str(py), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"],
                REPO_ROOT,
                "bootstrap",
                live_output,
                version,
            )
            if not ok:
                results.append(EnvResult(version, "fail", f"pip bootstrap failed\n{detail}"))
                continue

            ok, detail = _run_and_check(
                [str(py), "-m", "pip", "install", "-r", "requirements.txt"],
                REPO_ROOT,
                "deps",
                live_output,
                version,
            )
            if not ok:
                status = "warn" if version == "3.14" and allow_314_failure else "fail"
                results.append(EnvResult(version, status, f"dependency installation failed\n{detail}"))
                continue

            ok, detail = _run_and_check(
                [str(py), "-m", "pip", "install", "-e", "."],
                REPO_ROOT,
                "editable",
                live_output,
                version,
            )
            if not ok:
                status = "warn" if version == "3.14" and allow_314_failure else "fail"
                results.append(EnvResult(version, status, f"editable install failed\n{detail}"))
                continue

        smoke = [
            str(py),
            "-c",
            (
                "import mlchem; "
                "import numpy, pandas, scipy, sklearn, matplotlib; "
                "print('smoke-ok')"
            ),
        ]
        ok, detail = _run_and_check(smoke, REPO_ROOT, "smoke", live_output, version)
        if not ok:
            status = "warn" if version == "3.14" and allow_314_failure else "fail"
            results.append(EnvResult(version, status, f"smoke import failed\n{detail}"))
            continue

        test_cmd = [str(py), "-m", "pytest"] + pytest_args
        ok, detail = _run_and_check(test_cmd, REPO_ROOT, "tests", live_output, version)
        if ok:
            results.append(EnvResult(version, "pass", ""))
        else:
            status = "warn" if version == "3.14" and allow_314_failure else "fail"
            results.append(EnvResult(version, status, f"pytest failed\n{detail}"))

    return results


def print_summary(results: list[EnvResult]) -> None:
    print("\nMatrix summary")
    print("-" * 78)
    print(f"{'Python':<10} {'Status':<10} Detail")
    print("-" * 78)
    for item in results:
        first_line = item.detail.splitlines()[0] if item.detail else ""
        print(f"{item.version:<10} {item.status:<10} {first_line}")
    print("-" * 78)

    issues = [r for r in results if r.status in {"fail", "warn"} and r.detail]
    if issues:
        print("\nDetails")
        print("-" * 78)
        for item in issues:
            print(f"[{item.version}] {item.status}")
            print(item.detail)
            print("-" * 78)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run local compatibility matrix checks.")
    parser.add_argument(
        "--versions",
        nargs="+",
        choices=sorted(DEFAULT_ENVS.keys()),
        default=["3.12", "3.13", "3.14"],
        help="Python versions to run.",
    )
    parser.add_argument(
        "--allow-314-failure",
        action="store_true",
        default=True,
        help="Treat Python 3.14 failures as warnings.",
    )
    parser.add_argument(
        "--strict-314",
        action="store_true",
        help="Override and fail when 3.14 fails.",
    )
    parser.add_argument(
        "--full-install",
        action="store_true",
        help="Run pip install bootstrap and dependency installation before smoke/tests.",
    )
    parser.add_argument(
        "--skip-install",
        action="store_true",
        help="Deprecated compatibility flag. Install steps are skipped by default.",
    )
    parser.add_argument(
        "--live-output",
        action="store_true",
        help="Stream command output as each env and step runs.",
    )
    parser.add_argument(
        "pytest_args",
        nargs=argparse.REMAINDER,
        help="Arguments passed to pytest (example: -- -q tests).",
    )
    args = parser.parse_args()

    allow_314_failure = False if args.strict_314 else args.allow_314_failure
    pytest_args = args.pytest_args if args.pytest_args else ["-q", "tests"]
    if pytest_args and pytest_args[0] == "--":
        pytest_args = pytest_args[1:]

    skip_install = not args.full_install

    results = run_matrix(
        versions=args.versions,
        pytest_args=pytest_args,
        allow_314_failure=allow_314_failure,
        skip_install=skip_install,
        live_output=args.live_output,
    )
    print_summary(results)

    hard_fail = any(r.status == "fail" for r in results)
    return 1 if hard_fail else 0


if __name__ == "__main__":
    sys.exit(main())
