#!/usr/bin/env python
# mlchem - cheminformatics library
# Copyright © 2025 as Unilever Global IP Limited

# Redistribution and use in source and binary forms, with or without modification,
# are permitted under the terms of the BSD-3 License, provided that the following conditions are met:

#     1. Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#
#     2. Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in
#        the documentation and/or other materials provided with the distribution.
#
#     3. Neither the name of the copyright holder nor the names of its
#        contributors may be used to endorse or promote products derived
#        from this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS”
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
# THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS
# BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
# GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
# IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

# You should have received a copy of the BSD-3 License along with mlchem.
# If not, see https://interoperable-europe.ec.europa.eu/licence/bsd-3-clause-new-or-revised-license .
# It is the responsibility of mlchem users to familiarise themselves with all dependencies and their associated licenses.

"""Run reproducible install+test checks across local Python envs.

Default env roots are under ~/Envs.
This script is intentionally conservative:
- 3.12 and 3.13 are required to pass.
- 3.14 is reported as experimental by default.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENV_BASE = Path.home() / "Envs"
DEFAULT_ENVS = {
    "3.12": DEFAULT_ENV_BASE / "mlchemenv312",
    "3.13": DEFAULT_ENV_BASE / "mlchemenv313",
    "3.14": DEFAULT_ENV_BASE / "mlchemenv314",
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


def _refresh_coverage_badges(py312: Path, live_output: bool) -> tuple[bool, str]:
    """Run py3.12 coverage and refresh local badge SVGs under assets/."""
    ok, detail = _run_and_check(
        [
            str(py312),
            "-m",
            "pip",
            "install",
            "pytest-cov",
            "anybadge",
        ],
        REPO_ROOT,
        "install badge tooling",
        live_output,
        "3.12",
    )
    if not ok:
        return False, f"could not install badge tooling\n{detail}"

    ok, detail = _run_and_check(
        [
            str(py312),
            "-m",
            "pytest",
            "-q",
            "tests",
            "--cov=mlchem",
            "--cov-config=.coveragerc",
            "--cov-branch",
            "--cov-report=term",
            "--cov-report=xml:coverage.xml",
        ],
        REPO_ROOT,
        "coverage run",
        live_output,
        "3.12",
    )
    if not ok:
        return False, f"coverage run failed\n{detail}"

    coverage_xml = REPO_ROOT / "coverage.xml"
    if not coverage_xml.exists():
        return False, "coverage.xml was not generated"

    root = ET.parse(coverage_xml).getroot()
    line_pct = round(float(root.get("line-rate", 0.0)) * 100)
    branch_pct = round(float(root.get("branch-rate", 0.0)) * 100)

    line_badge = REPO_ROOT / "assets" / "coverage.svg"
    branch_badge = REPO_ROOT / "assets" / "coverage-branch.svg"

    ok, detail = _run_and_check(
        [
            str(py312),
            "-m",
            "anybadge",
            "--label",
            "line cov",
            "--value",
            str(line_pct),
            "--file",
            str(line_badge),
            "--overwrite",
            "50=red",
            "60=orange",
            "70=yellow",
            "80=yellowgreen",
            "90=green",
        ],
        REPO_ROOT,
        "line badge",
        live_output,
        "3.12",
    )
    if not ok:
        return False, f"line badge generation failed\n{detail}"

    ok, detail = _run_and_check(
        [
            str(py312),
            "-m",
            "anybadge",
            "--label",
            "branch cov",
            "--value",
            str(branch_pct),
            "--file",
            str(branch_badge),
            "--overwrite",
            "50=red",
            "60=orange",
            "70=yellow",
            "80=yellowgreen",
            "90=green",
        ],
        REPO_ROOT,
        "branch badge",
        live_output,
        "3.12",
    )
    if not ok:
        return False, f"branch badge generation failed\n{detail}"

    message = (
        f"coverage badges refreshed from py3.12 run "
        f"(line={line_pct}%, branch={branch_pct}%)."
    )
    return True, message


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
        default=True,
        help="Stream command output as each env and step runs (default: enabled).",
    )
    parser.add_argument(
        "--no-live-output",
        action="store_false",
        dest="live_output",
        help="Disable live output streaming and only show final summary.",
    )
    parser.add_argument(
        "pytest_args",
        nargs=argparse.REMAINDER,
        help="Arguments passed to pytest (example: -- -vv tests).",
    )
    parser.add_argument(
        "--refresh-badges",
        action="store_true",
        help=(
            "After matrix checks, run a dedicated py3.12 coverage pass and "
            "refresh assets/coverage.svg and assets/coverage-branch.svg locally."
        ),
    )
    args = parser.parse_args()

    allow_314_failure = False if args.strict_314 else args.allow_314_failure
    pytest_args = args.pytest_args if args.pytest_args else ["-vv", "tests"]
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

    badges_failed = False
    if args.refresh_badges:
        py312 = _python_path(DEFAULT_ENVS["3.12"])
        py312_result = next((r for r in results if r.version == "3.12"), None)
        if py312_result is None or py312_result.status != "pass":
            print(
                "\nSkipping badge refresh: Python 3.12 matrix leg did not pass. "
                "Run again once py3.12 is green."
            )
            badges_failed = True
        elif not py312.exists():
            print(f"\nSkipping badge refresh: interpreter not found at {py312}")
            badges_failed = True
        else:
            print("\nRefreshing coverage badges locally from py3.12...")
            ok, detail = _refresh_coverage_badges(py312, live_output=args.live_output)
            if ok:
                print(detail)
            else:
                print(detail)
                badges_failed = True

    hard_fail = any(r.status == "fail" for r in results)
    return 1 if hard_fail or badges_failed else 0


if __name__ == "__main__":
    sys.exit(main())
