# Contributing to mlchem

Thanks for your interest in contributing to mlchem.

## Ways to contribute

Contributions are welcome in these areas:

- Bug reports and bug fixes
- New tests and improved test coverage
- Documentation improvements
- New chemistry and ML utilities aligned with existing package scope
- Example notebooks/scripts in examples/

Before starting major changes, open an issue to discuss design and API impact.

## Development setup

1. Clone the repository and move into it.
2. Create and activate a virtual environment.
3. Install project dependencies.
4. Install mlchem in editable mode.

```bash
git clone https://github.com/seacunilever/mlchem
cd mlchem
python -m venv _venv
```

Activate the environment:

```bash
# macOS/Linux
source _venv/bin/activate

# Windows PowerShell
.\_venv\Scripts\Activate.ps1

# Windows cmd.exe
_venv\Scripts\activate.bat
```

Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Supported test policy

- Python 3.12 and 3.13 are required to pass.
- Python 3.14 is experimental (non-blocking by default).

## Run tests locally

For everyday local work, `pytest -q` from repo root is the default and usually sufficient.
Use `run_local_matrix.py` or `tox` when you need cross-version validation
(for example, larger refactors, pre-merge checks, or maintainer/release work).

Run the core suite:

```bash
pytest -q
```

Run selected tests:

```bash
pytest -q tests/path/to/test_file.py
```

Run compatibility matrix checks:

```bash
python scripts/run_local_matrix.py
```

Run tox matrix:

```bash
python -m pip install tox
python -m tox -e py312,py313,py314
```

## Documentation updates

If behavior, signatures, or examples change, update docs in the same PR.

Build docs from docs/:

```bash
cd docs
make clean
make html
```

On Windows (cmd or PowerShell):

```bash
cd docs
make.bat clean
make.bat html
```

The html target mirrors docs/build/html into docs/ via docs/_publish.py.
Commit regenerated docs output when documentation content changes.

## Coding expectations

- Keep changes focused and avoid unrelated refactors.
- Preserve public APIs unless discussed and approved.
- Add or update tests for behavior changes.
- Keep comments concise and meaningful.
- Follow existing style in each module.

## Branching and commits

1. Create a feature branch from master.
2. Keep commits small and descriptive.
3. Reference related issues when applicable.
4. Rebase/sync with latest master before opening a PR.

Example:

```bash
git checkout -b fix/descriptor-edge-case
```

## Pull request checklist

Include this in your PR description:

- Problem statement
- Proposed solution
- Test evidence (commands and results)
- Docs impact and updated files
- Risk/regression notes

Before requesting review, confirm:

- Tests pass locally
- No generated test artifacts outside tests/
- README/docs updated for user-facing changes
- No sensitive data or local machine paths are included

## Security and sensitive information

Do not commit:

- Credentials, tokens, or secrets
- Internal endpoints not meant for publication
- Personal local paths

If a sensitive value is committed:

1. Remove it from the working tree immediately.
2. Rewrite Git history to purge it.
3. Force-push rewritten history.
4. Rotate any exposed secrets.

## Review and merge

PRs are expected to:

- Pass required CI checks
- Resolve review comments
- Be merged only when scope is clear and test-backed

Thank you for helping improve mlchem.
