# Changelog

All notable user-facing changes are documented in this file.

## [1.1.5] - 2026-09-10

### Changed

- Loosen the `rdkit` requirement from a pinned `==2026.3.3` to `>2026.3.3` to allow newer RDKit patch releases.

### Fixed

- Stabilise `boltzmann_probability` with the log-sum-exp trick so large absolute energy values (not just large gaps) can no longer underflow `math.exp` into a spurious "partition function is zero" error; empty `energy_levels` now raises a clear `ValueError` instead.
- Use `sklearn.utils.parallel.Parallel`/`delayed` instead of `joblib`'s directly in the sequential feature selector, removing an upstream `UserWarning` about scikit-learn thread-config propagation during parallel feature evaluation.

### Docs

- Align the README's documented coverage command with the CI workflow's actual `-q` flag (was shown as `-vv`).

### Verified

- Re-confirmed 7 previously-`xfail`-tracked hardening tests (3D descriptor failure contract, `logit_to_proba` extreme-value stability, `calc_centroid` zero-mass guard, `ChemicalSpace.prepare` index validation, and `undersample` ratio bounds) already pass on current `main`; TODO.md checkboxes updated accordingly.

## [1.1.4] - 2026-08-28

### Changed

- Update `get_atomicDesc` to calculate descriptors for all atoms in a molecule and return one row per atom.
- Add `max_atoms` and `pad_value` support to cap or pad the atomic descriptor matrix.
- Emit a user-facing warning describing the `get_atomicDesc` v1.1.4 argument and output-style change.

### Fixed

- Enforced deprecation of `smarts_from_string()`, `smiles_from_smarts()`, `smiles_to_inchi()` in favour of `convert_molecule_string()`

## [1.1.3] - 2026-08-25

### Fixed

- Handle degenerate single-class classification scoring in `get_geometric_S`.
	- Commit: `e021555`
- Strengthen geometric-score regression coverage for degenerate single-class behavior.
	- Commits: `fe633b8`, `0479e22`
- Improve warning suppression test to assert warning emission is actually silenced.
	- Commit: `6c1df69`

### CI

- Add a compatibility-matrix guard that fails when forbidden test-generated artifacts appear after test execution.
	- Commit: `92237d1`

### Docs and Contribution Policy

- Clarify PR/changelog/API stability guidance in contributor documentation.
	- Commits: `f6a6cd7`, `c66af35`
- Refresh README compatibility guidance and related contributor-facing wording.
	- Commit: `1d70cfc`

### Repository Governance

- Add CODEOWNERS file for review ownership.
	- Commit: `f4c2180`

## [1.1.2] - 2026-08-22

- Version bump to 1.1.2.
	- Commit: `62c81cb`