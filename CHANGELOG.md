# Changelog

All notable user-facing changes are documented in this file.

## [1.2.1] - 2026-10-02

### Added

- Added flexible fold-definition support across modelling and selection workflows via `cv_splitter`, `cv_indices`, and optional `groups`, enabling scaffold/cluster/group-aware or fully precomputed CV strategies.
- Added `summarise_fold_scores()` helper to provide fold-score summaries (mean and standard error) in a single utility.
- Added leakage-aware reliability option in metrics with `cv_only` support for cleaner model-selection diagnostics.

### Changed

- Refactored `crossval()` to return structured train/CV fold outputs and summary statistics (`train_scores`, `train_mean`, `train_se`, `cv_scores`, `cv_mean`, `cv_se`) instead of a single aggregate score.
- Renamed `crossval()` argument `metric_function` to `metric` for API consistency across modelling utilities.
- Refactored `y_scrambling()` to evaluate robustness through the shared cross-validation pipeline and to consistently use its full argument set.
- Updated `SequentialForwardSelection` and `CombinatorialSelection` wrappers to consume explicit CV fold manifests when provided, improving consistency between candidate evaluations.
- Expanded `DescriptorExplainer` CV wiring to accept `cv_splitter`, `cv_indices`, and `groups`; retained `selection_strategy` as compatibility plumbing where applicable.
- Improved majority-vote workflow so final consensus fitting/reporting is centered on `fit()` outputs and prediction tables.

### Removed

- Removed `MajorityVote.predict()`; consensus prediction/report generation is now handled through `MajorityVote.fit()`.

### Fixed

- Fixed reliability-score component behaviour by removing test-score influence from the performance term in reliability calculations.
- Improved suppression of benign sklearn parallel/delayed warnings under threaded wrapper execution.
- Refined wrapper selection and filtering logic to reduce leakage risk and improve score interpretability.

### Docs

- Refreshed README, API docstrings, and generated docs to reflect new cross-validation interfaces, reliability behavior, and majority-vote workflow.
- Updated/re-ran tutorial notebooks and example artifacts to align with the refactored APIs.

### Compatibility Notes

- This release includes API-surface changes that may require downstream code updates, notably:
	- `crossval(..., metric_function=...)` -> `crossval(..., metric=...)`
	- `MajorityVote.predict()` removal in favor of `MajorityVote.fit()`

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