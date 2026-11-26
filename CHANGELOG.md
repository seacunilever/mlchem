# Changelog

All notable user-facing changes are documented in this file.

## [1.2.13] - 2026-11-25

### Added

- Independent `ChemicalSpace.plot()` colour, shape, opacity, and marker-size encodings through `*_by` columns and categorical `*_map` dictionaries. Extra input columns are preserved and aligned by SMILES during `prepare()`.
- Explicit continuous numeric colour mapping with `colour_continuous=True`, an optional palette, and a labelled colour bar.
- Validate encoding columns, complete category mappings, style bounds, and conflicts with legacy style lists, whose class/metadata-based behavior remains available.

### Fixed

- Use one real, non-empty renderer per legend key instead of overlaying all metadata shapes, preferring visible representatives. Keys identify each encoding and source column, including explicit alpha/pixel-size values because Bokeh normalises legend glyph dimensions.
- Read legacy shape categories from `METADATA` rather than the last input column, and reset Bokeh output before activating notebook output.
- Apply configured `title_fontsize` values in `ChemicalSpace.plot()` and reject unknown Bokeh option names instead of silently ignoring them.
- Derive legacy categorical `METADATA` shapes from prepared rows only, use first-appearance order for incomparable categories, and reject missing values when legacy shapes are actually plotted.
- Avoid Bokeh warnings from applying legend styling when a continuous-colour-only plot has no categorical legend.

### Docs

- Added categorical and continuous plotting examples, documented portable class-folder names and prepare-generated `MOLFILE` paths, and explained installation into a Python 3.12 notebook kernel.

## [1.2.12] - 2026-11-24

### Changed

- **Breaking:** `MajorityVote.evaluate_cv()` and `evaluate_consensus_cv()` now return one row per combination and strategy, with metric-prefixed score and fold columns (including for a single metric). The separate `metric` column and metric suffix in `combination` were removed; individual prediction columns are also metric-prefixed.
- Results are ranked by the first metric's reliability mean by default. Multiple metrics can be filtered together on the same row, and staged consensus still reuses cached predictions without refitting.
- Generate only eligible odd-sized classification combinations, avoiding the cost of constructing combinations that would be discarded.

### Fixed

- Reject invalid ensemble-size and performance-score settings before fitting, and preserve the previous screening cache if a new CV evaluation fails.

### Docs

- Updated the majority-vote README example and method docstrings for the wide results format.

## [1.2.11] - 2026-11-24

### Added

- Added an individual-only CV stage with fold predictions and a separate `evaluate_consensus_cv()` stage that evaluates shortlisted estimators without refitting.

### Docs

- Clarified the one-call and staged `MajorityVote` CV workflows and when their options are ignored or inherited.
- Documented model screening, cached consensus evaluation, and final hold-out fitting in the README.
- Refreshed the generated API documentation for 1.2.11.

## [1.2.10] - 2026-11-24

### Added

- Evaluate multiple named metrics in one `MajorityVote.evaluate_cv()` call, reusing fitted estimators and allowing per-metric optimization directions and soft-vote probability settings.
- Include minimum and maximum train, validation, and reliability fold scores in `MajorityVote.evaluate_cv()` results.

### Changed

- Accept non-list sequences for multiple `MajorityVote.evaluate_cv()` metrics and options, `generate_cartesian_product()` inputs, and similarity-map estimator columns. Clarified annotations for ordered molecule and drawing inputs and re-iterable combination inputs while retaining iterable annotations where arbitrary iterables are supported.

## [1.2.9] - 2026-11-19

### Fixed

- Fixed lower-is-better reliability to maximize the negative of loss plus the train/CV gap, rather than rewarding instability by dividing a negative loss. Raw metrics, result-column names, higher-is-better scoring, and ranking directions are unchanged; lower-is-better selection results may change.
- Saved generated ChemicalSpace structure images under portable, class-label-derived folders without clearing unrelated files. Rendering cached plot data continues to use its existing `MOLFILE` paths without generating another structure root.
- In Combinatorial model search, removed the full-train score gate at Stage2. It is kept as diagnostics but it no longer affects downstream feature retention.

## [1.2.8] - 2026-11-19

### Added

- Added per-class `opacity_list` and `marker_size_list` controls to `ChemicalSpace.plot()`.

### Changed

- Allowed sortable non-string class labels, such as integers, to be displayed in plot legends.

### Fixed

- Calculated automatic plot opacity from the rows in the dataframe being plotted, rather than files in class directories.
- Aligned coordinates, class labels, names, metadata, and structure images by unique SMILES in `ChemicalSpace.prepare()`, preventing hover images from being associated with the wrong compound when input rows are unsorted. Ambiguous or missing SMILES matches now raise a clear error.

## [1.2.7] - 2026-11-17

### Added

- Added `skip_binary_columns` to `ChemicalSpace.process()`. Binary 0/1 columns are left unchanged by default; set it to `False` to standardise them with the other descriptors.

### Changed

- Routed `ChemicalSpace` status and missing-descriptor messages through logging instead of printing directly to the console.

### Fixed

- Replaced the mutable `highlightAtoms` default in `MolDrawer.draw_mol()` with `None`; omitted or empty highlights continue to use the drawer's configured defaults.

### Docs

- Refreshed generated API documentation for the 1.2.7 release and the `ChemicalSpace.process()` option.

## [1.2.6] - 2026-11-15

### Added

- Made `ShapExplainer` background handling consistent across tree and permutation explainers: `background_data=None` uses `data` as the background, `max_samples=None` keeps all background rows, and a positive `max_samples` deterministically caps rows passed to the masker.
- Recorded the explainer actually used and its output scale, distinguishing raw TreeExplainer outputs from positive-class probabilities returned by the permutation path.
- Added usage examples to public function and method docstrings throughout the package.

### Fixed

- Fall back to the permutation explainer for SHAP's unsupported categorical-split errors, including native-categorical XGBoost models; unrelated TreeExplainer errors continue to propagate.

### Docs

- Refreshed generated documentation with the new API examples and clarified output-scale differences and tree-path-dependent SHAP as an explicit alternative.

### Development

- Added contributor issue-triage guidance for feature, refactor, and documentation work.
- Added a deterministic smoke test for combinatorial feature selection, exercising 25 candidate subsets.

## [1.2.5] - 2026-11-13

### Added

- Added optional `background_data` to `ShapExplainer`, allowing foreground samples to be explained against a separate background dataset. Explicit tree backgrounds use interventional perturbation and raw model output; omitting the option preserves the previous behavior.
- Added progress reporting for cross-validation folds, estimator fitting, and combination evaluation in `MajorityVote.evaluate_cv()`.

### Fixed

- Prevented `MolDrawer` instances from sharing mutable default highlight and canvas-size values; changed the palette-size default to an immutable tuple.

### Docs

- Added usage examples for `mol_from_string()`, `kekulise_smiles()`, and `unkekulise_smiles()`.

## [1.2.4] - 2026-11-05

### Added

- Added shared DataFrame column validation with fail-fast checks in feature-selection and majority-vote fitting, reused by scaling and undersampling.
- Allowed manifold-compression methods to take their parameters from `dict_params` without requiring duplicate positional arguments.

### Changed

- Built majority-vote prediction DataFrames from collected columns to avoid repeated insertion and reduce fragmentation warnings.
- Disabled nested estimator parallelism in y-scrambling runs to reduce parallelism conflicts.
- Strengthened undersampling input checks, including explicit target-column requirements when dropped rows are added to the test set.

### Docs

- Added an architecture guide, an end-to-end quickstart, linked API/module references, and usage examples in public docstrings.
- Added source-code links to documented modules, classes, and methods.

### CI

- Made Python 3.14 a required CI, tox, and local-matrix test version alongside Python 3.12 and 3.13.
- Isolated file-generating tests to avoid leaving test artifacts in the repository.

## [1.2.3] - 2026-11-04

### Added

- Added a `response_method` option to `crossval()`, y-scrambling, and feature-selection workflows. It defaults to `predict` for compatibility; use `predict_proba` or `decision_function` to score ROC-AUC with continuous scores.
- Added continuous probability averaging for soft-vote ROC-AUC in `MajorityVote.evaluate_cv()` while retaining rounded soft votes by default and hard voting as a separate strategy.

### Changed

- Aligned direct train/test metric calculations with the selected response method and selected binary probabilities using `estimator.classes_[1]`.
- Preserved multiclass score outputs and label-based scoring with the default `predict` response.

### Docs

- Documented line and branch coverage workflows and ignored the generated coverage report.

## [1.2.2] - 2026-10-26

### Changed

- Applied `cv_train_ratio` to each feature subset in both stages of combinatorial selection, rather than deriving one global CV threshold from the training threshold.
- Set stage-two training and CV floors from the weakest of the selected stage-one top-N subsets and accepted ties at those floors.

### Fixed

- Returned an empty result with a warning when no subset passes stage two, instead of raising an `IndexError`.

## [1.2.1] - 2026-10-03

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
- Documentation versioning now resolves from `setup.py` at docs build time (`make clean && make html`), removing manual version string sync steps.

### CI

- Refactored tag-time version workflow into validation-only docs checks (no repository write-back from CI).
- Removed CI-driven coverage badge commit/PR automation from compatibility matrix; badge refresh is now a local maintainer action via `scripts/run_local_matrix.py --refresh-badges`.

### Compatibility Notes

- This release includes API-surface changes that may require downstream code updates, notably:
	- `crossval(..., metric_function=...)` -> `crossval(..., metric=...)`
	- `MajorityVote.predict()` removal in favour of `MajorityVote.fit()`

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