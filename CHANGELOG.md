# Changelog

All notable user-facing changes are documented in this file.

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