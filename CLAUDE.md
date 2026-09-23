# Yohou-MLflow

An MLflow integration for saving and serving Yohou forecasters

Instructions for AI coding assistants working in this repository.

**This file is yours.** It is seeded once when the project is generated and never
delivered again: a template update will not overwrite your edits, and there is no
merge to lose. Rewrite it freely. What follows is a starting point, not a contract.

## Project overview

A Python package generated from [python-package-copier](https://github.com/stateful-y/python-package-copier).
Source lives in `src/yohou_mlflow/`, tests in `tests/`.

## Critical workflows

- **Package management**: `uv` exclusively. No `pip`, no `venv`.
- **Task running**: `just` (see `just --list`), or `uvx nox` for the CI sessions.
- **Setup**: `just install` creates the lockfile and installs the git hooks.
  The `-f` matters: without it an existing pre-commit shim is chained rather than
  replaced, and both run on every commit.
- **Testing**: `just test-fast` for quick feedback, `just test` for everything.
- **Lint and format**: `just fix`. Tools are pinned by `uv.lock`, so `--locked`
  failures mean the lockfile is stale, not that the tool changed.
- **Docs**: `just serve` to preview, `just build` to build.

## Conventions

- Conventional commits are enforced at commit time and on pull request titles.
- `uv.lock` is tracked and CI runs with `--locked`; commit it with dependency changes.
- Docstrings follow numpydoc. In `References` sections use a markdown ordered list
  with plain `[1]` citations, never reStructuredText (`.. [1]` or `[1]_`), which
  renders literally on the generated API pages.

## Package rules

- **Loading never runs code chosen by the file.** The trust policy lives in
  `src/yohou_mlflow/_trust.py` and can only be extended by the caller
  (`extra_trusted_types`). The type list recorded in `MLmodel` never grants trust.
  Models carry no bundled code: `code_paths` was removed, and loading never touches
  the import path, because a model's `code/` directory could otherwise shadow a trusted
  package such as `yohou`. The guarantee covers `load_model` and `check_compatibility`
  only. `mlflow.pyfunc.load_model` imports what `MLmodel` names before this package runs.
- **Loading is strict about versions**: yohou must match exactly, scikit-learn and
  polars by major and minor version. Migrations across yohou versions belong in yohou
  itself, not here.
- **Every save is loaded back and compared** before it is kept. skops 0.15.0 cannot
  rebuild `zoneinfo.ZoneInfo`, so forecasters fitted on time-zone-aware data are refused
  at save time until a skops release fixes it (skops-dev/skops#545).
- **Only public yohou API.** Reading private yohou attributes would break on any yohou
  rename. The private `mlflow.utils` helpers in `_persistence.py` were checked on
  mlflow-skinny 3.0.0 and 3.16.1.

## What a failing check means

- `codecov/patch` targets the base branch's coverage, which is 100%, so every new line
  and branch under `src/` needs a test.
- The pre-push `interrogate` hook scores each changed `src/` file on its own: a file
  with undocumented private helpers fails once it is the only file changed.
- Tests run from `tmp_path` (an autouse fixture in `tests/conftest.py`) because MLflow
  writes `mlflow.db` and `mlruns/` to the working directory.
- A local Zensical docs build can silently skip pages when the host runs out of inotify
  instances. Trust the CI docs build, or build in a clean container.

The OpenSpec specs for this package (`openspec/specs/`) are gitignored and exist only
locally.

## What to record here

The things a newcomer cannot derive from the code: why a dependency is pinned, which
assumptions this project breaks relative to the template, what a failing check
actually means. Keep it short enough that it stays true.
