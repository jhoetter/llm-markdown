# Contributing

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[test,all]"
```

Run preflight checks before integration tests:

```bash
python -m llm_markdown.preflight --strict
```

## Required quality gates

Before commit:

```bash
pytest -m "not integration"
```

Before push:

```bash
pytest -m "not integration"
python -m build
```

## Integration tests

Optional (requires provider credentials):

```bash
pytest -m integration
```

## Guidelines

- Add or update tests with behavior changes.
- Keep provider behavior consistent across sync/async paths.
- Document user-visible changes in release notes or PR descriptions.
- For public API changes under `llm_markdown/`, include docs or tests updates in the same PR.
- Use `docs/release-notes-template.md` when preparing release notes.
