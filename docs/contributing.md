# Contributing

## Setup

```bash
pip install -e ".[test,all]"
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
