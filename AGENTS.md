# AGENTS.md

This repository requires a green test run before code is committed or pushed.

## Mandatory quality gate before commit

Run:

```bash
pytest -m "not integration"
```

If tests fail, do not commit.

## Mandatory quality gate before push

Run:

```bash
pytest -m "not integration"
python -m build
```

If either command fails, do not push.

## Integration tests

Integration tests are optional for local commit/push gating because they require external API keys:

```bash
pytest -m integration
```

Run integration tests whenever provider or integration behavior changes and credentials are available.
