# Versioning

## Current approach

- Package version is declared in `llm_markdown/__init__.py` and `setup.py`.
- User-visible behavior changes should be tracked in release notes or PR descriptions.

## Policy

- Patch: bug fixes, no intentional breaking API changes.
- Minor: backward-compatible feature additions.
- Major: breaking API or behavioral changes.

## Release checklist

1. Run `pytest -m "not integration"`.
2. Run `python -m build`.
3. Prepare release notes for user-visible changes.
4. Publish via the existing release workflow.
