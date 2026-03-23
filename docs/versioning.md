# Versioning

## Current approach

- Package version is declared in `llm_markdown/__init__.py` and `setup.py`.
- User-visible behavior changes should be tracked in release notes or PR descriptions.

## Policy

- Patch: bug fixes, no intentional breaking API changes.
- Minor: backward-compatible feature additions.
- Major: breaking API or behavioral changes.

## API stability policy

- Stable: documented top-level APIs in `README.md` and `docs/`.
- Experimental: newly introduced surfaces without migration guarantees.
- Breaking changes require:
  - migration guidance
  - explicit release-note callout
  - tests and docs updated in same PR

## PyPI trusted publishing (same idea as hof-engine)

Publishing uses **OpenID Connect**: GitHub Actions requests a short-lived token from PyPI. You do **not** store `PYPI_API_TOKEN` in the repo.

The workflow [`.github/workflows/publish.yml`](../.github/workflows/publish.yml) already sets:

- `permissions.id-token: write`
- `environment: pypi`
- `pypa/gh-action-pypi-publish@release/v1`

### Owner must match GitHub exactly

The **Owner** field is the GitHub **username or organization** that owns the repo (as in `https://github.com/OWNER/llm-markdown`). It must match **character-for-character**. A mismatch (e.g. `jthoetter` vs `jhoetter`) will cause OIDC publish failures with no obvious PyPI error in the UI.

### One-time: register the publisher on PyPI

In [PyPI → Account settings → Publishing](https://pypi.org/manage/account/publishing/) (trusted publishers):

1. If the **`llm-markdown`** project **does not** exist on PyPI yet, use **“Add a new pending publisher”**. If it already exists, open the project → **Manage** → trusted publishers and add there.
2. Choose **GitHub** as the provider.
3. Fill in (must match the workflow exactly):

   | Field | Value |
   | ----- | ----- |
   | PyPI project name | `llm-markdown` |
   | Owner | `jhoetter` |
   | Repository name | `llm-markdown` |
   | Workflow filename | `publish.yml` |
   | Environment name | `pypi` |

The **environment name** is required because the workflow uses `environment: name: pypi`. If you leave it blank on PyPI, the run will not match.

### One-time: GitHub environment

In the GitHub repo **Settings → Environments**, create an environment named **`pypi`** (same as hof-engine). Add protection rules if you want (e.g. required reviewers) before the job can publish.

### How to release (“CLI”)

Trusted publishing runs **in CI**, not from your laptop without a token.

- **Manual run:**  
  `gh workflow run "Publish to PyPI" --repo jhoetter/llm-markdown`  
  Or: GitHub → **Actions** → **Publish to PyPI** → **Run workflow**.

- **Or** create a **GitHub Release** (publish) for a tag; the workflow also triggers on `release: published`.

The workflow runs tests, builds sdist/wheel, and uploads to PyPI via OIDC.

Local `uv publish` / `twine upload` still needs a **user API token** unless you only use the workflow above.

## Release checklist

1. Bump `setup.py` and `llm_markdown/__init__.py` version; tag `vX.Y.Z` and push the tag (or rely on your release process).
2. Run `pytest -m "not integration"`.
3. Run `python -m build`.
4. Prepare release notes using `docs/release-notes-template.md`.
5. Trigger **Publish to PyPI** (workflow_dispatch or GitHub Release) after the trusted publisher and `pypi` environment are configured.
