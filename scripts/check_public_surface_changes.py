#!/usr/bin/env python3
import os
import subprocess
import sys


def _run(cmd: list[str]) -> str:
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def _changed_files(base_ref: str) -> list[str]:
    if not base_ref:
        return []
    _run(["git", "fetch", "origin", base_ref, "--depth", "1"])
    output = _run(["git", "diff", "--name-only", f"origin/{base_ref}...HEAD"])
    if not output:
        return []
    return [line.strip() for line in output.splitlines() if line.strip()]


def main() -> int:
    base_ref = os.environ.get("GITHUB_BASE_REF", "")
    changed = _changed_files(base_ref)
    if not changed:
        print("No changed files detected for public surface check.")
        return 0

    public_changes = [
        path
        for path in changed
        if path.startswith("llm_markdown/") and path.endswith(".py")
    ]
    if not public_changes:
        print("No public Python surface changes detected.")
        return 0

    docs_or_tests_changed = any(
        path.startswith("docs/") or path.startswith("tests/") or path == "README.md"
        for path in changed
    )
    if docs_or_tests_changed:
        print("Public surface changes include docs/tests updates.")
        return 0

    print("Public Python changes require accompanying docs or tests updates.")
    print("Changed public files:")
    for path in public_changes:
        print(f"- {path}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
