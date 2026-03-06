import argparse
import importlib.util
import os
from dataclasses import dataclass


REQUIRED_KEYS = [
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GOOGLE_API_KEY",
    "OPENROUTER_API_KEY",
]

SDK_IMPORTS = {
    "openai": "openai",
    "anthropic": "anthropic",
    "google-genai": "google.genai",
}


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str


def check_env(required_keys: list[str]) -> list[CheckResult]:
    results: list[CheckResult] = []
    for key in required_keys:
        exists = bool(os.environ.get(key))
        detail = "set" if exists else "missing"
        results.append(CheckResult(name=f"env:{key}", ok=exists, detail=detail))
    return results


def check_sdk_imports() -> list[CheckResult]:
    results: list[CheckResult] = []
    for package_name, module_name in SDK_IMPORTS.items():
        exists = importlib.util.find_spec(module_name) is not None
        detail = "installed" if exists else "missing"
        results.append(
            CheckResult(
                name=f"sdk:{package_name}",
                ok=exists,
                detail=detail,
            )
        )
    return results


def check_openrouter_test_models() -> list[CheckResult]:
    """Best-effort static check: models configured in tests file."""
    try:
        from tests.test_integration_multi_providers import OPENROUTER_TEST_MODELS
    except Exception as exc:  # pragma: no cover - import guard
        return [
            CheckResult(
                name="openrouter:test-model-list",
                ok=False,
                detail=f"could not load OPENROUTER_TEST_MODELS: {exc}",
            )
        ]

    if not OPENROUTER_TEST_MODELS:
        return [
            CheckResult(
                name="openrouter:test-model-list",
                ok=False,
                detail="empty",
            )
        ]
    return [
        CheckResult(
            name="openrouter:test-model-list",
            ok=True,
            detail=", ".join(OPENROUTER_TEST_MODELS),
        )
    ]


def run_preflight() -> tuple[list[CheckResult], bool]:
    results = []
    results.extend(check_env(REQUIRED_KEYS))
    results.extend(check_sdk_imports())
    results.extend(check_openrouter_test_models())
    all_ok = all(r.ok for r in results)
    return results, all_ok


def _print_results(results: list[CheckResult]):
    for result in results:
        status = "OK" if result.ok else "FAIL"
        print(f"[{status}] {result.name}: {result.detail}")


def main():
    parser = argparse.ArgumentParser(
        description="Preflight checks for llm-markdown runtime/integration setup.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if any check fails.",
    )
    _ = parser.parse_args()

    results, all_ok = run_preflight()
    _print_results(results)
    if not all_ok:
        print(
            "\nPreflight has failures. For integration tests, install extras with "
            "`pip install -e \".[test,all]\"` and ensure env vars are set."
        )
    if _ and _.strict and not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
