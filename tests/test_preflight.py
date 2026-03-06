from llm_markdown.preflight import check_env, check_sdk_imports, run_preflight


def test_check_env_reports_missing(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    results = check_env(["OPENAI_API_KEY"])
    assert len(results) == 1
    assert results[0].ok is False


def test_check_env_reports_set(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    results = check_env(["OPENAI_API_KEY"])
    assert len(results) == 1
    assert results[0].ok is True


def test_check_sdk_imports_returns_expected_shape():
    results = check_sdk_imports()
    assert len(results) >= 1
    assert all(result.name.startswith("sdk:") for result in results)


def test_run_preflight_returns_summary():
    results, all_ok = run_preflight()
    assert isinstance(results, list)
    assert isinstance(all_ok, bool)
