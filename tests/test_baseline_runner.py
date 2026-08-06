import importlib
from pathlib import Path

import pytest


def test_load_settings_from_env_rejects_missing_file(
    tmp_path: Path,
) -> None:
    """项目根目录不存在 .env 时应抛出 FileNotFoundError。"""

    baseline_runner = importlib.import_module("app.baseline_runner")
    env_path = tmp_path / ".env"
    assert not env_path.exists()

    with pytest.raises(FileNotFoundError):
        baseline_runner.load_settings_from_env(tmp_path)


def test_load_settings_from_env_reads_project_dotenv(
    tmp_path: Path,
) -> None:
    """应从项目根目录的 .env 读取 baseline 所需配置。"""

    baseline_runner = importlib.import_module("app.baseline_runner")
    (tmp_path / ".env").write_text(
        "DASHSCOPE_API_KEY=test-key\n"
        "DASHSCOPE_BASE_URL=https://dashscope.example/v1\n"
        "EMBEDDING_MODEL=text-embedding-v4\n"
        "EMBEDDING_DIMENSIONS=1024\n"
        "CHROMA_PERSIST_DIR=./data/chroma\n",
        encoding="utf-8",
        newline="\n",
    )

    result = baseline_runner.load_settings_from_env(tmp_path)

    assert result == {
        "DASHSCOPE_API_KEY": "test-key",
        "DASHSCOPE_BASE_URL": "https://dashscope.example/v1",
        "EMBEDDING_MODEL": "text-embedding-v4",
        "EMBEDDING_DIMENSIONS": "1024",
        "CHROMA_PERSIST_DIR": "./data/chroma",
    }


def test_run_traditional_baseline_rejects_empty_api_key(
    tmp_path: Path,
) -> None:
    """DashScope API Key 为空时应在创建模型前抛出 ValueError。"""

    baseline_runner = importlib.import_module("app.baseline_runner")
    settings = {
        "DASHSCOPE_API_KEY": "   ",
        "DASHSCOPE_BASE_URL": "https://dashscope.example/v1",
        "EMBEDDING_MODEL": "text-embedding-v4",
        "EMBEDDING_DIMENSIONS": "1024",
        "CHROMA_PERSIST_DIR": "./data/chroma",
    }

    with pytest.raises(ValueError, match="DASHSCOPE_API_KEY"):
        baseline_runner.run_traditional_baseline_from_project(
            project_root=tmp_path,
            settings=settings,
        )


def test_build_traditional_baseline_result_filename_separates_modes() -> None:
    """思考与非思考基线应使用不同的结果文件名。"""

    baseline_runner = importlib.import_module("app.baseline_runner")
    traditional_rag = importlib.import_module("app.traditional_rag")

    thinking_off = baseline_runner.build_traditional_baseline_result_filename(
        traditional_rag.TraditionalRagConfig(enable_thinking=False)
    )
    thinking_on = baseline_runner.build_traditional_baseline_result_filename(
        traditional_rag.TraditionalRagConfig(enable_thinking=True)
    )

    assert thinking_off == (
        "traditional-baseline-qwen3.6-flash-thinking-off.json"
    )
    assert thinking_on == (
        "traditional-baseline-qwen3.6-flash-thinking-on.json"
    )


def test_run_traditional_baseline_from_project_wires_all_components(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """执行入口应按顺序连接配置、索引、模型、题集和结果保存。"""

    baseline_runner = importlib.import_module("app.baseline_runner")
    settings = {
        "DASHSCOPE_API_KEY": "test-key",
        "DASHSCOPE_BASE_URL": (
            "https://dashscope.aliyuncs.com/compatible-mode/v1"
        ),
        "EMBEDDING_MODEL": "text-embedding-v4",
        "EMBEDDING_DIMENSIONS": "1024",
        "CHROMA_PERSIST_DIR": "./data/chroma",
    }
    knowledge_root = tmp_path / "knowledge" / "education-v1"
    knowledge_root.mkdir(parents=True)
    fake_embeddings = object()
    fake_vector_store = object()
    fake_chat_model = object()
    fake_dataset = {"version": 2, "corpus_id": "education-v1"}
    fake_result = {"version": 2, "results": []}
    events: list[tuple[object, ...]] = []

    def fake_build_embeddings(**options):
        events.append(("embedding", options))
        return fake_embeddings

    def fake_build_index(root, persist_directory, embedding):
        events.append(("index", root, persist_directory, embedding))
        return fake_vector_store

    def fake_build_chat_model(config, api_key, base_url):
        events.append(("chat", config, api_key, base_url))
        return fake_chat_model

    def fake_load_questions(source_path):
        events.append(("load", source_path))
        return fake_dataset

    def fake_run_baseline(dataset, vector_store, chat_model, config):
        events.append(("run", dataset, vector_store, chat_model, config))
        return fake_result

    def fake_save_result(result, output_path):
        events.append(("save", result, output_path))

    monkeypatch.setattr(
        baseline_runner,
        "build_dashscope_embeddings",
        fake_build_embeddings,
    )
    monkeypatch.setattr(
        baseline_runner,
        "build_knowledge_index",
        fake_build_index,
    )
    monkeypatch.setattr(
        baseline_runner,
        "build_traditional_chat_model",
        fake_build_chat_model,
    )
    monkeypatch.setattr(
        baseline_runner,
        "load_evaluation_questions",
        fake_load_questions,
    )
    monkeypatch.setattr(
        baseline_runner,
        "run_traditional_baseline",
        fake_run_baseline,
    )
    monkeypatch.setattr(
        baseline_runner,
        "save_evaluation_result",
        fake_save_result,
    )

    output_path = baseline_runner.run_traditional_baseline_from_project(
        project_root=tmp_path,
        settings=settings,
    )

    expected_output_path = (
        tmp_path
        / "evaluation"
        / "results"
        / "traditional-baseline-qwen3.6-flash-thinking-off.json"
    )
    assert output_path == expected_output_path
    assert [event[0] for event in events] == [
        "embedding",
        "index",
        "chat",
        "load",
        "run",
        "save",
    ]
    assert events[0][1] == {
        "model": "text-embedding-v4",
        "dimensions": 1024,
        "api_key": "test-key",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    }
    assert events[1][1:4] == (
        knowledge_root.resolve(strict=False),
        (tmp_path / "data" / "chroma").resolve(strict=False),
        fake_embeddings,
    )
    assert events[-1] == ("save", fake_result, expected_output_path)


def test_main_loads_settings_runs_baseline_and_prints_output(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    """main 应读取配置、运行 baseline 并打印结果路径。"""

    baseline_runner = importlib.import_module("app.baseline_runner")
    settings = {"DASHSCOPE_API_KEY": "test-key"}
    output_path = tmp_path / "traditional-baseline.json"
    calls: list[tuple[object, ...]] = []

    def fake_load_settings(project_root: Path) -> dict[str, str]:
        calls.append(("load", project_root))
        return settings

    def fake_run_baseline(
        project_root: Path,
        settings: dict[str, str],
    ) -> Path:
        calls.append(("run", project_root, settings))
        return output_path

    monkeypatch.setattr(
        baseline_runner,
        "load_settings_from_env",
        fake_load_settings,
    )
    monkeypatch.setattr(
        baseline_runner,
        "run_traditional_baseline_from_project",
        fake_run_baseline,
    )

    baseline_runner.main(project_root=tmp_path)

    assert calls == [
        ("load", tmp_path),
        ("run", tmp_path, settings),
    ]
    assert capsys.readouterr().out == (
        f"Baseline result saved to: {output_path}\n"
    )
