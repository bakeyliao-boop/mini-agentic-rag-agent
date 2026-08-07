import importlib
from pathlib import Path


def test_run_agentic_evaluation_from_project_wires_and_saves_result(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Agentic 评测入口应连接题集、单题执行、汇总和独立结果保存。"""

    runner = importlib.import_module("app.agentic_evaluation_runner")
    settings = {
        "DASHSCOPE_API_KEY": "test-key",
        "DASHSCOPE_BASE_URL": "https://dashscope.example/v1",
        "EMBEDDING_MODEL": "text-embedding-v4",
        "EMBEDDING_DIMENSIONS": "1024",
        "CHROMA_PERSIST_DIR": "./data/chroma",
    }
    fake_dataset = {
        "version": 2,
        "corpus_id": "education-v1",
        "questions": [
            {
                "id": "exact-001",
                "category": "exact_fact",
                "question": "智慧农场如何灌溉？",
                "expected_answer_type": "knowledge",
            }
        ],
    }
    fake_question_result = {
        "answer_type": "knowledge",
        "answer": "根据不同农作物分类灌溉。",
        "citations": [],
        "tool_traces": [],
        "thread_id": "evaluation-exact-001",
    }
    fake_evaluation_result = {
        "version": 2,
        "corpus_id": "education-v1",
        "results": [fake_question_result],
    }
    fake_runtime = object()
    events: list[tuple[object, ...]] = []

    def fake_load_questions(source_path: Path) -> dict[str, object]:
        events.append(("load", source_path))
        return fake_dataset

    def fake_build_runtime(
        project_root: Path,
        settings: dict[str, str],
    ) -> object:
        events.append(("runtime", project_root, settings))
        return fake_runtime

    def fake_run_question(
        runtime: object,
        question: str,
        thread_id: str,
    ) -> dict[str, object]:
        events.append(
            (
                "question",
                runtime,
                question,
                thread_id,
            )
        )
        return fake_question_result

    def fake_run_evaluation(
        dataset,
        run_question,
        config,
        prompt_version,
    ) -> dict[str, object]:
        events.append(
            (
                "evaluate",
                dataset,
                config,
                prompt_version,
            )
        )
        assert run_question(
            "智慧农场如何灌溉？",
            "evaluation-exact-001",
        ) == fake_question_result
        return fake_evaluation_result

    def fake_save_result(result, output_path: Path) -> None:
        events.append(("save", result, output_path))

    monkeypatch.setattr(
        runner,
        "load_evaluation_questions",
        fake_load_questions,
    )
    monkeypatch.setattr(
        runner,
        "build_agentic_runtime_from_project",
        fake_build_runtime,
    )
    monkeypatch.setattr(
        runner,
        "run_agentic_question",
        fake_run_question,
    )
    monkeypatch.setattr(
        runner,
        "run_agentic_evaluation",
        fake_run_evaluation,
    )
    monkeypatch.setattr(
        runner,
        "save_evaluation_result",
        fake_save_result,
    )

    output_path = runner.run_agentic_evaluation_from_project(
        project_root=tmp_path,
        settings=settings,
    )

    expected_output_path = (
        tmp_path
        / "evaluation"
        / "results"
        / (
            "agentic-baseline-qwen3.6-flash-thinking-off-"
            "prompt-v1.3.json"
        )
    )
    assert output_path == expected_output_path
    assert [event[0] for event in events] == [
        "load",
        "runtime",
        "evaluate",
        "question",
        "save",
    ]
    assert events[0] == (
        "load",
        tmp_path / "evaluation" / "questions.json",
    )
    assert events[1] == ("runtime", tmp_path, settings)
    assert events[2][1] == fake_dataset
    assert events[2][2].model == "qwen3.6-flash"
    assert events[2][2].enable_thinking is False
    assert events[2][3] == "Prompt-V1.3"
    assert events[3] == (
        "question",
        fake_runtime,
        "智慧农场如何灌溉？",
        "evaluation-exact-001",
    )
    assert events[4] == (
        "save",
        fake_evaluation_result,
        expected_output_path,
    )


def test_main_loads_settings_runs_evaluation_and_prints_output(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    """命令行入口应读取配置、运行评测并打印结果路径。"""

    runner = importlib.import_module(
        "app.agentic_evaluation_runner"
    )
    settings = {
        "DASHSCOPE_API_KEY": "test-key",
    }
    output_path = (
        tmp_path
        / "evaluation"
        / "results"
        / (
            "agentic-baseline-qwen3.6-flash-thinking-off-"
            "prompt-v1.3.json"
        )
    )
    calls: list[tuple[object, ...]] = []

    def fake_load_settings(
        project_root: Path,
    ) -> dict[str, str]:
        calls.append(("load", project_root))
        return settings

    def fake_run_evaluation(
        project_root: Path,
        settings: dict[str, str],
    ) -> Path:
        calls.append(
            (
                "run",
                project_root,
                settings,
            )
        )
        return output_path

    monkeypatch.setattr(
        runner,
        "load_settings_from_env",
        fake_load_settings,
    )
    monkeypatch.setattr(
        runner,
        "run_agentic_evaluation_from_project",
        fake_run_evaluation,
    )

    runner.main(project_root=tmp_path)

    assert calls == [
        ("load", tmp_path),
        (
            "run",
            tmp_path,
            settings,
        ),
    ]
    assert capsys.readouterr().out == (
        f"Agentic evaluation result saved to: {output_path}\n"
    )


def test_run_agentic_evaluation_reuses_one_shared_runtime(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """批量评测应只构建一次共享环境，再用它执行所有问题。"""
    runner = importlib.import_module("app.agentic_evaluation_runner")
    settings = {"DASHSCOPE_API_KEY": "test-key"}
    fake_dataset = {
        "version": 2,
        "corpus_id": "education-v1",
        "questions": [
            {
                "id": "exact-001",
                "question": "智慧农场如何灌溉？",
            },
            {
                "id": "outside-001",
                "question": "知识库介绍量子计算机了吗？",
            },
        ],
    }
    shared_runtime = object()
    fake_evaluation_result = {
        "version": 2,
        "corpus_id": "education-v1",
        "results": [],
    }
    events: list[tuple[object, ...]] = []

    def fake_load_questions(source_path: Path) -> dict[str, object]:
        events.append(("load", source_path))
        return fake_dataset

    def fake_build_runtime(project_root, settings):
        events.append(("build", project_root, settings))
        return shared_runtime

    def fake_run_question(runtime, question, thread_id):
        events.append(("question", runtime, question, thread_id))
        return {
            "answer_type": "insufficient",
            "answer": "当前证据不足。",
            "citations": [],
            "tool_traces": [],
            "token_usage": {},
        }

    def fake_run_evaluation(
        dataset,
        run_question,
        config,
        prompt_version,
    ) -> dict[str, object]:
        events.append(("evaluate", dataset))
        run_question(
            "智慧农场如何灌溉？",
            "evaluation-exact-001",
        )
        run_question(
            "知识库介绍量子计算机了吗？",
            "evaluation-outside-001",
        )
        return fake_evaluation_result

    def fake_save_result(result, output_path: Path) -> None:
        events.append(("save", result, output_path))

    def reject_per_question_runtime_build(*args, **kwargs):
        raise AssertionError("不应为每道题重新构建项目运行环境")

    monkeypatch.setattr(
        runner,
        "load_evaluation_questions",
        fake_load_questions,
    )
    monkeypatch.setattr(
        runner,
        "build_agentic_runtime_from_project",
        fake_build_runtime,
        raising=False,
    )
    monkeypatch.setattr(
        runner,
        "run_agentic_question",
        fake_run_question,
        raising=False,
    )
    monkeypatch.setattr(
        runner,
        "run_agentic_question_from_project",
        reject_per_question_runtime_build,
        raising=False,
    )
    monkeypatch.setattr(
        runner,
        "run_agentic_evaluation",
        fake_run_evaluation,
    )
    monkeypatch.setattr(
        runner,
        "save_evaluation_result",
        fake_save_result,
    )

    runner.run_agentic_evaluation_from_project(
        project_root=tmp_path,
        settings=settings,
    )

    assert [event[0] for event in events] == [
        "load",
        "build",
        "evaluate",
        "question",
        "question",
        "save",
    ]
    assert events[1] == ("build", tmp_path, settings)
    assert events[3][1] is shared_runtime
    assert events[4][1] is shared_runtime
