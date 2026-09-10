import json
from importlib import import_module
from pathlib import Path

import pytest


def test_score_semantic_answer_points_accepts_paraphrased_match() -> None:
    """语义裁判确认同义改写时，该答案点应计为命中。"""

    scorer = import_module("evaluation.scorers.semantic")
    question = "智慧农场如何利用气象站降低农业生产风险？"
    answer_points = ["连接气象站等设备实时监测天气"]
    answer = "系统连接气象站获取天气数据，并持续监测天气状况。"

    def fake_judge(
        received_question: str,
        received_answer_points: list[str],
        received_answer: str,
    ) -> list[dict[str, object]]:
        """模拟语义裁判，不调用真实模型。"""

        assert received_question == question
        assert received_answer_points == answer_points
        assert received_answer == answer
        return [
            {
                "point_index": 0,
                "verdict": "matched",
                "reason": "回答表达了连接气象站监测天气的含义。",
            }
        ]

    score = scorer.score_semantic_answer_points(
        question=question,
        answer_points=answer_points,
        answer=answer,
        judge=fake_judge,
    )

    assert score == {
        "matched": 1,
        "not_matched": 0,
        "uncertain": 0,
        "total": 1,
        "coverage": 1.0,
        "judgments": [
            {
                "point_index": 0,
                "answer_point": "连接气象站等设备实时监测天气",
                "verdict": "matched",
                "reason": "回答表达了连接气象站监测天气的含义。",
            }
        ],
    }


def test_score_semantic_answer_points_rejects_contradiction() -> None:
    """语义裁判确认含义相反时，该答案点不应计为命中。"""

    scorer = import_module("evaluation.scorers.semantic")
    question = "气象站能否预测农业灾害？"
    answer_points = ["气象站可以预测农业灾害"]
    answer = "气象站不能预测农业灾害。"

    def fake_judge(
        received_question: str,
        received_answer_points: list[str],
        received_answer: str,
    ) -> list[dict[str, object]]:
        """模拟裁判识别出回答与标准答案点含义相反。"""

        assert received_question == question
        assert received_answer_points == answer_points
        assert received_answer == answer
        return [
            {
                "point_index": 0,
                "verdict": "not_matched",
                "reason": "回答明确否定了标准答案点。",
            }
        ]

    score = scorer.score_semantic_answer_points(
        question=question,
        answer_points=answer_points,
        answer=answer,
        judge=fake_judge,
    )

    assert score == {
        "matched": 0,
        "not_matched": 1,
        "uncertain": 0,
        "total": 1,
        "coverage": 0.0,
        "judgments": [
            {
                "point_index": 0,
                "answer_point": "气象站可以预测农业灾害",
                "verdict": "not_matched",
                "reason": "回答明确否定了标准答案点。",
            }
        ],
    }


def test_build_semantic_judge_uses_structured_model_offline() -> None:
    """语义裁判应把评分材料交给结构化模型并返回逐点判断。"""

    scorer = import_module("evaluation.scorers.semantic")

    class FakeStructuredChatModel:
        """记录结构化 Schema 和模型输入的离线假模型。"""

        def __init__(self) -> None:
            self.schema: type | None = None
            self.structured_options: dict[str, object] = {}
            self.received_messages: list[object] = []

        def with_structured_output(
            self,
            schema: type,
            **options: object,
        ) -> object:
            self.schema = schema
            self.structured_options = options
            return self

        def invoke(self, messages: list[object]) -> object:
            self.received_messages = messages
            assert self.schema is not None
            return self.schema(
                judgments=[
                    {
                        "point_index": 0,
                        "verdict": "matched",
                        "reason": "回答使用不同说法表达了相同含义。",
                    }
                ]
            )

    fake_model = FakeStructuredChatModel()
    judge = scorer.build_semantic_judge(fake_model)

    judgments = judge(
        "智慧农场如何利用气象站降低农业生产风险？",
        ["连接气象站等设备实时监测天气"],
        "系统连接气象站获取天气数据，并持续监测天气状况。",
    )

    prompt_text = "\n".join(
        str(message.content)
        for message in fake_model.received_messages
    )
    assert "智慧农场如何利用气象站降低农业生产风险" in prompt_text
    assert "连接气象站等设备实时监测天气" in prompt_text
    assert "持续监测天气状况" in prompt_text
    assert "含义相反" in prompt_text
    assert "not_matched" in prompt_text
    assert "JSON" in prompt_text
    assert fake_model.structured_options == {"method": "json_mode"}
    assert judgments == [
        {
            "point_index": 0,
            "verdict": "matched",
            "reason": "回答使用不同说法表达了相同含义。",
        }
    ]


def test_score_semantic_evaluation_only_scores_knowledge_questions() -> None:
    """批量语义评分应跳过目录题和知识库外拒答题。"""

    scorer = import_module("evaluation.scorers.semantic")
    dataset = {
        "version": 2,
        "corpus_id": "education-v1",
        "questions": [
            {
                "id": "knowledge-001",
                "question": "气象站能做什么？",
                "expected_answer_type": "knowledge",
                "answer_points": [
                    "实时监测天气",
                    "预测农业灾害",
                ],
            },
            {
                "id": "directory-001",
                "question": "课程资源下有哪些目录？",
                "expected_answer_type": "directory",
                "answer_points": ["初中", "小学"],
            },
            {
                "id": "outside-001",
                "question": "知识库是否介绍量子计算机？",
                "expected_answer_type": "insufficient",
                "answer_points": ["当前知识库证据不足"],
            },
        ],
    }
    run_result = {
        "version": 2,
        "corpus_id": "education-v1",
        "results": [
            {
                "id": "knowledge-001",
                "answer": "气象站能够监测天气状况并预判农业灾害。",
            },
            {
                "id": "directory-001",
                "answer": "直接子目录是初中和小学。",
            },
            {
                "id": "outside-001",
                "answer": "当前证据不足。",
            },
        ],
    }
    received_calls: list[tuple[str, list[str], str]] = []

    def fake_judge(
        question: str,
        answer_points: list[str],
        answer: str,
    ) -> list[dict[str, object]]:
        received_calls.append((question, answer_points, answer))
        return [
            {
                "point_index": 0,
                "verdict": "matched",
                "reason": "回答表达了监测天气的含义。",
            },
            {
                "point_index": 1,
                "verdict": "matched",
                "reason": "回答表达了预测农业灾害的含义。",
            },
        ]

    score = scorer.score_semantic_evaluation(
        dataset=dataset,
        run_result=run_result,
        judge=fake_judge,
    )

    assert received_calls == [
        (
            "气象站能做什么？",
            ["实时监测天气", "预测农业灾害"],
            "气象站能够监测天气状况并预判农业灾害。",
        )
    ]
    assert score["answer_points"] == {
        "eligible_questions": 1,
        "matched": 2,
        "not_matched": 0,
        "uncertain": 0,
        "total": 2,
        "coverage": 1.0,
    }
    assert score["per_question"] == [
        {
            "id": "knowledge-001",
            "matched": 2,
            "not_matched": 0,
            "uncertain": 0,
            "total": 2,
            "coverage": 1.0,
            "judgments": [
                {
                    "point_index": 0,
                    "answer_point": "实时监测天气",
                    "verdict": "matched",
                    "reason": "回答表达了监测天气的含义。",
                },
                {
                    "point_index": 1,
                    "answer_point": "预测农业灾害",
                    "verdict": "matched",
                    "reason": "回答表达了预测农业灾害的含义。",
                },
            ],
        }
    ]


def test_score_semantic_answer_points_rejects_missing_judgment() -> None:
    """裁判漏评任一答案点时，评分器应拒绝不完整结果。"""

    scorer = import_module("evaluation.scorers.semantic")

    def incomplete_judge(
        question: str,
        answer_points: list[str],
        answer: str,
    ) -> list[dict[str, object]]:
        return [
            {
                "point_index": 0,
                "verdict": "matched",
                "reason": "回答表达了监测天气的含义。",
            }
        ]

    with pytest.raises(
        ValueError,
        match="semantic judge must return exactly one judgment",
    ):
        scorer.score_semantic_answer_points(
            question="气象站能做什么？",
            answer_points=["实时监测天气", "预测农业灾害"],
            answer="气象站能够监测天气并预测农业灾害。",
            judge=incomplete_judge,
        )


def test_score_semantic_files_writes_independent_result(
    tmp_path: Path,
) -> None:
    """文件入口应写入独立语义评分，不修改 Agent 原始结果。"""

    scorer = import_module("evaluation.scorers.semantic")
    dataset_path = tmp_path / "questions.json"
    result_path = tmp_path / "agentic-result.json"
    output_path = tmp_path / "agentic-semantic-score.json"
    dataset_path.write_text(
        json.dumps(
            {
                "version": 2,
                "corpus_id": "education-v1",
                "questions": [
                    {
                        "id": "knowledge-001",
                        "question": "气象站能做什么？",
                        "expected_answer_type": "knowledge",
                        "answer_points": ["实时监测天气"],
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    original_result_text = json.dumps(
        {
            "version": 2,
            "corpus_id": "education-v1",
            "results": [
                {
                    "id": "knowledge-001",
                    "answer": "气象站可以持续观察天气状况。",
                }
            ],
        },
        ensure_ascii=False,
    )
    result_path.write_text(original_result_text, encoding="utf-8")

    def fake_judge(
        question: str,
        answer_points: list[str],
        answer: str,
    ) -> list[dict[str, object]]:
        return [
            {
                "point_index": 0,
                "verdict": "matched",
                "reason": "持续观察天气与实时监测天气语义一致。",
            }
        ]

    saved_path = scorer.score_semantic_files(
        dataset_path=dataset_path,
        result_path=result_path,
        output_path=output_path,
        judge=fake_judge,
    )

    assert saved_path == output_path
    assert result_path.read_text(encoding="utf-8") == original_result_text
    saved_score = json.loads(output_path.read_text(encoding="utf-8"))
    assert saved_score["scoring_method"] == {
        "answer_points": "llm_semantic_judge",
        "semantic_judge_used": True,
    }
    assert saved_score["answer_points"] == {
        "eligible_questions": 1,
        "matched": 1,
        "not_matched": 0,
        "uncertain": 0,
        "total": 1,
        "coverage": 1.0,
    }


def test_main_scores_current_v19_result_with_semantic_judge(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    """命令行入口应加载一次配置并生成 V1.9 独立语义评分。"""

    scorer = import_module("evaluation.scorers.semantic")
    fake_model = object()
    fake_judge = object()
    settings = {
        "SEMANTIC_JUDGE_MODEL": "deepseek-v4-flash",
        "SEMANTIC_JUDGE_API_KEY": "test-key",
        "SEMANTIC_JUDGE_BASE_URL": "https://api.deepseek.com",
    }
    events: list[tuple[object, ...]] = []

    def fake_load_settings(project_root: Path) -> dict[str, str]:
        events.append(("load_settings", project_root))
        return settings

    def fake_build_chat_model(
        received_settings: dict[str, str],
    ) -> object:
        events.append(("build_chat_model", received_settings))
        return fake_model

    def fake_build_judge(chat_model: object) -> object:
        events.append(("build_judge", chat_model))
        return fake_judge

    def fake_score_files(
        dataset_path: Path,
        result_path: Path,
        output_path: Path,
        judge: object,
    ) -> Path:
        events.append(
            (
                "score_files",
                dataset_path,
                result_path,
                output_path,
                judge,
            )
        )
        return output_path

    monkeypatch.setattr(
        scorer,
        "load_settings_from_env",
        fake_load_settings,
        raising=False,
    )
    monkeypatch.setattr(
        scorer,
        "build_semantic_judge_chat_model",
        fake_build_chat_model,
        raising=False,
    )
    monkeypatch.setattr(
        scorer,
        "build_semantic_judge",
        fake_build_judge,
    )
    monkeypatch.setattr(
        scorer,
        "score_semantic_files",
        fake_score_files,
    )

    scorer.main(project_root=tmp_path)

    results_directory = tmp_path / "evaluation" / "results"
    result_path = results_directory / (
        "agentic-baseline-qwen3.6-flash-thinking-off-prompt-v1.9.json"
    )
    output_path = results_directory / (
        "agentic-baseline-qwen3.6-flash-thinking-off-"
        "prompt-v1.9-semantic-score.json"
    )
    assert events == [
        ("load_settings", tmp_path),
        (
            "build_chat_model",
            settings,
        ),
        ("build_judge", fake_model),
        (
            "score_files",
            tmp_path / "evaluation" / "questions.json",
            result_path,
            output_path,
            fake_judge,
        ),
    ]
    assert capsys.readouterr().out == (
        f"Agentic semantic evaluation score saved to: {output_path}\n"
    )


def test_main_scores_selected_traditional_result_with_semantic_judge(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    """命令行应允许指定传统 RAG 结果并生成独立语义评分。"""

    scorer = import_module("evaluation.scorers.semantic")
    fake_model = object()
    fake_judge = object()
    settings = {
        "SEMANTIC_JUDGE_MODEL": "deepseek-v4-flash",
        "SEMANTIC_JUDGE_API_KEY": "test-key",
        "SEMANTIC_JUDGE_BASE_URL": "https://api.deepseek.com",
    }
    received_paths: list[tuple[Path, Path, Path, object]] = []

    monkeypatch.setattr(
        scorer,
        "load_settings_from_env",
        lambda project_root: settings,
    )
    monkeypatch.setattr(
        scorer,
        "build_semantic_judge_chat_model",
        lambda received_settings: fake_model,
    )
    monkeypatch.setattr(
        scorer,
        "build_semantic_judge",
        lambda chat_model: fake_judge,
    )

    def fake_score_files(
        dataset_path: Path,
        result_path: Path,
        output_path: Path,
        judge: object,
    ) -> Path:
        received_paths.append(
            (dataset_path, result_path, output_path, judge)
        )
        return output_path

    monkeypatch.setattr(
        scorer,
        "score_semantic_files",
        fake_score_files,
    )

    result_filename = (
        "traditional-baseline-qwen3.6-flash-thinking-off.json"
    )
    scorer.main(
        project_root=tmp_path,
        argv=["--result-filename", result_filename],
    )

    results_directory = tmp_path / "evaluation" / "results"
    output_path = results_directory / (
        "traditional-baseline-qwen3.6-flash-thinking-off-"
        "semantic-score.json"
    )
    assert received_paths == [
        (
            tmp_path / "evaluation" / "questions.json",
            results_directory / result_filename,
            output_path,
            fake_judge,
        )
    ]
    assert capsys.readouterr().out == (
        f"Semantic evaluation score saved to: {output_path}\n"
    )


def test_build_semantic_judge_chat_model_uses_deepseek_official_api(
    monkeypatch,
) -> None:
    """Judge 模型应使用 DeepSeek 官方接口并关闭思考模式。"""

    scorer = import_module("evaluation.scorers.semantic")
    received_options: list[dict[str, object]] = []
    fake_model = object()

    def fake_chat_openai(**options: object) -> object:
        received_options.append(options)
        return fake_model

    monkeypatch.setattr(
        scorer,
        "ChatOpenAI",
        fake_chat_openai,
        raising=False,
    )

    result = scorer.build_semantic_judge_chat_model(
        {
            "SEMANTIC_JUDGE_MODEL": "deepseek-v4-flash",
            "SEMANTIC_JUDGE_API_KEY": "test-key",
            "SEMANTIC_JUDGE_BASE_URL": "https://api.deepseek.com",
        }
    )

    assert result is fake_model
    assert received_options == [
        {
            "model": "deepseek-v4-flash",
            "temperature": 0,
            "api_key": "test-key",
            "base_url": "https://api.deepseek.com",
            "extra_body": {
                "thinking": {"type": "disabled"},
            },
        }
    ]
