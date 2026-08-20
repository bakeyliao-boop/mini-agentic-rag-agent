"""A 组回答运行器（复用传统回答链路）离线测试。

测试目标：run_pilot_arm_a_answer 应：
1. 复用 answer_with_traditional_rag，对完整问题只检索一次 Top-5。
2. 完整记录 answer/hits/latency_ms/token_usage/检索计数。
3. 校验 A 组固定配置，非法配置直接报错。
4. 项目入口能接线构建模型、打开独立索引并保存结果。
"""

import importlib
import json
from pathlib import Path

import pytest

ARM_A_SPEC = {
    "id": "multi-chunk-guide-001",
    "question": "请整理三项业务的办理清单。",
    "comparison_arms": [
        {
            "id": "A",
            "name": "traditional-single-top5",
            "retrieval": {
                "queries": ["use_full_question"],
                "top_k_per_query": 5,
                "maximum_retrieval_calls": 1,
            },
        }
    ],
}

FAKE_HITS = [
    {
        "path": "/办事指南.md",
        "start_line": 212,
        "end_line": 220,
        "score": 0.81,
        "preview": "优先审查的受理条件。",
    },
    {
        "path": "/办事指南.md",
        "start_line": 3633,
        "end_line": 3671,
        "score": 0.72,
        "preview": "维权援助的申请材料和办理步骤。",
    },
]


def test_run_pilot_arm_a_answer_uses_traditional_pipeline_and_records_fields(
    monkeypatch,
) -> None:
    """A 组回答应复用传统链路，并完整记录回答、命中和运行指标。"""

    pilot_runner = importlib.import_module("evaluation.runners.pilot")
    captured: dict[str, object] = {}
    fake_answer = {
        "answer": "台账：三项业务清单。",
        "hits": FAKE_HITS,
        "latency_ms": 12.5,
        "token_usage": {
            "input_tokens": 10,
            "output_tokens": 20,
            "total_tokens": 30,
        },
        "retrieval_status": "relevant",
    }

    def fake_answer_with_traditional_rag(
        question: str,
        vector_store: object,
        chat_model: object,
        config: object,
    ) -> dict[str, object]:
        captured["question"] = question
        captured["vector_store"] = vector_store
        captured["chat_model"] = chat_model
        captured["config"] = config
        return fake_answer

    monkeypatch.setattr(
        pilot_runner,
        "answer_with_traditional_rag",
        fake_answer_with_traditional_rag,
    )
    fake_vector_store = object()
    fake_chat_model = object()
    traditional_service = importlib.import_module(
        "rag_core.traditional.service"
    )
    config = traditional_service.TraditionalRagConfig()

    result = pilot_runner.run_pilot_arm_a_answer(
        spec=ARM_A_SPEC,
        vector_store=fake_vector_store,
        chat_model=fake_chat_model,
        config=config,
    )

    assert captured["question"] == "请整理三项业务的办理清单。"
    assert captured["vector_store"] is fake_vector_store
    assert captured["chat_model"] is fake_chat_model
    assert captured["config"].top_k == 5
    assert result == {
        "pilot_id": "multi-chunk-guide-001",
        "arm_id": "A",
        "arm_name": "traditional-single-top5",
        "question": "请整理三项业务的办理清单。",
        "answer": "台账：三项业务清单。",
        "hits": FAKE_HITS,
        "latency_ms": 12.5,
        "token_usage": {
            "input_tokens": 10,
            "output_tokens": 20,
            "total_tokens": 30,
        },
        "retrieval_call_count": 1,
        "retrieved_chunk_count": 2,
        "retrieval_status": "relevant",
    }


def test_run_pilot_arm_a_answer_wires_real_traditional_pipeline(
    monkeypatch,
) -> None:
    """不替换传统链路时：只检索一次 Top-5，prompt 包含候选内容与完整问题。"""

    pilot_runner = importlib.import_module("evaluation.runners.pilot")
    traditional_service = importlib.import_module(
        "rag_core.traditional.service"
    )
    search_calls: list[dict[str, object]] = []
    prompts: list[str] = []

    def fake_search_chroma_index(
        vector_store: object,
        query: str,
        **kwargs: object,
    ) -> dict[str, object]:
        search_calls.append(
            {"vector_store": vector_store, "query": query, **kwargs}
        )
        return {"hits": FAKE_HITS, "retrieval_status": "relevant"}

    monkeypatch.setattr(
        traditional_service,
        "search_chroma_index",
        fake_search_chroma_index,
    )

    class FakeChatModel:
        def invoke(self, prompt: str) -> object:
            prompts.append(prompt)
            return _FakeResponse()

    fake_vector_store = object()
    config = traditional_service.TraditionalRagConfig()

    result = pilot_runner.run_pilot_arm_a_answer(
        spec=ARM_A_SPEC,
        vector_store=fake_vector_store,
        chat_model=FakeChatModel(),
        config=config,
    )

    assert search_calls == [
        {
            "vector_store": fake_vector_store,
            "query": "请整理三项业务的办理清单。",
            "path": "/",
            "limit": 5,
        }
    ]
    assert len(prompts) == 1
    assert "优先审查的受理条件。" in prompts[0]
    assert "维权援助的申请材料和办理步骤。" in prompts[0]
    assert "请整理三项业务的办理清单。" in prompts[0]
    assert result["answer"] == "按指南整理三项业务清单。"
    assert result["token_usage"] == {
        "input_tokens": 10,
        "output_tokens": 20,
        "total_tokens": 30,
    }
    assert result["retrieval_status"] == "relevant"
    assert result["retrieval_call_count"] == 1
    assert result["retrieved_chunk_count"] == 2


def test_run_pilot_arm_a_answer_rejects_non_top5_config() -> None:
    """A 组回答运行器必须坚持单次 Top-5，非法配置直接报错。"""

    pilot_runner = importlib.import_module("evaluation.runners.pilot")
    spec = json.loads(json.dumps(ARM_A_SPEC))
    spec["comparison_arms"][0]["retrieval"]["top_k_per_query"] = 3

    with pytest.raises(ValueError):
        pilot_runner.run_pilot_arm_a_answer(
            spec=spec,
            vector_store=object(),
            chat_model=object(),
        )


def test_run_pilot_arm_a_answer_from_project_wires_components_and_saves(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """A 组回答项目入口应接线构建模型、打开独立索引并保存结果。"""

    pilot_cli = importlib.import_module("evaluation.cli.pilot")
    spec_path = (
        tmp_path
        / "evaluation"
        / "pilots"
        / "multi_chunk_guide_001.json"
    )
    spec_path.parent.mkdir(parents=True)
    spec = {
        "id": "multi-chunk-guide-001",
        "question": "请整理三项业务的办理清单。",
        "comparison_arms": [],
    }
    spec_path.write_text(
        json.dumps(spec, ensure_ascii=False),
        encoding="utf-8",
    )
    settings = {
        "DASHSCOPE_API_KEY": "test-key",
        "DASHSCOPE_BASE_URL": (
            "https://dashscope.aliyuncs.com/compatible-mode/v1"
        ),
        "EMBEDDING_MODEL": "text-embedding-v4",
        "EMBEDDING_DIMENSIONS": "1024",
        "EMBEDDING_BATCH_SIZE": "10",
        "PILOT_CHROMA_PERSIST_DIR": "data/pilots/multi-chunk-guide-001",
    }
    fake_embedding = object()
    fake_vector_store = object()
    fake_chat_model = object()
    fake_config = object()
    fake_answer_result = {"pilot_id": "multi-chunk-guide-001", "arm_id": "A"}
    calls: list[tuple[object, ...]] = []

    def fake_config_factory() -> object:
        return fake_config

    def fake_build_embeddings(**kwargs: object) -> object:
        calls.append(("embedding", kwargs))
        return fake_embedding

    def fake_chroma(**kwargs: object) -> object:
        calls.append(("open_index", kwargs))
        return fake_vector_store

    def fake_build_chat_model(
        config: object,
        api_key: str,
        base_url: str,
    ) -> object:
        calls.append(("chat_model", config, api_key, base_url))
        return fake_chat_model

    def fake_run_answer(
        loaded_spec: dict[str, object],
        vector_store: object,
        chat_model: object,
        config: object,
    ) -> dict[str, object]:
        calls.append(("run", loaded_spec, vector_store, chat_model, config))
        return fake_answer_result

    def fake_save_result(
        result: dict[str, object],
        output_path: Path,
    ) -> None:
        calls.append(("save", result, output_path))

    monkeypatch.setattr(
        pilot_cli,
        "build_dashscope_embeddings",
        fake_build_embeddings,
    )
    monkeypatch.setattr(pilot_cli, "Chroma", fake_chroma)
    monkeypatch.setattr(pilot_cli, "TraditionalRagConfig", fake_config_factory)
    monkeypatch.setattr(
        pilot_cli,
        "build_traditional_chat_model",
        fake_build_chat_model,
    )
    monkeypatch.setattr(
        pilot_cli,
        "run_pilot_arm_a_answer",
        fake_run_answer,
    )
    monkeypatch.setattr(
        pilot_cli,
        "save_evaluation_result",
        fake_save_result,
    )

    output_path = pilot_cli.run_pilot_arm_a_answer_from_project(
        project_root=tmp_path,
        settings=settings,
    )

    expected_persist_directory = (
        tmp_path / "data" / "pilots" / "multi-chunk-guide-001"
    ).resolve(strict=False)
    expected_output_path = (
        tmp_path
        / "evaluation"
        / "results"
        / "pilot-multi-chunk-guide-001-arm-a-answer.json"
    )
    assert output_path == expected_output_path
    assert calls == [
        (
            "embedding",
            {
                "model": "text-embedding-v4",
                "dimensions": 1024,
                "batch_size": 10,
                "api_key": "test-key",
                "base_url": (
                    "https://dashscope.aliyuncs.com/compatible-mode/v1"
                ),
            },
        ),
        (
            "open_index",
            {
                "collection_name": "knowledge_chunks",
                "persist_directory": str(expected_persist_directory),
                "embedding_function": fake_embedding,
            },
        ),
        (
            "chat_model",
            fake_config,
            "test-key",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        ),
        ("run", spec, fake_vector_store, fake_chat_model, fake_config),
        ("save", fake_answer_result, expected_output_path),
    ]


class _FakeResponse:
    """模拟 ChatOpenAI 返回的响应：内容与 token 用量。"""

    def __init__(self) -> None:
        self.content = "按指南整理三项业务清单。"
        self.usage_metadata = {
            "input_tokens": 10,
            "output_tokens": 20,
            "total_tokens": 30,
        }
