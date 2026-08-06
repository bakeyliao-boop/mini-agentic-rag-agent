"""固定评测问题集的读取与基础校验。"""

import json
from dataclasses import asdict
from pathlib import Path

from langchain_chroma import Chroma

from collections.abc import Callable
from time import perf_counter

from app.traditional_rag import (
    TraditionalRagConfig,
    answer_with_traditional_rag,
)

EVALUATION_VERSION = 2
EVALUATION_CORPUS_ID = "education-v1"
EVALUATION_QUESTION_COUNT = 10


def load_evaluation_questions(source_path: Path) -> dict[str, object]:
    """读取固定问题集，并校验版本、语料 ID 和问题数量。"""

    data = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("evaluation data must be a JSON object")
    if data.get("version") != EVALUATION_VERSION:
        raise ValueError("evaluation version must be 2")
    if data.get("corpus_id") != EVALUATION_CORPUS_ID:
        raise ValueError("evaluation corpus_id must be education-v1")

    questions = data.get("questions")
    if not isinstance(questions, list):
        raise ValueError("evaluation questions must be a list")
    if len(questions) != EVALUATION_QUESTION_COUNT:
        raise ValueError("evaluation questions must contain exactly 10 items")

    required_fields = (
        "id",
        "category",
        "question",
        "expected_answer_type",
    )
    question_ids: set[str] = set()
    for index, question in enumerate(questions):
        if not isinstance(question, dict):
            raise ValueError(f"evaluation questions[{index}] must be an object")

        for field_name in required_fields:
            field_value = question.get(field_name)
            if not isinstance(field_value, str) or not field_value.strip():
                raise ValueError(
                    f"evaluation questions[{index}].{field_name} "
                    "must be a non-empty string"
                )

        question_id = question["id"]
        if question_id in question_ids:
            raise ValueError(f"duplicate evaluation question id: {question_id}")
        question_ids.add(question_id)

    return data


def run_traditional_baseline(
    dataset: dict[str, object],
    vector_store: Chroma,
    chat_model: object,
    config: TraditionalRagConfig,
) -> dict[str, object]:
    """按固定顺序运行所有问题，并汇总传统 RAG 基线结果。"""

    questions = dataset.get("questions")
    if not isinstance(questions, list):
        raise ValueError("evaluation questions must be a list")

    results: list[dict[str, object]] = []
    for question_data in questions:
        if not isinstance(question_data, dict):
            raise ValueError("each evaluation question must be an object")

        rag_result = answer_with_traditional_rag(
            question=question_data["question"],
            vector_store=vector_store,
            chat_model=chat_model,
            config=config,
        )
        results.append(
            {
                "id": question_data["id"],
                "category": question_data["category"],
                "question": question_data["question"],
                "expected_answer_type": question_data[
                    "expected_answer_type"
                ],
                **rag_result,
            }
        )

    return {
        "version": dataset.get("version"),
        "corpus_id": dataset.get("corpus_id"),
        "config": asdict(config),
        "results": results,
    }


def run_agentic_evaluation(
        dataset:dict[str,object],
        run_question:Callable[
            [str,str],
            dict[str,object]
        ],
        config:TraditionalRagConfig,
        prompt_version:str,
        clock:Callable[[],float]=perf_counter,
)-> dict[str,object]:
    """
    按固定顺序运行所有问题，并且汇总Agentic RAG评测结果
    """
    questions = dataset.get('questions')
    if not isinstance(questions,list):
        raise ValueError('evaluation questions must be a list')

    results: list[dict[str,object]] = []

    for question_data in questions:
        if not isinstance(question_data,dict):
            raise ValueError(
                'each evaluation question must be an object'
            )

        question_id = question_data['id']
        question = question_data['question']
        thread_id = f'evaluation-{question_id}'

        start_time = clock()

        agent_result = run_question(
            question,
            thread_id,
        )

        latency_ms = (clock() - start_time) * 1000.0

        tool_traces = agent_result.get('tool_traces',[])
        if not isinstance(tool_traces,list):
            tool_traces = []

        results.append(
            {
                'id':question_id,
                'category':question_data['category'],
                'question':question,
                'expected_answer_type':question_data['expected_answer_type'],
                **agent_result,
                'thread_id':thread_id,
                'latency_ms':latency_ms,
                'tool_call_count':len(tool_traces),
            }
        )

    return {
        'version':dataset.get('version'),
        'corpus_id':dataset.get('corpus_id'),
        'config':{
            'model':config.model,
            'temperature':config.temperature,
            'enable_thinking':config.enable_thinking,
            'corpus_version':config.corpus_version,
            'prompt_version':prompt_version,
        },
        'results':results,
    }

def save_evaluation_result(
    result: dict[str, object],
    output_path: Path,
) -> None:
    """将评测结果保存为便于阅读和比较的 UTF-8 JSON。"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
