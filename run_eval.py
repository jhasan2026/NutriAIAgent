# eval/run_eval.py
#
# Run evaluation over JSONL test cases.
#
# Produces:
# - results.jsonl: per-case metrics + artifacts
# - summary.json: aggregate metrics
#
# Usage:
#   python -m eval.run_eval \
#     --cases eval/test_cases.jsonl \
#     --out eval/results.jsonl \
#     --summary_out eval/summary.json
#
# Disable LLM judges:
#   python -m eval.run_eval --no_llm_judge
#
# Note:
# This expects your agent to be importable as `chatbot`
# from agent_backend.py

from __future__ import annotations

import argparse
import json
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage

from agent_backend import chatbot

from eval.metrics import (
    structure_score,
    constraint_adherence,
    safety_rule_flags,
    tool_usage_score,
    strict_tool_value_usage,
    precision_recall_at_k,
    keyword_recall,
)

from eval.llm_judges import run_judges


# -----------------------------
# IO helpers
# -----------------------------
def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: str, obj: Dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def now_ms() -> int:
    return int(time.time() * 1000)


def new_thread_id() -> str:
    return str(uuid.uuid4())


# -----------------------------
# Agent runner
# -----------------------------
def run_agent_case(
    user_message: str,
    user_profile: Dict[str, Any],
    ehr_json: Dict[str, Any],
    hard_constraints: List[str],
) -> Dict[str, Any]:
    thread_id = new_thread_id()

    graph_input = {
        "messages": [HumanMessage(content=user_message)],
        "user_profile": user_profile,
        "ehr_json": ehr_json,
        "ehr_ready": bool(ehr_json),
        "hard_constraints": hard_constraints,
        "ehr_context": "",
        "tool_context": {},
        "final_answer": "",
    }

    t0 = now_ms()
    result = chatbot.invoke(
        graph_input,
        config={"configurable": {"thread_id": thread_id}},
    )
    t1 = now_ms()

    return {
        "thread_id": thread_id,
        "latency_ms": t1 - t0,
        "answer": result.get("final_answer", "") or "",
        "ehr_context": result.get("ehr_context", "") or "",
        "tool_context": result.get("tool_context", {}) or {},
        "retrieved_ids": result.get("retrieved_ids", []) or [],
    }


# -----------------------------
# Aggregation
# -----------------------------
def avg(values: List[Optional[float]]) -> Optional[float]:
    vals = [v for v in values if isinstance(v, (int, float))]
    if not vals:
        return None
    return round(sum(vals) / len(vals), 4)


def aggregate(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    lat = [r["latency_ms"] for r in results]

    struct = [r["metrics"]["structure"]["score"] for r in results]
    cons = [r["metrics"]["constraints"]["score"] for r in results]
    safe = [r["metrics"]["safety_rules"]["score"] for r in results]
    tool = [r["metrics"]["tool_usage"]["score"] for r in results]
    strict_tool = [r["metrics"]["strict_tool"]["score"] for r in results]

    grounding = [
        r["metrics"]["llm_judge"].get("grounding", {}).get("score")
        for r in results
    ]
    safety_j = [
        r["metrics"]["llm_judge"].get("safety", {}).get("score")
        for r in results
    ]
    pers = [
        r["metrics"]["llm_judge"].get("personalization", {}).get("score")
        for r in results
    ]

    ret_prec = [
        r["metrics"]["retrieval"].get("precision_at_k")
        for r in results
    ]
    ret_rec = [
        r["metrics"]["retrieval"].get("recall_at_k")
        for r in results
    ]
    kw_rec = [
        r["metrics"]["retrieval"].get("keyword_recall")
        for r in results
    ]

    return {
        "n": len(results),
        "avg_latency_ms": round(sum(lat) / max(1, len(lat)), 2),
        "avg_structure": avg(struct),
        "avg_constraint_adherence": avg(cons),
        "avg_safety_rules": avg(safe),
        "avg_tool_usage": avg(tool),
        "avg_strict_tool_usage": avg(strict_tool),
        "avg_precision_at_k": avg(ret_prec),
        "avg_recall_at_k": avg(ret_rec),
        "avg_keyword_recall": avg(kw_rec),
        "avg_grounding_llm": avg(grounding),
        "avg_safety_llm": avg(safety_j),
        "avg_personalization_llm": avg(pers),
    }


# -----------------------------
# Main evaluation loop
# -----------------------------
def evaluate(
    cases: List[Dict[str, Any]],
    use_llm_judge: bool,
    judge_model: str,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []

    for case in cases:
        case_id = case.get("case_id", "unknown_case")
        user_message = str(case["user_message"])

        user_profile = case.get("user_profile", {}) or {}
        ehr_json = case.get("ehr_json", {}) or {}
        hard_constraints = case.get("hard_constraints", []) or []

        must_avoid = case.get("must_avoid", []) or []

        require_weather = bool(case.get("require_weather", True))
        require_prices = bool(case.get("require_prices", True))

        expected_relevant_ids = case.get("expected_relevant_ids", []) or []
        expected_keywords = case.get("expected_keywords", []) or []
        k = int(case.get("k", 6))

        agent_out = run_agent_case(
            user_message=user_message,
            user_profile=user_profile,
            ehr_json=ehr_json,
            hard_constraints=hard_constraints,
        )

        answer = agent_out["answer"]
        ehr_context = agent_out["ehr_context"]
        tool_context = agent_out["tool_context"]
        retrieved_ids = agent_out["retrieved_ids"]

        m_structure = structure_score(answer)
        m_constraints = constraint_adherence(answer, must_avoid)
        m_safety = safety_rule_flags(answer)
        m_tool = tool_usage_score(
            answer=answer,
            require_weather=require_weather,
            require_prices=require_prices,
        )
        m_strict_tool = strict_tool_value_usage(
            answer=answer,
            tool_context=tool_context,
        )

        retrieval_metrics: Dict[str, Any] = {}
        pr = precision_recall_at_k(
            retrieved_ids=retrieved_ids,
            expected_relevant_ids=expected_relevant_ids,
            k=k,
        )
        retrieval_metrics["precision_at_k"] = pr.get("precision")
        retrieval_metrics["recall_at_k"] = pr.get("recall")

        kw = keyword_recall(
            retrieved_text=ehr_context,
            expected_keywords=expected_keywords,
        )
        retrieval_metrics["keyword_recall"] = kw.get("score")

        llm_scores: Dict[str, Any] = {}
        if use_llm_judge:
            llm_scores = run_judges(
                judge_model=judge_model,
                user_message=user_message,
                answer=answer,
                retrieved_context=ehr_context,
            )

        results.append(
            {
                "case_id": case_id,
                "latency_ms": agent_out["latency_ms"],
                "metrics": {
                    "structure": m_structure,
                    "constraints": m_constraints,
                    "safety_rules": m_safety,
                    "tool_usage": m_tool,
                    "strict_tool": m_strict_tool,
                    "retrieval": retrieval_metrics,
                    "llm_judge": llm_scores,
                },
                "artifacts": {
                    "user_message": user_message,
                    "answer": answer,
                    "ehr_context": ehr_context,
                    "tool_context": tool_context,
                    "retrieved_ids": retrieved_ids,
                },
            }
        )

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=str, default="eval/test_cases.jsonl")
    parser.add_argument("--out", type=str, default="eval/results.jsonl")
    parser.add_argument("--summary_out", type=str, default="eval/summary.json")
    parser.add_argument("--no_llm_judge", action="store_true")
    parser.add_argument("--judge_model", type=str, default="gpt-4o-mini")
    args = parser.parse_args()

    cases = read_jsonl(args.cases)

    results = evaluate(
        cases=cases,
        use_llm_judge=not args.no_llm_judge,
        judge_model=args.judge_model,
    )

    write_jsonl(args.out, results)
    summary = aggregate(results)
    write_json(args.summary_out, summary)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()