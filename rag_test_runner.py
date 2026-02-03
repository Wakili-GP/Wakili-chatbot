#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG Test Runner (metrics + CSV)
--------------------------------
Runs a JSON test set against either:
1) a local Python module exposing `ask(question) -> (answer: str, sources: list[dict])`, OR
2) an HTTP endpoint that returns JSON {"answer": "...", "sources": [...]}

Outputs:
- summary metrics to stdout
- a CSV report with per-case results (pass/fail + recall@k + rank)
"""
from __future__ import annotations

import argparse
import csv
import importlib
import json
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

try:
    import requests  # optional for --mode http
except Exception:  # pragma: no cover
    requests = None  # type: ignore


@dataclass
class CaseResult:
    case_id: str
    category: str
    question: str
    expected_articles: List[str]
    got_articles: List[str]
    recall_at_k: float
    rank: Optional[int]
    passed: bool
    fail_reason: Optional[str]
    answer_snippet: str


def normalize_article_number(x: Any) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip()
    s = re.sub(r"\s+", " ", s)
    return s or None


def extract_articles(sources: Any, key: str) -> List[str]:
    out: List[str] = []
    if not sources:
        return out
    if isinstance(sources, dict):
        sources = [sources]
    if not isinstance(sources, list):
        return out

    for s in sources:
        if not isinstance(s, dict):
            continue
        val = s.get(key)
        num = normalize_article_number(val)
        if num:
            out.append(num)
    return out


def recall_and_rank(expected: List[str], retrieved: List[str], k: int) -> Tuple[float, Optional[int]]:
    if not expected:
        return 0.0, None
    topk = retrieved[:k]
    exp_set = set(expected)
    for i, a in enumerate(topk, start=1):
        if a in exp_set:
            return 1.0, i
    return 0.0, None


def run_local(module_name: str, question: str) -> Tuple[str, Any]:
    mod = importlib.import_module(module_name)
    if not hasattr(mod, "ask"):
        raise AttributeError(f"Module '{module_name}' must expose ask(question) -> (answer, sources)")
    answer, sources = mod.ask(question)
    return str(answer), sources


def run_http(url: str, question: str, timeout: float = 60.0) -> Tuple[str, Any]:
    if requests is None:
        raise RuntimeError("requests is not installed. Install it or use --mode local.")
    resp = requests.post(url, json={"question": question}, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    return str(data.get("answer", "")), data.get("sources", [])


def evaluate_case(
    case: Dict[str, Any],
    mode: str,
    module_name: str,
    url: str,
    k: int,
    source_article_key: str,
) -> CaseResult:
    case_id = str(case.get("id", ""))
    category = str(case.get("category", ""))
    question = str(case.get("question", "")).strip()

    expected = case.get("expected", {}) or {}
    expected_articles = [normalize_article_number(x) for x in (expected.get("expected_article_numbers") or [])]
    expected_articles = [x for x in expected_articles if x]
    expected_prefix = expected.get("expected_answer_prefix")

    # Run model
    if mode == "local":
        answer, sources = run_local(module_name, question)
    else:
        answer, sources = run_http(url, question)

    got_articles = extract_articles(sources, source_article_key)

    r_at_k, rank = recall_and_rank(expected_articles, got_articles, k)

    # Pass/fail logic
    passed = True
    fail_reason = None

    if expected_articles:
        if r_at_k < 1.0:
            passed = False
            fail_reason = f"missed expected articles in top-{k}"
    else:
        # No expected articles => check prefix if provided (procedural general gating)
        if expected_prefix:
            if not answer.strip().startswith(str(expected_prefix).strip()):
                passed = False
                fail_reason = "answer did not follow expected procedural prefix"

    snippet = answer.strip().replace("\n", " ")
    if len(snippet) > 220:
        snippet = snippet[:220] + "…"

    return CaseResult(
        case_id=case_id,
        category=category,
        question=question,
        expected_articles=expected_articles,
        got_articles=got_articles[:k],
        recall_at_k=r_at_k,
        rank=rank,
        passed=passed,
        fail_reason=fail_reason,
        answer_snippet=snippet,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True, help="Path to JSON test suite")
    ap.add_argument("--mode", choices=["local", "http"], default="local")
    ap.add_argument("--module", default="rag", help="Python module name for local mode (default: rag)")
    ap.add_argument("--url", default="http://127.0.0.1:8000/ask", help="HTTP endpoint for http mode")
    ap.add_argument("--k", type=int, default=5, help="k for recall@k (default: 5)")
    ap.add_argument("--out", default="rag_test_report.csv", help="Output CSV file path")
    ap.add_argument(
        "--source-article-key",
        default="article_number",
        help="Key used inside each source dict for the article number (default: article_number)",
    )
    args = ap.parse_args()

    with open(args.suite, "r", encoding="utf-8") as f:
        suite = json.load(f)

    cases = suite.get("cases") if isinstance(suite, dict) else None
    if not isinstance(cases, list):
        raise ValueError("Invalid suite format: expected {'cases': [...]}")

    results: List[CaseResult] = []

    for case in cases:
        if not isinstance(case, dict):
            continue

        try:
            res = evaluate_case(case, args.mode, args.module, args.url, args.k, args.source_article_key)
            results.append(res)  # ✅ FIX: append success results
        except Exception as e:
            cid = str(case.get("id", ""))
            results.append(
                CaseResult(
                    case_id=cid,
                    category=str(case.get("category", "")),
                    question=str(case.get("question", "")),
                    expected_articles=[
                        str(x)
                        for x in (case.get("expected", {}) or {}).get("expected_article_numbers", [])
                    ],
                    got_articles=[],
                    recall_at_k=0.0,
                    rank=None,
                    passed=False,
                    fail_reason=f"runner error: {e}",
                    answer_snippet="",
                )
            )

    # Aggregate metrics
    total = len(results)
    passed_cnt = sum(1 for r in results if r.passed)
    pass_rate = (passed_cnt / total) if total else 0.0

    with_expected = [r for r in results if r.expected_articles]
    avg_recall = (sum(r.recall_at_k for r in with_expected) / len(with_expected)) if with_expected else 0.0

    # MRR on cases with expected
    rr_vals = [(1.0 / r.rank) for r in with_expected if r.rank]
    mrr = (sum(rr_vals) / len(with_expected)) if with_expected else 0.0

    # Write CSV
    fieldnames = [
        "id",
        "category",
        "question",
        "expected_articles",
        "retrieved_topk_articles",
        "recall_at_k",
        "rank",
        "passed",
        "fail_reason",
        "answer_snippet",
    ]
    with open(args.out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            w.writerow(
                {
                    "id": r.case_id,
                    "category": r.category,
                    "question": r.question,
                    "expected_articles": "|".join(r.expected_articles),
                    "retrieved_topk_articles": "|".join(r.got_articles),
                    "recall_at_k": f"{r.recall_at_k:.3f}",
                    "rank": "" if r.rank is None else r.rank,
                    "passed": "PASS" if r.passed else "FAIL",
                    "fail_reason": r.fail_reason or "",
                    "answer_snippet": r.answer_snippet,
                }
            )

    print("=== RAG Test Summary ===")
    print(f"Timestamp: {datetime.utcnow().isoformat()}Z")
    print(f"Suite: {args.suite}")
    print(f"Mode: {args.mode}")
    print(f"k: {args.k}")
    print(f"Total cases: {total}")
    print(f"Passed: {passed_cnt} ({pass_rate*100:.1f}%)")
    print(f"Avg recall@{args.k} (cases with expected): {avg_recall:.3f}")
    print(f"MRR (cases with expected): {mrr:.3f}")
    print(f"CSV report: {args.out}")

    return 0 if passed_cnt == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
