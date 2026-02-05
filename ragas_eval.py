# ragas_eval.py
# تقييم نظام RAG باستخدام Ragas
# الاستخدام:
#   1) ضعي الملف ragas_dataset_100.csv بجانب هذا السكربت (أو عدّلي DATASET_PATH)
#   2) تأكدي أن دالة ask() تعمل وتُرجع (answer, sources)
#   3) شغّلي: python ragas_eval.py
#
# ملاحظات:
# - لو ما عندك Ground Truth اتركي العمود ground_truth فاضيًا وسيتم تجاهل context_recall تلقائيًا.
# - Ragas يعتمد على LLM للتقييم؛ يمكنك ضبطه على نفس مزودك (Groq) أو أي LLM آخر.

import os
import pandas as pd
from datasets import Dataset

from rag import ask  # <-- تأكدي أن اسم ملفك rag.py وبه ask()

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

# (اختياري) استخدمي نفس LLM كمُقيّم
try:
    from langchain_groq import ChatGroq
    EVAL_LLM = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.1-8b-instant",
        temperature=0
    )
except Exception:
    EVAL_LLM = None

DATASET_PATH = "ragas_dataset_100.csv"
OUT_PATH = "ragas_results.csv"

def build_eval_rows(df: pd.DataFrame):
    rows = []
    for _, r in df.iterrows():
        q = str(r["question"]).strip()
        gt = str(r.get("ground_truth", "")).strip()

        ans, sources = ask(q)

        # contexts: النصوص المسترجعة (chunks)
        contexts = []
        for s in (sources or []):
            c = (s.get("content") or "").strip()
            if c:
                contexts.append(c)

        row = {
            "question": q,
            "answer": ans,
            "contexts": contexts,
        }
        if gt:
            row["ground_truths"] = [gt]  # ragas expects list[str]
        rows.append(row)
    return rows

def main():
    df = pd.read_csv(DATASET_PATH)

    rows = build_eval_rows(df)
    dataset = Dataset.from_list(rows)

    # لو في أي Ground Truth فعلاً، نفعل context_recall، وإلا نشيله
    has_gt = any(("ground_truths" in x) for x in rows)
    metrics = [faithfulness, answer_relevancy, context_precision]
    if has_gt:
        metrics.append(context_recall)

    kwargs = {}
    if EVAL_LLM is not None:
        kwargs["llm"] = EVAL_LLM

    result = evaluate(dataset, metrics=metrics, **kwargs)

    # تلخيص
    print("\n=== RAGAS SUMMARY ===")
    try:
        print(result)
    except Exception:
        pass

    # تفاصيل لكل سؤال
    out_df = result.to_pandas()
    out_df.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
    print(f"\nSaved per-question results to: {OUT_PATH}")

    # متوسطات
    print("\n=== MEANS ===")
    for col in out_df.columns:
        if col.startswith("faithfulness") or col.startswith("answer_relevancy") or col.startswith("context_precision") or col.startswith("context_recall"):
            try:
                mean = out_df[col].mean()
                print(f"{col}: {mean:.4f}")
            except Exception:
                pass

if __name__ == "__main__":
    main()
