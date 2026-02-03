# RAG Evaluation with Ragas

## 📊 Evaluation Metrics

This evaluation uses **Ragas** library to measure RAG system quality:

| Metric | Description | Range | Goal |
|--------|-------------|-------|------|
| **faithfulness** | Is the answer grounded in the context? | 0-1 | Higher |
| **answer_relevancy** | Does the answer relate to the question? | 0-1 | Higher |
| **context_precision** | How much retrieved context was relevant? | 0-1 | Higher |
| **context_recall** | Did we retrieve all needed information? | 0-1 | Higher |
| **context_relevancy** | Overall relevance of context to question | 0-1 | Higher |

## 🚀 How to Run

### 1. Prerequisites

Make sure you have:
- ✅ Ragas installed: `pip install ragas datasets`
- ✅ OpenAI API key set in `.env` file (Ragas uses it for evaluation)
- ✅ Your RAG system working (test with `streamlit run app_final.py`)

### 2. Prepare Test Dataset

Edit `test_dataset.json` with your test questions:

```json
[
    {
        "question": "Your question in Arabic",
        "ground_truth": "Expected correct answer (optional but recommended)"
    }
]
```

### 3. Run Evaluation

```bash
python evaluate_rag.py
```

### 4. Check Results

The script will generate:
- `evaluation_results.json` - Summary metrics
- `evaluation_detailed.json` - Full Q&A pairs with contexts

## 📈 Understanding Results

### Good Scores (Target)
- **faithfulness**: > 0.7 (answers stay within context)
- **answer_relevancy**: > 0.8 (answers address the question)
- **context_precision**: > 0.6 (low noise in retrieval)
- **context_recall**: > 0.7 (retrieved all needed info)
- **context_relevancy**: > 0.7 (context matches question)

### If Scores Are Low

**Low Faithfulness (<0.5)**
- LLM is hallucinating or adding external info
- Solution: Adjust prompt to be stricter about using only context

**Low Answer Relevancy (<0.6)**
- Answers don't match questions
- Solution: Check if retrieval is getting right documents

**Low Context Precision (<0.4)**
- Too much irrelevant context retrieved
- Solution: Tune retrieval k parameter, adjust reranker top_n

**Low Context Recall (<0.5)**
- Missing important information
- Solution: Increase k in retrievers, check if documents have the info

**Low Context Relevancy (<0.5)**
- Retrieved documents don't match question
- Solution: Check embeddings, tune hybrid search weights

## 🔧 Customization

### Add More Test Questions

Edit `test_dataset.json`:
```json
{
    "question": "ما هي شروط الترشح للبرلمان؟",
    "ground_truth": "يجب أن يكون مصرياً، متمتعاً بحقوقه المدنية والسياسية..."
}
```

### Change Metrics

Edit `evaluate_rag.py` and modify the metrics list:
```python
evaluation_results = evaluate(
    dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
        # Add or remove metrics here
    ],
)
```

### Use Different LLM for Evaluation

Ragas supports different LLMs. Edit the evaluation call:
```python
from ragas.llms import LangchainLLMWrapper
from langchain_groq import ChatGroq

# Use Groq instead of OpenAI for evaluation
evaluator_llm = LangchainLLMWrapper(ChatGroq(model="llama-3.1-70b-versatile"))

evaluation_results = evaluate(
    dataset,
    metrics=[...],
    llm=evaluator_llm
)
```

## 📝 Notes

1. **OpenAI API Cost**: Ragas uses OpenAI API for evaluation (GPT-3.5/4). Each evaluation costs ~$0.01-0.10 depending on dataset size.

2. **Ground Truth**: While optional, providing ground_truth improves `context_recall` metric accuracy.

3. **Arabic Support**: Ragas works with Arabic text, but evaluation quality depends on the LLM used (GPT-4 is better for Arabic than GPT-3.5).

4. **Batch Size**: For large datasets (>50 questions), consider splitting into batches to avoid API rate limits.

## 🐛 Troubleshooting

**Error: "OpenAI API key not found"**
- Add `OPENAI_API_KEY=sk-...` to your `.env` file

**Error: "Rate limit exceeded"**
- Wait a few minutes or reduce test dataset size

**Error: "Import error"**
- Run: `pip install ragas datasets openai langchain-openai`

**Metrics return NaN or 0**
- Check if answers/contexts are empty
- Ensure ground_truth is provided for context_recall
- Try with a smaller dataset first (2-3 questions)

## 📚 Resources

- [Ragas Documentation](https://docs.ragas.io/)
- [Ragas GitHub](https://github.com/explodinggradients/ragas)
- [RAG Evaluation Best Practices](https://docs.ragas.io/en/latest/concepts/metrics/index.html)
