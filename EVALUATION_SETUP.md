# RAG Evaluation Setup - Summary

## 📁 Files Created

1. **evaluate_rag.py** - Full evaluation script with detailed explanations
2. **quick_eval.py** - Simple evaluation script (faster to run)
3. **test_dataset.json** - Test questions and ground truth answers
4. **EVALUATION_README.md** - Complete documentation

## 🎯 Quick Start (3 Steps)

### Step 1: Add OpenAI API Key
Add to your `.env` file:
```
OPENAI_API_KEY=sk-your-key-here
```

### Step 2: Edit Test Questions (Optional)
Edit `test_dataset.json` to add/modify test questions

### Step 3: Run Evaluation
```bash
# Option A: Quick evaluation
python quick_eval.py

# Option B: Full evaluation with explanations
python evaluate_rag.py
```

## 📊 What Gets Evaluated

### The 6 Ragas Metrics:

1. **faithfulness** (0-1, higher better)
   - Is answer grounded in context?
   - Checks if model added external information

2. **answer_relevancy** (0-1, higher better)
   - Does answer match the question?
   - Checks if answer is on-topic

3. **context_precision** (0-1, higher better)
   - How much retrieved context was useful?
   - Measures retrieval signal-to-noise ratio

4. **context_recall** (0-1, higher better)
   - Did we retrieve all needed info?
   - Requires ground_truth to measure

5. **context_relevancy** (0-1, higher better)
   - Overall context relevance to question
   - Measures retrieval quality

6. **response_relevancy** (included in answer_relevancy)
   - Similar to answer_relevancy

## 📈 Expected Scores

### Good Performance:
- faithfulness: > 0.7
- answer_relevancy: > 0.8
- context_precision: > 0.6
- context_recall: > 0.7
- context_relevancy: > 0.7

### If Scores Are Low:

**Low Faithfulness?**
→ Tighten prompt to avoid hallucinations

**Low Answer Relevancy?**
→ Check retrieval quality

**Low Context Precision?**
→ Reduce k in retrievers or increase reranker top_n

**Low Context Recall?**
→ Increase k in retrievers, check if info exists

**Low Context Relevancy?**
→ Adjust hybrid search beta weights

## 🔧 Tuning Your System

Based on evaluation results, you can tune:

1. **In app_final.py:**
   - Line 172: `base_retriever k` (semantic search)
   - Line 208: `bm25_retriever k` (keyword search)
   - Line 260: `metadata_retriever k` (metadata filter)
   - Line 335: `beta_semantic, beta_keyword, beta_metadata` (hybrid weights)
   - Line 427: `compressor top_n` (reranker final count)
   - Line 440: `temperature` (LLM creativity)

2. **Test Changes:**
   ```bash
   # After tuning parameters
   python quick_eval.py
   # Compare new scores with previous results
   ```

## 💡 Tips

1. **Start Small**: Test with 3-5 questions first
2. **Add Ground Truth**: Improves context_recall accuracy
3. **Compare Before/After**: Save results before making changes
4. **Use GPT-4**: Set `OPENAI_MODEL=gpt-4` for better Arabic evaluation

## 📝 Example Workflow

```bash
# 1. Baseline evaluation
python quick_eval.py
# Save as: evaluation_baseline.json

# 2. Tune parameter (e.g., increase reranker top_n from 5 to 7)
# Edit app_final.py line 427: top_n=7

# 3. Re-evaluate
python quick_eval.py
# Save as: evaluation_tuned.json

# 4. Compare scores
# If improved → keep change
# If worse → revert and try different parameter
```

## 🐛 Common Issues

**"OpenAI API key not found"**
```bash
# Check .env file has:
OPENAI_API_KEY=sk-...
```

**"Rate limit exceeded"**
```bash
# Wait 1 minute and retry, or reduce test dataset size
```

**"Import ragas failed"**
```bash
pip install ragas datasets openai langchain-openai
```

**Scores all 0 or NaN**
```bash
# Check:
# 1. Answers are being generated (not empty)
# 2. Contexts are being retrieved
# 3. Ground truth is provided for context_recall
```

## 📚 Output Files

After running evaluation:
- `evaluation_results.json` - Summary metrics
- `evaluation_detailed.json` - Full Q&A with contexts
- Console output - Formatted metrics display

## 🎓 Next Steps

1. Run baseline evaluation
2. Identify lowest-scoring metric
3. Tune relevant parameters
4. Re-evaluate and compare
5. Repeat until satisfied

---

**Need Help?**
- Ragas Docs: https://docs.ragas.io/
- Example Notebooks: https://github.com/explodinggradients/ragas/tree/main/docs/examples
