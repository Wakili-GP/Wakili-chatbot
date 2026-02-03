---
license: apache-2.0
language:
- ar
pipeline_tag: text-ranking
tags:
- transformers
- sentence-transformers
- text-embeddings-inference
library_name: sentence-transformers
---

# Introducing ARM-V1 | Arabic Reranker Model (Version 1)

**For more info please refer to this blog: [ARM | Arabic Reranker Model](www.omarai.me).**

✨ This model is designed specifically for Arabic language reranking tasks, optimized to handle queries and passages with precision. 

✨ Unlike embedding models, which generate vector representations, this reranker directly evaluates the similarity between a question and a document, outputting a relevance score.

✨ Trained on a combination of positive and hard negative query-passage pairs, it excels in identifying the most relevant results. 

✨ The output score can be transformed into a [0, 1] range using a sigmoid function, providing a clear and interpretable measure of relevance.

## Arabic RAG Pipeline 


![Arabic RAG Pipeline](https://i.ibb.co/z4Fc3Kd/Screenshot-2024-11-28-at-10-17-39-AM.png)



## Usage 
### Using sentence-transformers

```
pip install sentence-transformers
```
```python
from sentence_transformers import CrossEncoder

# Load the cross-encoder model

# Define a query and a set of candidates with varying degrees of relevance
query = "تطبيقات الذكاء الاصطناعي تُستخدم في مختلف المجالات لتحسين الكفاءة."

# Candidates with varying relevance to the query
candidates = [
    "الذكاء الاصطناعي يساهم في تحسين الإنتاجية في الصناعات المختلفة.", # Highly relevant
    "نماذج التعلم الآلي يمكنها التعرف على الأنماط في مجموعات البيانات الكبيرة.", # Moderately relevant
    "الذكاء الاصطناعي يساعد الأطباء في تحليل الصور الطبية بشكل أفضل.", # Somewhat relevant
    "تستخدم الحيوانات التمويه كوسيلة للهروب من الحيوانات المفترسة.", # Irrelevant
]

# Create pairs of (query, candidate) for each candidate
query_candidate_pairs = [(query, candidate) for candidate in candidates]

# Get relevance scores from the model
scores = model.predict(query_candidate_pairs)

# Combine candidates with their scores and sort them by score in descending order (higher score = higher relevance)
ranked_candidates = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)

# Output the ranked candidates with their scores
print("Ranked candidates based on relevance to the query:")
for i, (candidate, score) in enumerate(ranked_candidates, 1):
    print(f"Rank {i}:")
    print(f"Candidate: {candidate}")
    print(f"Score: {score}\n")
```
## Evaluation 
### Dataset

Size: 3000 samples.

### Structure:
🔸 Query: A string representing the user's question.

🔸 Candidate Document: A candidate passage to answer the query.

🔸 Relevance Label: Binary label (1 for relevant, 0 for irrelevant).

### Evaluation Process

🔸 Query Grouping: Queries are grouped to evaluate the model's ability to rank candidate documents correctly for each query.

🔸 Model Prediction: Each model predicts relevance scores for all candidate documents corresponding to a query.

🔸 Metrics Calculation: Metrics are computed to measure how well the model ranks relevant documents higher than irrelevant ones.

| Model                                     | MRR              | MAP              | nDCG@10          |
|-------------------------------------------|------------------|------------------|------------------|
| cross-encoder/ms-marco-MiniLM-L-6-v2      | 0.631 | 0.6313| 0.725 |
| cross-encoder/ms-marco-MiniLM-L-12-v2     | 0.664 | 0.664 | 0.750 |
| BAAI/bge-reranker-v2-m3                   | 0.902 | 0.902 | 0.927 |
| Omartificial-Intelligence-Space/ARA-Reranker-V1 | **0.934**           | **0.9335**           | **0.951** |



## <span style="color:blue">Acknowledgments</span>

The author would like to thank Prince Sultan University for their invaluable support in this project. Their contributions and resources have been instrumental in the development and fine-tuning of these models.


```markdown
## Citation

If you use the GATE, please cite it as follows:

@misc{nacar2025ARM,
      title={ARM, Arabic Reranker Model}, 
      author={Omer Nacar},
      year={2025},
      url={https://huggingface.co/Omartificial-Intelligence-Space/ARA-Reranker-V1},
}




