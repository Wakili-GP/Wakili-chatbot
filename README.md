# ⚖️ Constitutional Legal Assistant - Egyptian Constitution Chatbot

An intelligent RAG-based chatbot for answering questions about the Egyptian Constitution in Arabic.

---

## 📁 Project Structure

```
Chatbot_me/
├── app_final.py                 # Main Streamlit app (production)
├── app_final_pheonix.py         # Streamlit app with Phoenix tracing
├── evaluate_rag.py              # RAG evaluation with RAGAS metrics
├── evaluate.py                  # Full standalone evaluation script
├── requirements.txt             # Python dependencies
├── .env                         # Environment variables (create this)
├── Egyptian_Constitution_legalnature_only.json  # Constitution data
├── chroma_db/                   # Vector database (auto-generated)
├── reranker/                    # Arabic reranker model files
│   ├── model.safetensors
│   ├── config.json
│   └── ...
└── *.whl                        # Local wheel packages for Phoenix
```

---

## 🚀 Quick Start

### Step 1: Create Virtual Environment (Recommended)

```powershell
# Create virtual environment
python -m venv venv

# Activate it (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Or (Windows CMD)
.\venv\Scripts\activate.bat
```

### Step 2: Install Dependencies

```powershell
# Install all requirements
pip install -r requirements.txt
```

### Step 3: Install Local Wheel Packages (For Phoenix Tracing)

```powershell
# Install OpenInference instrumentation packages
pip install openinference_instrumentation_langchain-0.1.56-py3-none-any.whl
pip install openinference_instrumentation_openai-0.1.41-py3-none-any.whl
```

### Step 4: Create `.env` File

Create a `.env` file in the project root with:

```env
# Required: Groq API Key (get from https://console.groq.com)
GROQ_API_KEY=gsk_your_groq_api_key_here

# Optional: For Phoenix tracing
PHOENIX_OTLP_ENDPOINT=http://localhost:6006/v1/traces
PHOENIX_SERVICE_NAME=constitutional-assistant
```

---

## 🏃 Running the Applications

### 1. Run Main App (`app_final.py`)

The standard chatbot without tracing:

```powershell
streamlit run app_final.py
```

Then open: **http://localhost:8501**

---

### 2. Run App with Phoenix Tracing (`app_final_pheonix.py`)

This version includes observability/tracing with Phoenix.

#### Step A: Start Phoenix Server First

```powershell
# In a separate terminal
python -m phoenix.server.main serve
```

Phoenix UI will be at: **http://localhost:6006**

#### Step B: Run the App

```powershell
streamlit run app_final_pheonix.py
```

Then open:
- **App**: http://localhost:8501
- **Phoenix Traces**: http://localhost:6006

---

### 3. Run Evaluation (`evaluate.py`)

More comprehensive evaluation with external test dataset and rate limiting:

```powershell
# Basic run (uses test_dataset.json)
python evaluate.py

# With custom test file
python evaluate.py test_dataset_small.json

# With custom test and output files
python evaluate.py test_dataset_small.json my_results.json
```

**⚠️ Note:** This script has a **2-minute delay** between questions to avoid Groq API rate limits.

---

## 📊 Understanding RAGAS Metrics

| Metric | Description | Good Score |
|--------|-------------|------------|
| **faithfulness** | Is answer grounded in context? | > 0.7 |
| **answer_relevancy** | Does answer match the question? | > 0.8 |
| **context_precision** | How much context was useful? | > 0.6 |
| **context_recall** | Did we retrieve all needed info? | > 0.7 |

---

## 🔧 Troubleshooting

### "GROQ_API_KEY not found"
Make sure your `.env` file exists and contains:
```env
GROQ_API_KEY=gsk_your_key_here
```

### "Reranker path not found"
Ensure the `reranker/` folder exists with model files:
```
reranker/
├── model.safetensors
├── config.json
├── tokenizer.json
└── ...
```

### "Phoenix connection refused"
Start Phoenix server first:
```powershell
python -m phoenix.server.main serve
```

### Rate Limit Errors (Groq)
- Wait a few minutes and try again
- Use `test_dataset_small.json` for fewer questions
- The `evaluate.py` script has built-in 2-minute delays

### Import Errors
```powershell
# Reinstall all dependencies
pip install -r requirements.txt --force-reinstall
```

---

## 📝 API Keys Required

| Service | Purpose | Get Key From |
|---------|---------|--------------|
| **Groq** | LLM (Llama 3.1 8B) | https://console.groq.com |
| **HuggingFace** | Embeddings (auto-download) | No key needed |

---

## 🔄 How the System Works

```
User Question (Arabic)
        ↓
┌─────────────────────────────────┐
│  Hybrid Retrieval (RRF)         │
│  ├── Semantic Search (50%)      │
│  ├── BM25 Keyword (30%)         │
│  └── Metadata Filter (20%)      │
└─────────────────────────────────┘
        ↓
┌─────────────────────────────────┐
│  Cross-Reference Expansion      │
│  (Fetch related articles)       │
└─────────────────────────────────┘
        ↓
┌─────────────────────────────────┐
│  Arabic Reranker (ARM-V1)       │
│  (Select top 5 most relevant)   │
└─────────────────────────────────┘
        ↓
┌─────────────────────────────────┐
│  LLM (Llama 3.1 via Groq)       │
│  (Generate Arabic answer)       │
└─────────────────────────────────┘
        ↓
    Final Answer
```

---

## 📞 Support

For issues, check:
1. `.env` file has correct API keys
2. All dependencies installed
3. `reranker/` folder exists with model files
4. Internet connection for API calls

---

## 📄 License

This project is for educational purposes - Egyptian Constitution Legal Assistant.
