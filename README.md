# ⚖️ Constitutional Legal Assistant - Egyptian Constitution Chatbot

An intelligent RAG-based chatbot for answering questions about the Egyptian Constitution in Arabic.

---

## 📁 Project Structure

```
Chatbot_me/
├── app_final.py                 # Main Streamlit app (v1 - basic)
├── app_final_pheonix.py         # Streamlit app with Phoenix tracing
├── app_final_updated.py         # Latest production version with improvements
├── evaluate_rag.py              # RAG evaluation with RAGAS metrics (simplified output)
├── evaluate.py                  # Full standalone evaluation script
├── requirements.txt             # Python dependencies
├── .env                         # Environment variables (create this - NOT in repo)
├── .gitignore                   # Git ignore rules
├── test_dataset_5_questions.json # Test dataset (5 questions from different categories)
├── data/                        # Legal documents (NOT in repo)
│   ├── Egyptian_Constitution_legalnature_only.json
│   ├── Egyptian_Civil.json
│   ├── Egyptian_Labour_Law.json
│   ├── Egyptian_Personal Status Laws.json
│   ├── Technology Crimes Law.json
│   └── قانون_الإجراءات_الجنائية.json
├── chroma_db/                   # Vector database (auto-generated - NOT in repo)
├── reranker/                    # Arabic reranker model files (NOT in repo)
│   ├── model.safetensors
│   ├── config.json
│   └── ...
└── *.whl                        # Local wheel packages for Phoenix (NOT in repo)
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

### 1. Run Latest Production App (`app_final_updated.py`) ⭐ RECOMMENDED

The most recent version with improved prompt engineering and decision tree logic:

```powershell
streamlit run app_final_updated.py
```

Then open: **http://localhost:8501**

**Features:**
- Enhanced Arabic RTL support
- Improved decision tree for handling different question types
- Better handling of procedural vs. constitutional questions
- Cleaner response formatting

---

### 2. Run Basic App (`app_final.py`)

The original version:

```powershell
streamlit run app_final.py
```

Then open: **http://localhost:8501**

---

### 3. Run App with Phoenix Tracing (`app_final_pheonix.py`)

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

### 4. Run Evaluation (`evaluate_rag.py`) ⭐ NEW SIMPLIFIED FORMAT

Evaluate the RAG system with simplified output showing only essential information:

```powershell
# Uses default test dataset (test_dataset_5_questions.json)
python evaluate_rag.py

# With custom test file
python evaluate_rag.py path/to/your_test.json

# Set via environment variable
set QA_FILE_PATH=test_dataset_5_questions.json
python evaluate_rag.py
```

**Output Files:**
- `evaluation_breakdown.json` - **Simplified format** with:
  - Question
  - Ground truth
  - Actual answer
  - Score (average of all metrics per question)
  - Average score across all questions
- `evaluation_results.json` - Detailed metrics breakdown
- `evaluation_detailed.json` - Full raw evaluation data

**Sample Output Format:**
```json
{
  "questions": [
    {
      "question": "ما الطبيعة القانونية لحق العمل في الدستور المصري؟",
      "ground_truth": "حق أساسي/حرية: العمل حق وواجب...",
      "actual_answer": "حسب المادة (12) من الدستور المصري...",
      "score": 0.8542
    }
  ],
  "average_score": 0.8542
}
```

**⚠️ Note:** This script has a **60-second delay** between questions to avoid Groq API rate limits.

---

### 5. Run Full Evaluation (`evaluate.py`)

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

## 📊 Test Dataset

The project includes a curated test dataset with 5 questions covering different legal categories:

**`test_dataset_5_questions.json`** includes:
1. **الدستور (Constitution)** - Constitutional rights and principles
2. **قانون العمل (Labour Law)** - Workplace rights and regulations
3. **الإجراءات الجنائية (Criminal Procedures)** - Criminal law procedures
4. **جرائم تقنية المعلومات (Technology Crimes)** - Cybercrime laws
5. **الأحوال الشخصية (Personal Status Laws)** - Family law matters

This diverse dataset ensures comprehensive testing across all major legal domains covered by the system.

---

## 📊 Understanding RAGAS Metrics

The evaluation system uses RAGAS metrics to assess the quality of the RAG pipeline. The simplified output combines these into a single score per question:

| Metric | Description | Good Score |
|--------|-------------|------------|
| **faithfulness** | Is answer grounded in context? | > 0.7 |
| **answer_relevancy** | Does answer match the question? | > 0.8 |
| **context_precision** | How much context was useful? | > 0.6 |
| **context_recall** | Did we retrieve all needed info? | > 0.7 |

**Question Score** = Average of all four metrics (0-1 scale)

**Overall Score** = Average of all question scores

---

## � Repository Structure & Git

### Files NOT Included in Repository (via `.gitignore`)

The following files are excluded from version control for security, size, or privacy reasons:

1. **`reranker/`** - Large model files (download separately or train locally)
2. **`__pycache__/`** - Python compiled bytecode
3. **`chroma_db/`** - Vector database (auto-generated on first run)
4. **`.env`** - Environment variables with API keys (NEVER commit this!)
5. **`*.json`** - All JSON files EXCEPT `test_dataset_5_questions.json`
6. **`*.csv`** - CSV data files
7. **`*.md`** - All markdown files EXCEPT `README.md`
8. **`*.whl`** - Wheel package files

### First-Time Setup

When cloning this repository, you'll need to:

1. **Create `.env` file** with your API keys
2. **Download/prepare data files** in the `data/` folder
3. **Download reranker model** to `reranker/` folder
4. **Install dependencies** from `requirements.txt`
5. **Run the app** - ChromaDB will auto-generate on first run

---

## �🔧 Troubleshooting

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
│  Decision Tree Logic            │
│  (app_final_updated.py)         │
│  ├── Constitutional questions   │
│  ├── Procedural questions       │
│  ├── General legal advice       │
│  └── Out-of-scope filtering     │
└─────────────────────────────────┘
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
│  - Separate system/user prompts │
│  - Citation with article numbers│
│  - Temperature: 0.3              │
└─────────────────────────────────┘
        ↓
    Final Answer
```

---

## 📋 Version History

### Latest Updates (Feb 2026)
- ✅ Added `app_final_updated.py` with improved decision tree logic
- ✅ Simplified evaluation output (question, ground_truth, answer, score)
- ✅ Created curated 5-question test dataset covering 5 legal categories
- ✅ Added comprehensive `.gitignore` for repository management
- ✅ Updated documentation with all recent changes
- ✅ Improved Arabic RTL support and number formatting

### Previous Features
- Multi-source legal document support (Constitution, Civil, Labour, etc.)
- Hybrid retrieval with RRF (Reciprocal Rank Fusion)
- Arabic-specific reranker integration
- Phoenix tracing for observability
- RAGAS-based evaluation system

---

## ☁️ Upload to GitHub (Branch: `ahmed`)

Use these commands from the project root:

```powershell
# Create/switch to your branch
git checkout -B ahmed

# Review what will be committed
git status

# Stage all safe files (filtered by .gitignore)
git add .

# Commit
git commit -m "Project update: secure gitignore + README"

# Push branch to GitHub
git push -u origin ahmed
```

Before pushing, verify that `.env`, local DB files, and private keys are not listed by `git status`.

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
