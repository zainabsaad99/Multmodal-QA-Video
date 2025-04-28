# 🎬🔍 Multimodal RAG for Video Question Answering

Ask any question about a video — and get instant answers based on the most relevant transcript segments and keyframes.

---

## ✨ Project Overview

This system builds a **Multimodal Retrieval-Augmented Generation (RAG)** pipeline by:
- Downloading YouTube videos
- Extracting frames and transcribing speech
- Generating **text** and **image** embeddings
- Storing embeddings in **FAISS** and **PostgreSQL + pgvector**
- Enabling **semantic**, **lexical**, and **hybrid** search
- Providing an interactive **Streamlit** app for question answering

---

## 🚀 Features

- **Semantic Retrieval:** FAISS, PostgreSQL (IVFFLAT & HNSW)
- **Lexical Retrieval:** TF-IDF, BM25
- **Hybrid Search:** Combines semantic and lexical relevance
- **Multimodal Embeddings:** Text (Sentence Transformers) + Frames (CLIP)
- **Streamlit App:** User-friendly interface with video timestamp jumping

---

## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/zainabsaad99/video-qa-multimodal-rag.git
cd video-qa-multimodal-rag
```

---

### 2. Create and Activate the Conda Environment

```bash
conda env create -f environment.yml
conda activate videoqa
```

> ✅ This will install **all required packages** using Conda and Pip.

---

### 3. Set up PostgreSQL + pgvector on Windows

#### Step 1: Install PostgreSQL
👉 Download from [PostgreSQL Official Website](https://www.postgresql.org/download/windows/).

#### Step 2: Set Environment Variable
```bash
set PGROOT=C:\Program Files\PostgreSQL\17
```

#### Step 3: Install pgvector Extension
```bash
cd %TEMP%
git clone --branch v0.8.0 https://github.com/pgvector/pgvector.git
cd pgvector
nmake /F Makefile.win
nmake /F Makefile.win install
```

#### Step 4: Enable pgvector in PostgreSQL
Connect to your database and run:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

---

## 📋 Workflow

Follow these steps **in order**:

### 1. Download and Process Video
```bash
python data_preparation.py
```

- Downloads the video
- Generates audio transcription
- Extracts video keyframes

---

### 2. Generate Text and Image Embeddings
```bash
python embedding_generator.py
```

- Creates text and image embeddings

---

### 3. Build Retrieval System
```bash
python retrieval_system.py
```

- Stores embeddings into FAISS and PostgreSQL
- Builds TF-IDF and BM25 models

---

### 4. Launch Streamlit App
```bash
streamlit run video_qa_app.py
```

Open [http://localhost:8501](http://localhost:8501) and start asking questions!

---

## 📂 Project Structure

```bash
├── processed_data/
│   ├── transcription.json          # Video transcript
│   └── frames/                      # Keyframes from the video
├── embeddings/
│   ├── text_embeddings.npy
│   └── image_embeddings.npy
├── retrieval/
│   ├── faiss_text.index
│   ├── faiss_image.index
│   ├── tfidf_vectorizer.pkl
│   └── bm25_model.pkl
├── Evaluation/
│   ├── evaluation.py
│   ├── evaluation_text_based_results.csv
│   └──  gold_test_set.csv
├── video_qa_app.py                  # Streamlit web application
├── data_preparation.py              # Video downloader + processor
├── embedding_generator.py           # Text and image embedding generator
├── retrieval_system.py              # Index and database builder
├── retrieval_engine.py              # Retrieval engine backend
├── semantic_retriever.py            # Semantic search utilities
├── environment.yml                  # Conda environment definition
```

---
Got it — you want a **full professional README section** explaining:

- How to **prepare the gold test set**  
- How the **evaluation works**  
- How the **results look** with a real example (showing the table you provided).

---

## 📊 Evaluation of Retrieval Systems

### 1. Gold Test Set Preparation (`gold_test_set.xlsx`)

Before running the evaluation, you need to prepare a **gold standard** dataset in an Excel file called `gold_test_set.xlsx`.  
This file must have **three columns**:

| Column Name | Description |
|:------------|:------------|
| `Question` | The question you will query to the retrieval system |
| `Answer` | The expected gold answer text (empty if unanswerable) |
| `Question Type` | Should be either `"Answerable"` or `"Unanswerable"` |

Example rows:

| Question | Answer | Question Type |
|:---------|:-------|:--------------|
| Who is the speaker? | Hello everyone, welcome to the PC seminar. Today we have with us Professor Amir Mohad. | Answerable |
| What is token sliding? | Tokens can only move along graph edges to adjacent vertices while maintaining independence. | Answerable |
| What is the speaker's favorite food? |  | Unanswerable |

> **Note:**  
> - For **unanswerable** questions, leave the `Answer` cell empty.  
> - Be sure that `"Answerable"` and `"Unanswerable"` labels are spelled correctly.

---

### 2. How the Evaluation Works

- The system loads the `gold_test_set.xlsx`.
- For each question:
  - If the question is **Answerable**, it retrieves the top-1 result and compares it to the gold `Answer` using **cosine similarity** (threshold = **0.65** by default).
  - If the similarity is ≥ 0.65, it counts as a **Correct Match**.
- If the question is **Unanswerable**, the system expects **no good retrieval**. If any text is retrieved, it counts as a **False Positive**.

For each retrieval engine (e.g., FAISS, pgvector, TF-IDF, BM25), we compute:

| Metric | Meaning |
|:-------|:--------|
| Accuracy (Answerable) | Correctly retrieved answers / Total answerable questions |
| Rejection Rate (Unanswerable) | Correctly rejected unanswerable / Total unanswerable questions |
| Average Latency (s) | Average time to perform one retrieval |

---

### 3. Example Results

Here’s a sample evaluation table using a **65% similarity threshold**:

| Retrieval Method | Accuracy (Answerable) | Rejection Rate (Unanswerable) | Average Latency (s) |
|:------------------|:---------------------|:------------------------------|:--------------------|
| FAISS             | 0.4                   | 0.0                           | 2.287               |
| pgvector-IVFFLAT  | 0.3                   | 0.0                           | 2.423               |
| pgvector-HNSW     | 0.3                   | 0.0                           | 2.327               |
| TF-IDF            | 0.5                   | 0.0                           | 2.468               |
| BM25              | 0.2                   | 0.0                           | 2.098               |

---
**Interpretation:**
- Higher **Accuracy** means the model retrieves better matches for answerable questions.
- Higher **Rejection Rate** means the model correctly refuses to answer when no valid answer exists.
- Lower **Latency** means faster response time per query.

---

## ⚡ Notes

- Make sure **PostgreSQL server** is running before building or querying the database.
- You can **change the YouTube video URL** inside `data_preparation.py`.
- Adjust frame extraction rate by changing `interval_seconds` in `capture_video_frames()`.

---



