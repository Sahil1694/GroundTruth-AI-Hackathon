# 🚀 H-002 | Customer Experience Automation

**Location-Aware, Context-Aware, Personalized Retail Assistant powered by RAG + LLMs**

<img width="1024" height="1024" alt="Gemini_Generated_Image_59sejf59sejf59se" src="https://github.com/user-attachments/assets/a98603c0-0ae4-4cd3-bd7d-b1d3fc355ac3" />

## 📌 Overview

This project is built for the **GroundTruth AI Hackathon** (Problem Statement H-002).

I designed an AI-driven Customer Experience Agent that provides hyper-personalized, context-aware recommendations based on:

- Customer preferences
- Historical purchase behavior
- Real-time location
- Store availability
- Weather context
- Offers & store timings
- Reward points & order history

The system uses a **RAG (Retrieval Augmented Generation)** pipeline to ensure every answer is grounded in real evidence from store data, customer profiles, and historical transactions.

✨ **Think of it as an intelligent on-site assistant that knows where the customer is, what they like, and what's available nearby — and responds in seconds.**

---

## 🧠 Key Features

- **Context-Aware Responses** (location, weather, customer preferences)
- **RAG-Based Personalization** using embeddings + FAISS
- **Geo-Priority Retrieval** (store-aware ranking, distance filtering)
- **Cross-Encoder Re-Ranking** for better precision
- **Strict Evidence-Based LLM Output** (JSON format)
- **PII Masking** for safe LLM usage
- **FastAPI Backend** for serving recommendations
- **Synthetic Dataset** with 7500+ rows and 70+ PDFs

---

## 📂 Dataset

<img width="1024" height="1024" alt="Gemini_Generated_Image_ywffmcywffmcywff" src="https://github.com/user-attachments/assets/ab128ded-cfe3-4cd3-909e-7ef46fb5d8c6" />


We created a comprehensive synthetic retail dataset to simulate a real customer engagement environment.

### 1. `customers.csv`
- 500 customers
- Fields: preferences, allergies, order time, reward points, last store, etc.

### 2. `stores.csv`
- 200 store locations
- Offers, timings, lat/lon, popular items

### 3. `customer_history.csv`
- 5000+ historical orders
- Items, sizes, timestamps, ratings

### 4. `live_location_events.csv`
- 2000 simulated real-time events
- Latitude, longitude, distance, weather, customer_id

### 5. `store_pdfs/`
- Store descriptions + offers for RAG

### 6. `customer_pdfs/`
- Individual customer profiles summarized in natural language

---

## 🧱 Tech Stack

### ⚙️ Backend
- Python 3.10+
- FastAPI
- Uvicorn

### 🤖 AI / ML
- **LLMs**: Gemini Models
- **Embeddings**: all-MiniLM-L6-v2 (SentenceTransformers)
- **Reranking Model**: cross-encoder/ms-marco-MiniLM-L-6-v2
- **Vector Store**: FAISS IndexFlatIP (Cosine Search)
- **Chunking & Preprocessing**: Python, Pandas
- **RAG Pipeline**: Custom-built multi-step retrieval, ranking & prompting

### 📑 Data Processing
- Pandas
- PyPDF2 for PDF extraction
- NumPy
- Custom chunk builder + metadata linker

### 🛠 Dev Tools
- VS Code
- GitHub

---

## 🔍 RAG Workflow (Step-by-Step)

### 1. Offline Pipeline (One-time)
1. Extract text from CSVs + PDFs
2. Chunk into 300–400 token pieces
3. Create metadata for each chunk
4. Compute embeddings (384-dim)
5. Build FAISS vector index
6. Precompute customer summaries

### 2. Online Query Pipeline (Real-Time)

#### Step 1 — Live Event Input
- `customer_id`
- `detected_store_id`
- `latitude/longitude`
- `weather`
- Custom message

#### Step 2 — Build Query
Combine:
- Customer summary
- User message
- Location context

#### Step 3 — FAISS Retrieval
- Search top-k (50) relevant chunks
- Return raw candidates

#### Step 4 — Store Priority Boost
If `store_id` provided:
- Boost offers
- Boost store PDF info
- Boost store's items

#### Step 5 — Cross Encoder Reranking
- Sort candidates by true relevance score

#### Step 6 — Evidence Packing
- Pick final 3–5 pieces of evidence

#### Step 7 — Prompt Construction
Strict JSON format:
```json
{
  "message": "...",
  "reason": "...",
  "sources": [1, 3]
}
```

#### Step 8 — LLM Generation
- Deterministic output
- No hallucinations
- No PII leakage

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10 or higher
- pip or conda for package management

### Installation
```bash
# Clone the repository
git clone https://github.com/Sahil1694/GroundTruth-AI-Hackathon.git
cd GroundTruth-AI-Hackathon

# Install dependencies
pip install -r requirements.txt

# Build embeddings and FAISS index
python build_embeddings.py

# Run the application
python main.py
```



## 🎯 Use Cases

1. **Location-Based Recommendations**: Customer walks near a store → Agent suggests relevant items
2. **Personalized Offers**: Based on purchase history and preferences
3. **Weather-Aware Suggestions**: Hot day → Suggest cold beverages
4. **Reward Point Optimization**: Recommend items that maximize rewards
5. **Real-Time Store Availability**: Only suggest items in stock at nearby stores




