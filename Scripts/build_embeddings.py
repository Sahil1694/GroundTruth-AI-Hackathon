#!/usr/bin/env python3
"""
Build documents, chunk them, mask PII, and compute embeddings.
Outputs:
 - Dataset/chunks.jsonl   (one JSON per chunk: {"text":..., "chunk_id":..., "meta":{...}})
 - Dataset/metas.jsonl
 - Dataset/embeddings.npy
 - optional: faiss index creation (commented)
"""

import os
import json
import re
import uuid
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd

# Choose embedding backend: "sentence-transformers" or "openai"
EMBEDDING_BACKEND = "sentence-transformers"  # or "openai"
SENT_TRANSFORMER_MODEL = "all-MiniLM-L6-v2"  # fast & small
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"  # replace if you prefer another

# Chunking params (character-based approx)
CHUNK_SIZE_CHARS = 1600   # approx chunk size (~200-400 tokens depending on text)
CHUNK_OVERLAP_CHARS = 300

# Use absolute paths based on script location
SCRIPT_DIR = Path(__file__).parent.resolve()
BASE = SCRIPT_DIR / "Dataset"
OUT_DIR = BASE
CHUNKS_FILE = OUT_DIR / "chunks.jsonl"
METAS_FILE = OUT_DIR / "metas.jsonl"
EMB_FILE = OUT_DIR / "embeddings.npy"

# Utility: mask PII (phone numbers & emails)
PHONE_RE = re.compile(r'(\+?\d{1,3}[\s-]?)?(\d{4}[\d\-\s]{4,}\d+)')  # loose
EMAIL_RE = re.compile(r'[\w\.-]+@[\w\.-]+')

def mask_pii(text: str) -> str:
    text = EMAIL_RE.sub("[email_masked]", text)
    text = PHONE_RE.sub(lambda m: "[phone_masked]" + (m.group(2)[-4:] if m.group(2) else ""), text)
    return text

# Read CSV rows and convert to searchable text documents
def csv_row_to_text(row: pd.Series, source_name: str) -> tuple[str, dict]:
    # Customize per CSV schema
    if source_name.endswith("customers.csv"):
        text = (
            f"CustomerID: {row['customer_id']}. Name: {row.get('name','')}. "
            f"Prefs: {row.get('preferred_drinks','')}. Size: {row.get('preferred_size','')}. "
            f"Allergies: {row.get('allergies','')}. UsualTime: {row.get('usual_order_time','')}. "
            f"Rewards: {row.get('reward_points', '')}."
        )
        meta = {"source":"customers.csv", "customer_id": int(row['customer_id'])}
        return text, meta

    if source_name.endswith("stores.csv"):
        text = (
            f"StoreID: {row['store_id']}. Name: {row.get('store_name','')}. "
            f"Hours: {row.get('open_time','')}-{row.get('close_time','')}. "
            f"Offers: {row.get('current_offer','')}. Popular: {row.get('popular_items','')}."
        )
        meta = {"source":"stores.csv", "store_id": int(row['store_id']), "lat": row.get("latitude"), "lon": row.get("longitude")}
        return text, meta

    if source_name.endswith("customer_history.csv"):
        text = (
            f"OrderID: {row['order_id']}. CustomerID: {row['customer_id']}. "
            f"Item: {row['item']} ({row['size']}). Timestamp: {row['timestamp']}. Rating: {row.get('satisfaction_rating','')}"
        )
        meta = {"source":"customer_history.csv", "order_id": int(row['order_id']), "customer_id": int(row['customer_id'])}
        return text, meta

    # Fallback
    text = json.dumps(row.to_dict())
    return text, {"source": source_name}

# Extract text from PDFs using PyPDF2 (simple)
def extract_text_from_pdf(path: Path) -> str:
    try:
        import PyPDF2
    except ImportError:
        print(f"Warning: PyPDF2 not installed. Skipping PDF: {path.name}")
        return ""
    text = []
    with open(path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for p in reader.pages:
            try:
                page_text = p.extract_text() or ""
            except Exception:
                page_text = ""
            text.append(page_text)
    return "\n".join(text)

# Simple chunker (character-based)
def chunk_text(text: str, chunk_size=CHUNK_SIZE_CHARS, overlap=CHUNK_OVERLAP_CHARS):
    if len(text) <= chunk_size:
        yield text
        return
    start = 0
    step = chunk_size - overlap
    while start < len(text):
        chunk = text[start:start+chunk_size]
        yield chunk
        start += step

# Create docs list (text + meta)
def build_documents(base_path: Path):
    docs = []  # each item: (text, meta)
    # CSVs
    for csv_name in ["customers.csv", "stores.csv", "customer_history.csv"]:
        path = base_path / csv_name
        if not path.exists():
            continue
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            text, meta = csv_row_to_text(row, csv_name)
            text = mask_pii(text)
            docs.append((text, meta))

    # PDFs: both store_pdfs and customer_pdfs
    pdf_dirs = [base_path / "store_pdfs", base_path / "customer_pdfs"]
    for pd_dir in pdf_dirs:
        if not pd_dir.exists():
            continue
        for p in pd_dir.iterdir():
            if p.suffix.lower() not in [".pdf", ".txt"]:
                continue
            if p.suffix.lower() == ".txt":
                raw = p.read_text(encoding="utf-8", errors="ignore")
            else:
                raw = extract_text_from_pdf(p)
            raw = raw.strip()
            if not raw:
                continue
            raw = mask_pii(raw)
            # metadata: try to infer IDs from filename
            meta = {"source": str(p.parent.name), "filename": p.name}
            docs.append((raw, meta))

    return docs

# Embedding functions
def embed_with_sentence_transformers(texts, model_name=SENT_TRANSFORMER_MODEL):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    embs = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    # normalize for cosine similarity if desired
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms==0] = 1.0
    embs = embs / norms
    return embs

def embed_with_openai(texts, model_name=OPENAI_EMBEDDING_MODEL):
    import os
    import openai
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise ValueError("OPENAI_API_KEY not set in environment")
    openai.api_key = key
    embs = []
    BATCH = 32
    for i in tqdm(range(0, len(texts), BATCH), desc="OpenAI embed"):
        batch = texts[i:i+BATCH]
        resp = openai.Embeddings.create(model=model_name, input=batch)
        vecs = [r["embedding"] for r in resp["data"]]
        embs.extend(vecs)
    embs = np.array(embs, dtype=np.float32)
    # normalize
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms==0] = 1.0
    embs = embs / norms
    return embs

def main():
    # Check if Dataset directory exists
    if not BASE.exists():
        raise FileNotFoundError(f"Dataset directory not found at: {BASE}")
    
    print("Building docs from dataset...")
    docs = build_documents(BASE)
    print(f"Total raw docs found: {len(docs)}")

    # Expand docs into chunks
    chunks = []
    metas = []
    for text, meta in tqdm(docs, desc="Chunking docs"):
        # some docs are short single-line CSV rows; some are long PDFs
        for c in chunk_text(text):
            chunk_id = str(uuid.uuid4())
            chunk_meta = dict(meta)  # copy
            chunk_meta["chunk_id"] = chunk_id
            chunk_meta["char_length"] = len(c)
            chunks.append({"chunk_id": chunk_id, "text": c})
            metas.append(chunk_meta)

    print(f"Total chunks produced: {len(chunks)}")

    # Save chunks and metas (jsonl)
    with open(CHUNKS_FILE, "w", encoding="utf-8") as fch, open(METAS_FILE, "w", encoding="utf-8") as fmeta:
        for ch, m in zip(chunks, metas):
            fch.write(json.dumps({"chunk_id": ch["chunk_id"], "text": ch["text"]}, ensure_ascii=False) + "\n")
            fmeta.write(json.dumps(m, ensure_ascii=False) + "\n")

    # Prepare text list for embedding
    texts = [c["text"] for c in chunks]
    print("Computing embeddings using:", EMBEDDING_BACKEND)

    if EMBEDDING_BACKEND == "sentence-transformers":
        embs = embed_with_sentence_transformers(texts)
    elif EMBEDDING_BACKEND == "openai":
        embs = embed_with_openai(texts)
    else:
        raise ValueError("Unknown EMBEDDING_BACKEND")

    # Save embeddings numpy array (rows aligned with chunks.jsonl & metas.jsonl)
    np.save(EMB_FILE, embs.astype(np.float32))
    print("Saved embeddings to:", EMB_FILE)
    print("Saved chunks to:", CHUNKS_FILE)
    print("Saved metas to:", METAS_FILE)

    # Optional: create a FAISS index (uncomment if faiss-cpu is installed)
    try:
        import faiss
        dim = embs.shape[1]
        index = faiss.IndexFlatIP(dim)  # inner product on normalized vectors ~ cosine
        index.add(embs.astype(np.float32))
        faiss.write_index(index, str(OUT_DIR / "faiss_index.index"))
        print("FAISS index created at", OUT_DIR / "faiss_index.index")
    except Exception as e:
        print("Skipping FAISS index creation (faiss not available or error):", e)

if __name__ == "__main__":
    main()
