# A_retrieve_with_store_priority.py
from collections import defaultdict
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss, json
from pathlib import Path

# Use absolute path based on script location (go up one level from Scripts/)
SCRIPT_DIR = Path(__file__).parent.parent.resolve()
BASE = SCRIPT_DIR / "Dataset"
CHUNKS_META = BASE/"chunks_meta.jsonl"
INDEX_FILE = BASE/"faiss_index.index"
MODEL_NAME = "all-MiniLM-L6-v2"

# load chunks
chunks = []
with open(CHUNKS_META, "r", encoding="utf-8") as f:
    for line in f:
        chunks.append(json.loads(line))
print("Loaded chunks:", len(chunks))

# build store->indices map
store_to_indices = defaultdict(list)
for i, ch in enumerate(chunks):
    meta = ch.get("meta",{})
    sid = meta.get("store_id")
    if sid:
        store_to_indices[int(sid)].append(i)

# load faiss & model
index = faiss.read_index(str(INDEX_FILE))
model = SentenceTransformer(MODEL_NAME)

def retrieve_with_store_priority(query, detected_store_id=None, k=50, top_k=6):
    q_emb = model.encode([query], convert_to_numpy=True)
    q_emb = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)
    D, I = index.search(q_emb.astype("float32"), k)

    candidates = []
    for score, idx in zip(D[0], I[0]):
        if idx < 0 or idx >= len(chunks): continue
        item = chunks[idx]
        meta = item.get("meta", {})
        is_store = (meta.get("store_id") == detected_store_id) or ("store" in str(meta.get("source","")).lower())
        candidates.append({"idx": idx, "score": float(score), "meta": meta, "text": item["text"], "is_store": is_store})

    # boost store matches
    for c in candidates:
        if detected_store_id and c["meta"].get("store_id")==detected_store_id:
            c["score"] += 0.08
        if "store" in str(c["meta"].get("source","")).lower():
            c["score"] += 0.05

    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[:top_k]

# quick test
q = "customer likes hot cocoa and is near a store"
print(retrieve_with_store_priority(q, detected_store_id=None, k=100, top_k=10))
