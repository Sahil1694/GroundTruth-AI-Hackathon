# recompute_embeddings_faiss.py
import json, numpy as np
from pathlib import Path
from tqdm import tqdm

# Use absolute path based on script location
SCRIPT_DIR = Path(__file__).parent.resolve()
BASE = SCRIPT_DIR / "Dataset"
CHUNKS_FILE = BASE/"chunks.jsonl"
METAS_FILE = BASE/"metas.jsonl"
EMB_OUT = BASE/"embeddings.npy"
try:
    from sentence_transformers import SentenceTransformer
except Exception as e:
    raise SystemExit("Please pip install sentence-transformers (and faiss-cpu if you want the index).")

MODEL = "all-MiniLM-L6-v2"  # fast and good
model = SentenceTransformer(MODEL)

texts = []
metas = []
with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
    for line in f:
        j = json.loads(line)
        texts.append(j.get("text",""))

# Load metas separately (optional, for reference)
with open(METAS_FILE, "r", encoding="utf-8") as f:
    for line in f:
        j = json.loads(line)
        metas.append(j)

print("Computing embeddings for", len(texts), "chunks ...")
embs = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
# normalize for cosine
norms = np.linalg.norm(embs, axis=1, keepdims=True)
norms[norms==0] = 1.0
embs = embs / norms

np.save(EMB_OUT, embs.astype(np.float32))
print("Saved embeddings to", EMB_OUT)

# Optional: build FAISS index
try:
    import faiss
    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embs.astype(np.float32))
    faiss.write_index(index, str(BASE/"faiss_index.index"))
    print("FAISS index written to", BASE/"faiss_index.index")
except Exception as e:
    print("Faiss not available or error building index:", e)
    print("You still have embeddings.npy and chunks_meta.jsonl to use with other vector DBs.")
