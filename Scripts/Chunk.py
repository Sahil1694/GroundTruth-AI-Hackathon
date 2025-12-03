# check_ids.py
import json
from pathlib import Path

# Use absolute path based on script location
SCRIPT_DIR = Path(__file__).parent.resolve()
BASE = SCRIPT_DIR / "Dataset"
CHUNKS_F = BASE/"chunks.jsonl"
METAS_F  = BASE/"metas.jsonl"

def load_ids(path, id_key="chunk_id"):
    ids = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f,1):
            line=line.strip()
            if not line: continue
            try:
                j = json.loads(line)
            except Exception as e:
                print(f"PARSE ERROR {path} line {i}: {e}")
                raise
            if id_key not in j:
                print(f"WARNING: no '{id_key}' in line {i} of {path}")
            ids.append(j.get(id_key))
    return ids

chunks_ids = load_ids(CHUNKS_F)
metas_ids  = load_ids(METAS_F)

set_chunks = set(chunks_ids)
set_metas  = set(metas_ids)

print("counts -> chunks:", len(chunks_ids), ", metas:", len(metas_ids))
only_in_chunks = sorted(list(set_chunks - set_metas))
only_in_metas  = sorted(list(set_metas - set_chunks))

print("Only in chunks (count):", len(only_in_chunks))
print("Only in metas  (count):", len(only_in_metas))

if only_in_chunks:
    print("Sample chunk_ids present only in chunks.jsonl:", only_in_chunks[:20])
if only_in_metas:
    print("Sample chunk_ids present only in metas.jsonl:", only_in_metas[:20])
