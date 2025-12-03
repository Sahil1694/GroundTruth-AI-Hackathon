# merge_chunks_metas.py
import json
from pathlib import Path

# Use absolute path based on script location
SCRIPT_DIR = Path(__file__).parent.resolve()
BASE = SCRIPT_DIR / "Dataset"
CHUNKS_F = BASE/"chunks.jsonl"
METAS_F  = BASE/"metas.jsonl"
OUT_MERGED = BASE/"chunks_meta.jsonl"

# load metas into map
metas_map = {}
with open(METAS_F, "r", encoding="utf-8") as f:
    for i, line in enumerate(f,1):
        line=line.strip()
        if not line: continue
        j = json.loads(line)
        cid = j.get("chunk_id")
        if cid in metas_map:
            # multiple metas for same chunk_id -> keep list
            if isinstance(metas_map[cid], list):
                metas_map[cid].append(j)
            else:
                metas_map[cid] = [metas_map[cid], j]
        else:
            metas_map[cid] = j

missing_meta_count = 0
duplicate_meta_count = 0
with open(CHUNKS_F, "r", encoding="utf-8") as fin, open(OUT_MERGED, "w", encoding="utf-8") as fout:
    for i, line in enumerate(fin, 1):
        line=line.strip()
        if not line: continue
        c = json.loads(line)
        cid = c.get("chunk_id")
        meta = metas_map.get(cid)
        # handle missing meta
        if meta is None:
            missing_meta_count += 1
            meta = {"chunk_id": cid, "source":"unknown", "note":"meta_missing"}
        # handle duplicate metas (if list)
        if isinstance(meta, list):
            duplicate_meta_count += 1
            # choose the first (you can refine this logic)
            meta = meta[0]
            meta["note"] = meta.get("note","") + ";;duplicate_meta_used_first"
        # normalize text (remove literal "nan")
        text = c.get("text","")
        if isinstance(text, str):
            text = text.replace("nan","").replace("None","").strip()
        merged = {"chunk_id": cid, "text": text, "meta": meta}
        fout.write(json.dumps(merged, ensure_ascii=False) + "\n")

print("Merged written to", OUT_MERGED)
print("Missing metas created:", missing_meta_count)
print("Duplicate meta entries used (first chosen):", duplicate_meta_count)
