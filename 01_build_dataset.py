"""
Build a Persian-miniature training set from The Met's open-access API (CC0, no key).
Downloads public-domain Safavid / Shahnameh folios, resizes them, and writes a
caption .txt next to each image for the LoRA trainer.

    python 01_build_dataset.py

Output:  ./dataset/0001.jpg + ./dataset/0001.txt , ...
"""

import os, time, requests
from io import BytesIO
from PIL import Image

SEARCH = "https://collectionapi.metmuseum.org/public/collection/v1/search"
OBJECT = "https://collectionapi.metmuseum.org/public/collection/v1/objects/{}"

QUERIES = [
    "Shahnama",
    "Shahnameh",
    "Safavid painting",
    "Persian miniature",
    "Folio illustrated manuscript Iran",
]

OUT      = "dataset"
TARGET   = 45            # 30-50 images is plenty for a style LoRA
MAX_SIDE = 1024
TRIGGER  = "shahnameh persian miniature style"

os.makedirs(OUT, exist_ok=True)


def met_search(q):
    r = requests.get(SEARCH, params={"q": q, "hasImages": "true"}, timeout=30)
    r.raise_for_status()
    return r.json().get("objectIDs") or []


def fetch_object(oid):
    r = requests.get(OBJECT.format(oid), timeout=30)
    return r.json() if r.status_code == 200 else None


seen, saved = set(), 0
for q in QUERIES:
    if saved >= TARGET:
        break
    try:
        ids = met_search(q)
    except Exception as e:
        print("search failed:", q, e); continue

    for oid in ids:
        if saved >= TARGET or oid in seen:
            continue
        seen.add(oid)
        obj = fetch_object(oid)
        time.sleep(0.15)                      # be polite to the API
        if not obj or not obj.get("isPublicDomain"):
            continue
        url = obj.get("primaryImage")
        if not url:
            continue

        # keep it Persian / Islamic material
        tag = (obj.get("culture") or "") + (obj.get("department") or "")
        if not any(k in tag for k in ("Islam", "Iran", "Pers")):
            continue

        try:
            raw = requests.get(url, timeout=60).content
            im  = Image.open(BytesIO(raw)).convert("RGB")
        except Exception as e:
            print("img failed:", oid, e); continue

        w, h = im.size
        s = MAX_SIDE / max(w, h)
        if s < 1:
            im = im.resize((int(w * s), int(h * s)), Image.LANCZOS)

        saved += 1
        stem  = f"{saved:04d}"
        im.save(os.path.join(OUT, stem + ".jpg"), quality=95)
        title = (obj.get("title") or "Persian manuscript painting").strip().rstrip(".")
        with open(os.path.join(OUT, stem + ".txt"), "w") as f:
            f.write(f"{TRIGGER}. {title}.")     # style half = fixed trigger, content half = Met title
        print(f"[{saved}/{TARGET}] {title[:60]}")

print(f"\nDone. {saved} images in ./{OUT}/")
print("Skim the captions before training — Met titles are the 'content' half.")
print("Upgrade path: swap titles for Florence-2 / BLIP captions for richer content text.")
