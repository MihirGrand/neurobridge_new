import csv
import math
import os
import tkinter as tk
import requests
import numpy as np
from PIL import Image, ImageTk
import cv2
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as cosine_sim

# ── Config ────────────────────────────────────────────────────────────────────

CANVAS_W = 128
CANVAS_H = 64
BOB_AMPLITUDE = 4   # px up/down
BOB_SPEED = 0.08    # radians per frame
FPS = 30
SCALE = 4           # preview window scale (128x64 -> 512x256)

PNG_CDN = "https://cdn.jsdelivr.net/gh/hfg-gmuend/openmoji@master/color/618x618"
CSV_URL = "https://raw.githubusercontent.com/hfg-gmuend/openmoji/master/data/openmoji.csv"

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "openmoji_cache")
META_PATH = os.path.join(CACHE_DIR, "openmoji.csv")
IMG_DIR = os.path.join(CACHE_DIR, "png")
XBM_DIR = os.path.join(CACHE_DIR, "xbm")

# ── Data loading & caching ────────────────────────────────────────────────────

def load_metadata() -> list:
    """Load OpenMoji CSV metadata. Download if not cached."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    if not os.path.exists(META_PATH):
        print("  Downloading OpenMoji metadata...")
        r = requests.get(CSV_URL)
        r.raise_for_status()
        with open(META_PATH, "w", encoding="utf-8") as f:
            f.write(r.text)

    emojis = []
    with open(META_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("skintone_combination", ""):
                continue
            hexcode = row["hexcode"].strip()
            annotation = row["annotation"].strip()
            tags = row.get("tags", "").strip()
            openmoji_tags = row.get("openmoji_tags", "").strip()
            emojis.append({
                "hexcode": hexcode,
                "annotation": annotation,
                "tags": tags,
                "openmoji_tags": openmoji_tags,
            })
    return emojis


def download_image(hexcode: str) -> Image.Image:
    """Download an emoji PNG from CDN, cache locally."""
    os.makedirs(IMG_DIR, exist_ok=True)
    path = os.path.join(IMG_DIR, f"{hexcode}.png")
    if not os.path.exists(path):
        url = f"{PNG_CDN}/{hexcode}.png"
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            raise ValueError(f"Failed to download {hexcode}")
        with open(path, "wb") as f:
            f.write(r.content)
    return Image.open(path).convert("RGBA")


# ── Embeddings (pre-computed for fast search) ─────────────────────────────────

class EmojiSearcher:
    """TF-IDF primary + semantic embedding secondary for fast, accurate search.

    TF-IDF handles exact/substring word matching ("happy" matches "happy face",
    does NOT match "share").  Sentence-transformer embeddings provide a small
    boost for synonyms.  Both are pre-computed so search is instant.
    """

    def __init__(self, metadata: list):
        self.metadata = metadata
        self.annotations = [e["annotation"] for e in metadata]

        # Build search texts: annotation + tags weighted by repetition
        # to increase their TF-IDF weight
        texts = []
        for e in metadata:
            t = e["annotation"]
            if e["tags"]:
                t += " " + e["tags"]
            if e["openmoji_tags"]:
                t += " " + e["openmoji_tags"]
            texts.append(t)

        # ── TF-IDF (primary) ─────────────────────────────────────────────
        # analyzer=char_wb catches substring matches: "happy" matches
        # "happy" inside "happy face" even without word boundary issues.
        self.tfidf = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            token_pattern=r"(?u)\b\w+\b",
        )
        self.tfidf_matrix = self.tfidf.fit_transform(texts)

        # ── Semantic embeddings (secondary boost) ─────────────────────────
        print("  Computing semantic embeddings (one-time)...")
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.embed_matrix = self.embedder.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        )
        print(f"  {len(metadata)} entries ready.")

    def search(self, query: str, top_n: int = 5) -> list:
        """Search by query string, return ranked list of emoji dicts."""
        # ── TF-IDF score (primary - handles word matching) ────────────────
        q_tfidf = self.tfidf.transform([query])
        tfidf_scores = cosine_sim(q_tfidf, self.tfidf_matrix)[0]

        # ── Semantic score (secondary - handles synonyms) ─────────────────
        q_embed = self.embedder.encode([query], normalize_embeddings=True)
        sem_scores = cosine_sim(q_embed, self.embed_matrix)[0]

        # Combine: TF-IDF dominant, semantic as tiebreaker
        final_scores = tfidf_scores * 0.7 + sem_scores * 0.3

        # Apply lightweight post-processing
        query_lower = query.lower().strip()
        query_words = set(query_lower.split())
        for i, emoji in enumerate(self.metadata):
            ann_lower = self.annotations[i].lower()
            ann_words = set(ann_lower.split())

            # Exact match - overwhelming boost
            if ann_lower == query_lower:
                final_scores[i] += 2.0

            # Prefer shorter annotations when scores are close
            final_scores[i] -= len(self.annotations[i]) * 0.001

        # Get top indices
        top_idx = np.argsort(final_scores)[::-1][:top_n]
        return [self.metadata[i] for i in top_idx]


# ── Image prep ────────────────────────────────────────────────────────────────

def prepare_icon(img: Image.Image) -> Image.Image:
    """Extract black lines from OpenMoji icon as white-on-black 1-bit image.

    OpenMoji uses clear black outlines and internal line art.  We detect
    dark pixels (luminance < 50) and render them white on black, then
    quantize to 1-bit for the final XBM output.
    """
    img = img.convert("RGBA")
    arr = np.array(img)

    r_ch, g_ch, b_ch, a = arr[..., 0], arr[..., 1], arr[..., 2], arr[..., 3]

    gray = (0.299 * r_ch + 0.587 * g_ch + 0.114 * b_ch).astype(np.float64)
    visible = a > 30

    # Dark line detection
    dark_mask = visible & (gray < 50)

    line_img = np.zeros_like(gray, dtype=np.uint8)
    line_img[dark_mask] = 255

    # Dilate to thicken for small OLED readability
    kernel = np.ones((2, 2), np.uint8)
    line_img = cv2.dilate(line_img, kernel, iterations=1)

    out = Image.fromarray(line_img, mode="L")

    # Fit inside canvas, maintaining aspect ratio
    out.thumbnail((CANVAS_W, CANVAS_H), Image.LANCZOS)

    # Center on black canvas
    canvas = Image.new("L", (CANVAS_W, CANVAS_H), 0)
    x = (CANVAS_W - out.width) // 2
    y = (CANVAS_H - out.height) // 2
    canvas.paste(out, (x, y))

    return canvas


def image_to_xbm(img: Image.Image, name: str = "oled") -> str:
    """Convert a 1-bit PIL image to XBM format string and save file.
    Returns the file path."""
    os.makedirs(XBM_DIR, exist_ok=True)
    path = os.path.join(XBM_DIR, f"{name}.xbm")
    # Quantize to 1-bit (black=0, white=1)
    bw = img.convert("1")
    bw.save(path)
    return path


# ── OLED simulator ────────────────────────────────────────────────────────────

class OLEDSimulator:
    def __init__(self, searcher: EmojiSearcher):
        self.searcher = searcher
        self.root = tk.Tk()
        self.root.title("OLED Preview – OpenMoji XBM (128x64)")
        self.root.configure(bg="#111")
        self.root.resizable(False, False)

        tk.Label(
            self.root, text="OLED SIMULATOR (OPENMOJI → XBM)", bg="#111", fg="#444",
            font=("Courier", 9)
        ).pack(pady=(10, 4))

        bezel = tk.Frame(self.root, bg="#222", bd=4, relief="ridge")
        bezel.pack(padx=20, pady=4)

        self.canvas = tk.Canvas(
            bezel,
            width=CANVAS_W * SCALE,
            height=CANVAS_H * SCALE,
            bg="#0a0a0a",
            highlightthickness=0,
        )
        self.canvas.pack(padx=8, pady=8)

        bar = tk.Frame(self.root, bg="#111")
        bar.pack(pady=(4, 12))
        self.entry = tk.Entry(
            bar, font=("Courier", 11), bg="#1a1a1a", fg="#eee",
            insertbackground="#eee", relief="flat", width=20
        )
        self.entry.pack(side="left", padx=(0, 6), ipady=4, ipadx=6)
        self.entry.bind("<Return>", self._on_search)
        tk.Button(
            bar, text="SEARCH", font=("Courier", 9, "bold"),
            bg="#1db954", fg="#000", relief="flat", cursor="hand2",
            command=self._on_search, padx=8
        ).pack(side="left")

        self.status = tk.Label(
            self.root, text=f"{len(searcher.metadata)} emojis ready. Type a query.",
            bg="#111", fg="#555", font=("Courier", 8)
        )
        self.status.pack(pady=(0, 4))

        self.xbm_label = tk.Label(
            self.root, text="", bg="#111", fg="#1db954", font=("Courier", 8)
        )
        self.xbm_label.pack(pady=(0, 8))

        self._icon_image = None
        self._xbm_path = None
        self._tk_image = None
        self._bob_t = 0.0
        self._animating = False

    def _on_search(self, event=None):
        query = self.entry.get().strip()
        if not query:
            return
        self.status.config(text="Searching...", fg="#888")
        self.root.update()
        try:
            candidates = self.searcher.search(query, top_n=5)
            if not candidates:
                raise ValueError(f"No emojis found for '{query}'")
            best = candidates[0]
            print(f"  Best match: {best['annotation']} ({best['hexcode']})")

            img = download_image(best["hexcode"])
            self._icon_image = prepare_icon(img)
            self._xbm_path = image_to_xbm(self._icon_image, best["hexcode"])

            self.status.config(
                text=f"{best['annotation']} ({best['hexcode']})",
                fg="#1db954",
            )
            self.xbm_label.config(text=f"XBM: {self._xbm_path}")

            if not self._animating:
                self._animating = True
                self._bob_t = 0.0
                self._animate()
        except Exception as e:
            self.status.config(text=f"Error: {e}", fg="#e74c3c")
            self.xbm_label.config(text="")

    def _animate(self):
        if self._icon_image is None:
            return

        y_offset = int(BOB_AMPLITUDE * math.sin(self._bob_t))
        self._bob_t += BOB_SPEED

        frame = Image.new("L", (CANVAS_W, CANVAS_H), 0)
        frame.paste(self._icon_image, (0, y_offset))

        frame_big = frame.resize(
            (CANVAS_W * SCALE, CANVAS_H * SCALE), Image.NEAREST
        )

        # Colorize: black bg → #0a0a0a, white pixels → OLED green-white
        frame_rgb = Image.new("RGB", frame_big.size, (10, 10, 10))
        lit = Image.new("RGB", frame_big.size, (210, 255, 220))
        frame_rgb.paste(lit, mask=frame_big)

        self._tk_image = ImageTk.PhotoImage(frame_rgb)
        self.canvas.create_image(0, 0, anchor="nw", image=self._tk_image)

        self.root.after(1000 // FPS, self._animate)

    def run(self):
        self.root.mainloop()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading OpenMoji metadata...")
    metadata = load_metadata()
    print(f"Loaded {len(metadata)} emojis.")
    searcher = EmojiSearcher(metadata)
    sim = OLEDSimulator(searcher)
    sim.run()
