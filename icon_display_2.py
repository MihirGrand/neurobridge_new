import requests
import cv2
import json
import math
import time
import tkinter as tk
import numpy as np
from PIL import Image, ImageTk
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from io import BytesIO

BASE_URL = "https://piixes.com"
SEARCH_HEADERS = {
    "Content-Type": "application/json",
    "Next-Action": "606c5853bbb3b4f77319c196eaaaef7715942cba88",
}

CANVAS_W = 128
CANVAS_H = 64
BOB_AMPLITUDE = 4   # px up/down
BOB_SPEED = 0.08    # radians per frame
FPS = 30
SCALE = 4           # preview window scale (128x64 -> 512x256)

embedder = SentenceTransformer("all-MiniLM-L6-v2")


# ── Search & ranking ──────────────────────────────────────────────────────────

def search_icons(query: str) -> list:
    short_query = query.split()[0]
    r = requests.post(BASE_URL, json=[short_query], headers=SEARCH_HEADERS)
    r.raise_for_status()
    for line in r.text.splitlines():
        if line.startswith("1:"):
            return json.loads(line[2:])
    return []


def rank_icons(icons: list, query: str, prefer_pixel: bool = True) -> list:
    query_vec = embedder.encode([query])
    scored = []
    for icon in icons:
        text = icon["name"].replace("-", " ") + " " + " ".join(icon.get("_tags", []))
        icon_vec = embedder.encode([text])
        score = cosine_similarity(query_vec, icon_vec)[0][0]
        if prefer_pixel and "pixel" in icon["name"]:
            score += 0.15
        scored.append((score, icon))
    scored.sort(reverse=True, key=lambda x: x[0])
    return [icon for _, icon in scored]


def get_best_icon(query: str) -> dict:
    print(f"  Searching for '{query}'...")
    icons = search_icons(query)
    if not icons:
        raise ValueError(f"No icons found for '{query}'")
    ranked = rank_icons(icons, query)
    best = ranked[0]
    print(f"  Best match: {best['name']}")
    return best


# ── Image prep ────────────────────────────────────────────────────────────────

def download_icon(icon: dict, size: int = 128) -> Image.Image:
    url = f"{BASE_URL}/api/icon/{size}/{icon['name']}.png"
    r = requests.get(url)
    r.raise_for_status()
    return Image.open(BytesIO(r.content)).convert("RGBA")


def prepare_icon(img: Image.Image) -> Image.Image:
    """Outline style: detect prominent edges and render them white on black.
    Works for any icon color by finding boundaries in both alpha and luminance."""
    import cv2

    img = img.convert("RGBA")
    arr = np.array(img)

    r, g, b, a = arr[..., 0], arr[..., 1], arr[..., 2], arr[..., 3]

    # Edge source 1: alpha boundary (outer shape outline)
    alpha_edges = cv2.Canny(a, 30, 100)

    # Edge source 2: internal luminance edges - pre-multiply by alpha so
    # transparent areas don't produce fake edges at the border
    alpha_f = a / 255.0
    lum = (0.299 * r + 0.587 * g + 0.114 * b) * alpha_f
    lum_uint8 = np.clip(lum, 0, 255).astype(np.uint8)
    lum_edges = cv2.Canny(lum_uint8, 40, 120)

    # Combine both edge maps
    combined = np.clip(alpha_edges.astype(np.int32) + lum_edges.astype(np.int32), 0, 255).astype(np.uint8)

    # Dilate slightly so edges are visible at small canvas size
    kernel = np.ones((2, 2), np.uint8)
    combined = cv2.dilate(combined, kernel, iterations=1)

    edge_img = Image.fromarray(combined, mode="L")

    # Fit inside canvas
    edge_img.thumbnail((CANVAS_W, CANVAS_H), Image.LANCZOS)
    edge_img = edge_img.point(lambda p: 255 if p > 30 else 0)

    # Center on black canvas
    canvas = Image.new("L", (CANVAS_W, CANVAS_H), 0)
    x = (CANVAS_W - edge_img.width) // 2
    y = (CANVAS_H - edge_img.height) // 2
    canvas.paste(edge_img, (x, y))

    return canvas


# ── OLED simulator ────────────────────────────────────────────────────────────

class OLEDSimulator:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("OLED Preview (128×64)")
        self.root.configure(bg="#111")
        self.root.resizable(False, False)

        # Label above
        tk.Label(
            self.root, text="OLED SIMULATOR", bg="#111", fg="#444",
            font=("Courier", 9)
        ).pack(pady=(10, 4))

        # OLED bezel frame
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

        # Search bar
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
            self.root, text="Type a query and press Enter",
            bg="#111", fg="#555", font=("Courier", 8)
        )
        self.status.pack(pady=(0, 8))

        self._icon_image = None   # prepared 'L' image
        self._tk_image = None     # current PhotoImage
        self._bob_t = 0.0
        self._animating = False

    def _on_search(self, event=None):
        query = self.entry.get().strip()
        if not query:
            return
        self.status.config(text="Searching...", fg="#888")
        self.root.update()
        try:
            icon = get_best_icon(query)
            raw = download_icon(icon, size=128)
            self._icon_image = prepare_icon(raw)
            self.status.config(text=f"Showing: {icon['name']}", fg="#1db954")
            if not self._animating:
                self._animating = True
                self._bob_t = 0.0
                self._animate()
        except Exception as e:
            self.status.config(text=f"Error: {e}", fg="#e74c3c")

    def _animate(self):
        if self._icon_image is None:
            return

        y_offset = int(BOB_AMPLITUDE * math.sin(self._bob_t))
        self._bob_t += BOB_SPEED

        # Render frame: shift icon vertically
        frame = Image.new("L", (CANVAS_W, CANVAS_H), 0)
        frame.paste(self._icon_image, (0, y_offset))

        # Scale up for visibility
        frame_big = frame.resize(
            (CANVAS_W * SCALE, CANVAS_H * SCALE), Image.NEAREST
        )

        # Colorize: black bg → #0a0a0a, white pixels → OLED green-white
        frame_rgb = Image.new("RGB", frame_big.size, (10, 10, 10))
        lit = Image.new("RGB", frame_big.size, (210, 255, 220))  # soft OLED glow
        frame_rgb.paste(lit, mask=frame_big)

        self._tk_image = ImageTk.PhotoImage(frame_rgb)
        self.canvas.create_image(0, 0, anchor="nw", image=self._tk_image)

        self.root.after(1000 // FPS, self._animate)

    def run(self):
        self.root.mainloop()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sim = OLEDSimulator()
    sim.run()
