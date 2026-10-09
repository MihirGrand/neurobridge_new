"""
Face tools - camera capture, DeepFace analysis, and face-identity management.

Depends on:
  - DeepFace (lazy-loaded on first use to avoid slow TensorFlow init at startup)
  - ChromaDB face_collection (passed in via init or imported from db module)
"""

import base64
import os
import uuid
from concurrent.futures import ThreadPoolExecutor

import chromadb
from pathlib import Path

import requests

# ── ChromaDB setup (face collection) ──────────────────────────────────────────
CHROMA_DIR = Path("./face_db")
CHROMA_DIR.mkdir(exist_ok=True)

chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
face_collection = chroma_client.get_or_create_collection(
    name="faces",
    metadata={"hnsw:space": "cosine"},  # cosine similarity for face embeddings
)

DEEPFACE_MODEL = "Facenet"  # 128-dim; fast and accurate enough for local use
SIMILARITY_THRESHOLD = 0.30  # cosine distance - lower = more similar

# ── Lazy DeepFace loader ───────────────────────────────────────────────────────

_deepface = None


def _get_deepface():
    """Lazy-load DeepFace so TensorFlow only initializes when first needed."""
    global _deepface
    if _deepface is None:
        print("[DeepFace] Loading DeepFace (first use)...", flush=True)
        from deepface import DeepFace as _df

        _deepface = _df
        print("[DeepFace] Loaded.", flush=True)
    return _deepface


# ── ChromaDB face helpers ──────────────────────────────────────────────────────


def _embed_image(image_path: str) -> list[float] | None:
    """Extract face embedding using DeepFace.represent(). Returns None if no face found."""
    DeepFace = _get_deepface()
    try:
        result = DeepFace.represent(
            img_path=image_path,
            model_name=DEEPFACE_MODEL,
            enforce_detection=True,
            detector_backend="opencv",
        )
        return result[0]["embedding"]  # first face in image
    except Exception as e:
        print(f"[DeepFace] embedding failed: {e}")
        return None


def search_face(embedding: list[float]) -> dict | None:
    """
    Search ChromaDB for the closest matching face.
    Returns the stored metadata if within threshold, else None.
    """
    if face_collection.count() == 0:
        return None

    results = face_collection.query(
        query_embeddings=[embedding],
        n_results=1,
        include=["metadatas", "distances"],
    )

    if not results["ids"][0]:
        return None

    distance = results["distances"][0][0]
    if distance <= SIMILARITY_THRESHOLD:
        meta = results["metadatas"][0][0]
        meta["_face_id"] = results["ids"][0][0]
        meta["_distance"] = round(distance, 4)
        return meta

    return None


def register_face(
    embedding: list[float], name: str | None = None, extra_meta: dict | None = None
) -> str:
    """
    Store a new face embedding in ChromaDB.
    Returns the new face_id.
    """
    face_id = str(uuid.uuid4())
    metadata = {
        "name": name or "unknown",
        "familiar": "false",
        "first_seen": str(uuid.uuid4()),  # placeholder timestamp-like id
    }
    if extra_meta:
        metadata.update({k: str(v) for k, v in extra_meta.items()})

    face_collection.add(
        ids=[face_id],
        embeddings=[embedding],
        metadatas=[metadata],
    )
    return face_id


def update_face_name(face_id: str, name: str):
    """Update the name stored for an existing face entry."""
    face_collection.update(
        ids=[face_id],
        metadatas=[{"name": name, "familiar": "true"}],
    )


def mark_face_familiar(face_id: str):
    """Mark a previously unknown face as familiar (seen before, name unknown)."""
    existing = face_collection.get(ids=[face_id], include=["metadatas"])
    if existing["ids"]:
        meta = existing["metadatas"][0]
        if meta.get("name") == "unknown":
            meta["familiar"] = "true"
            face_collection.update(ids=[face_id], metadatas=[meta])


# ── Public tool functions ──────────────────────────────────────────────────────


def get_image() -> str:
    image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "image.jpg")
    with open(image_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()


def capture_camera_frame() -> dict:
    """
    Capture a frame from the camera stream (currently a local placeholder image).
    Runs DeepFace analysis (emotion, age, gender) and face identity lookup via ChromaDB.
    Returns a structured dict with the base64 image + all analysis results.

    Optimised: analyze() and represent() run in parallel threads so face
    detection + attribute analysis overlaps with embedding extraction.
    """
    print("[capture_camera_frame] called", flush=True)

    # ── Capture from ESP32-CAM ─────────────────────────────────────────────────
    # ESP32_CAM_IP = "192.168.0.251"
    ESP32_CAM_IP = "10.96.42.251"
    ROTATE_90_CW = True

    params = {
        "framesize": 12,      # XGA 1024x768
        "quality": 4,
        "gainceiling": 1,    # 0 = 2x
        "led_intensity": 4
    }
    # Save to a temp file since DeepFace needs a file path
    image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "image.jpg")

    try:
        for var, val in params.items():
            requests.get(f"http://{ESP32_CAM_IP}/control", params={"var": var, "val": val}, timeout=5)

        response = requests.get(f"http://{ESP32_CAM_IP}/capture", timeout=10)
        response.raise_for_status()
        raw = response.content

        if ROTATE_90_CW:
            import numpy as np
            import cv2
            nparr = np.frombuffer(raw, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            cv2.imwrite(image_path, img)
            with open(image_path, "rb") as f:
                raw = f.read()
        else:
            with open(image_path, "wb") as f:
                f.write(raw)
    except Exception as e:
        print(f"[Camera] Failed to capture from {ESP32_CAM_IP}: {e}. Attempting first system camera...", flush=True)
        import cv2
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()

        if ret:
            cv2.imwrite(image_path, frame)
            with open(image_path, "rb") as f:
                raw = f.read()
        else:
            print("[Camera] No system camera available.", flush=True)
            # image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "image.jpg")
            # with open(image_path, "rb") as f:
            #     raw = f.read()
            raw = b""

    image_b64 = base64.b64encode(raw).decode() if raw else ""

    DeepFace = _get_deepface()

    # ── Helper closures for parallel execution ─────────────────────────────────
    def _run_analyze():
        """Attribute analysis (age, gender, emotion, race)."""
        try:
            analysis = DeepFace.analyze(
                img_path=image_path,
                actions=["age", "gender", "emotion"],
                enforce_detection=True,
                detector_backend="opencv",
                silent=True,
            )
            a = analysis[0]  # first face
            attrs = {
                "age": a.get("age"),
                "gender": a.get("dominant_gender"),
                "emotion": a.get("dominant_emotion"),
                "emotion_scores": {
                    k: round(v, 1) for k, v in a.get("emotion", {}).items()
                },
            }
            print(
                f"[DeepFace] Attributes: age={attrs['age']}, "
                f"gender={attrs['gender']}, emotion={attrs['emotion']}"
            )
            return attrs
        except Exception as e:
            print(f"[DeepFace] analyze() failed: {e}")
            return {"error": str(e)}

    def _run_embed():
        """Face embedding extraction."""
        return _embed_image(image_path)

    # ── Run both in parallel ───────────────────────────────────────────────────
    with ThreadPoolExecutor(max_workers=2) as pool:
        attr_future = pool.submit(_run_analyze)
        embed_future = pool.submit(_run_embed)
        attributes = attr_future.result()
        embedding = embed_future.result()

    # ── Face identity (embedding → ChromaDB) ──────────────────────────────────
    identity = {
        "recognized": False,
        "name": None,
        "familiar": False,
        "face_id": None,
        "is_new_face": False,
    }

    if embedding:
        match = search_face(embedding)
        if match:
            # Known face
            identity["recognized"] = True
            identity["face_id"] = match["_face_id"]
            identity["name"] = (
                match.get("name") if match.get("name") != "unknown" else None
            )
            identity["familiar"] = match.get("familiar") == "true"
            identity["similarity_distance"] = match.get("_distance")
            # Auto-mark as familiar if seen again
            mark_face_familiar(match["_face_id"])
            print(
                f"[FaceDB] Match found → name={identity['name']}, "
                f"familiar={identity['familiar']}, dist={match['_distance']}"
            )
        else:
            # New face - store it
            face_id = register_face(embedding)
            identity["is_new_face"] = True
            identity["face_id"] = face_id
            print(f"[FaceDB] New face registered → face_id={face_id}")
    else:
        print("[FaceDB] No embedding extracted; skipping identity lookup.")

    return {
        "image_b64": image_b64,
        "face_count": 1 if embedding else 0,
        "attributes": attributes,
        "identity": identity,
    }


def register_face_name(face_id: str, name: str) -> str:
    """
    Associate a name with a stored face_id in the face database.
    Call this when the user introduces themselves or tells you their name.
    """
    try:
        update_face_name(face_id, name)
        print(f"[FaceDB] Name '{name}' stored for face_id={face_id}")
        return f"Successfully registered name '{name}' for this face. I'll remember you next time!"
    except Exception as e:
        return f"Failed to register name: {e}"
