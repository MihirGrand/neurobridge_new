import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import base64
import json
import logging
import tempfile
import uuid
from pathlib import Path

logging.getLogger("tensorflow").setLevel(logging.ERROR)
import datetime as dt
import json
import os
import time
from typing import Optional

import chromadb
import numpy as np
import requests
from dotenv import load_dotenv
from google.auth.transport.requests import Request
from openai import OpenAI

from tts_engine_alt import StreamingTTS

load_dotenv()

API_KEY = os.getenv("LLM_API_KEY")
BASE_URL = os.getenv("LLM_BASE_URL")
MODEL = os.getenv("LLM_MODEL")

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# ── ChromaDB setup ─────────────────────────────────────────────────────────────
CHROMA_DIR = Path("./face_db")
CHROMA_DIR.mkdir(exist_ok=True)

chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
face_collection = chroma_client.get_or_create_collection(
    name="faces",
    metadata={"hnsw:space": "cosine"},  # cosine similarity for face embeddings
)

memory_collection = chroma_client.get_or_create_collection(
    name="memory",
    metadata={"hnsw:space": "cosine"},
)

DEEPFACE_MODEL = "Facenet"  # 128-dim; fast and accurate enough for local use
SIMILARITY_THRESHOLD = 0.30  # cosine distance - lower = more similar

# ──────────────────────────────────────────────
# Google Auth Helper (shared across Google tools)
# ──────────────────────────────────────────────

# Scopes your agent needs - add more here if you expand tooling.
#   - calendar  : read & write events
#   - tasks     : read & write tasks
#   - gmail     : read & send mail, modify labels
GOOGLE_SCOPES = [
    "https://www.googleapis.com/auth/calendar",
    "https://www.googleapis.com/auth/tasks",
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.modify",
]

# Module-level cache so the auth flow only runs once per process.
_cached_creds = None


def _get_token_path() -> str:
    return os.getenv("GOOGLE_TOKEN_PATH", "token.json")


def _get_credentials_path() -> str:
    return os.getenv("GOOGLE_CREDENTIALS_PATH", "credentials.json")


def setup_google_auth() -> None:
    """
    One-time setup: authenticate with Google and save the token.
    Call this at the start of your application (or it will be called
    automatically on first use by _get_google_credentials).

    Flow:
      1. Reads OAuth client secrets from credentials.json
      2. If token.json exists → loads & refreshes it
      3. If token.json is missing/expired → opens browser for consent
      4. Saves the refreshed/new token back to token.json

    Setup steps (only needed once):
      a. Go to https://console.cloud.google.com/apis/credentials
      b. Create a project (or pick existing)
      c. Create an "OAuth client ID" (Desktop app type)
      d. Download the JSON → save as credentials.json
      e. Enable APIs: Google Calendar, Tasks API, Gmail API
      f. Run this function → it opens a browser window for you to authorise
    """
    global _cached_creds

    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow

    token_path = _get_token_path()
    creds_path = _get_credentials_path()

    creds = None

    # 1. Try loading an existing token
    if os.path.exists(token_path):
        try:
            creds = Credentials.from_authorized_user_file(token_path, GOOGLE_SCOPES)
        except Exception:
            creds = None  # corrupt or incompatible token → re-auth

    # 2. If no valid creds, run the browser-based OAuth flow
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            # Token expired but we have a refresh token - refresh silently
            try:
                creds.refresh(Request())
            except Exception:
                # Refresh failed (revoked, password changed, etc.) → full re-auth
                creds = None

        if not creds or not creds.valid:
            if not os.path.exists(creds_path):
                raise FileNotFoundError(
                    f"Missing '{creds_path}'.\n\n"
                    "To set up Google auth, follow these steps:\n"
                    "  1. Go to https://console.cloud.google.com/apis/credentials\n"
                    "  2. Create an OAuth 2.0 Client ID (type: Desktop app)\n"
                    "  3. Enable these APIs: Google Calendar, Tasks, Gmail\n"
                    "  4. Download the JSON and save it as 'credentials.json'\n"
                    "     (or set GOOGLE_CREDENTIALS_PATH env var)\n"
                    "  5. Run setup_google_auth() again\n"
                )

            flow = InstalledAppFlow.from_client_secrets_file(creds_path, GOOGLE_SCOPES)

            # ── IMPORTANT ──────────────────────────────────────────
            # Default run_local_server() opens a browser tab on the
            # same machine.  If your agent runs on a headless server /
            # Docker container, use run_console() instead, which prints
            # the auth URL to the terminal for you to open manually.
            # ───────────────────────────────────────────────────────
            try:
                creds = flow.run_local_server(port=0)
            except Exception:
                # Fallback: if no browser available, use copy-paste flow
                creds = flow.run_console()

    # 3. Persist token for future runs
    with open(token_path, "w") as f:
        f.write(creds.to_json())

    _cached_creds = creds
    print(f"✅ Google auth successful. Token saved to {token_path}")


def _get_google_credentials():
    """
    Return valid Google credentials, auto-running setup_google_auth()
    on the very first call if no token.json exists yet.
    """
    global _cached_creds

    if _cached_creds is not None:
        # Check if cached creds are still valid
        if _cached_creds.valid:
            return _cached_creds
        if _cached_creds.expired and _cached_creds.refresh_token:
            try:
                _cached_creds.refresh(Request())
                # Update the token file with refreshed creds
                with open(_get_token_path(), "w") as f:
                    f.write(_cached_creds.to_json())
                return _cached_creds
            except Exception as refresh_err:
                # Refresh failed (revoked, password changed, etc.) → re-auth
                _cached_creds = None

    # First run or token invalid → kick off the full auth flow
    setup_google_auth()
    return _cached_creds


def _build_service(api_name: str, api_version: str):
    """Return an authorised Google API service object."""
    from googleapiclient.discovery import build

    creds = _get_google_credentials()
    return build(api_name, api_version, credentials=creds)


def get_weather_aqi(location: str = "") -> str:
    """
    Fetch current weather and air quality data using ipinfo.io for geolocation
    and Open-Meteo for weather + AQI. No API keys required.

    If a location is provided, it is used as a display label. The actual
    data is always based on the server's IP geolocation (ipinfo.io), so the
    weather/AQI reflects where this agent is running.

    If no location is provided, ipinfo.io also supplies the city and region.

    Args:
        location: Optional display label, e.g. "Mumbai" or "Home". If empty,
                  the city from ipinfo.io is used.

    Returns:
        A formatted string with weather details and AQI information.
    """
    # --- Step 1: Get lat/lon from ipinfo.io ---
    try:
        ipinfo_headers = {}
        ipinfo_key = os.getenv("IPINFO_API_KEY")
        if ipinfo_key:
            ipinfo_headers["Authorization"] = f"Bearer {ipinfo_key}"

        ip_resp = requests.get(
            "https://ipinfo.io/json", headers=ipinfo_headers, timeout=10
        )
        ip_resp.raise_for_status()
        ip_data = ip_resp.json()
    except requests.RequestException as e:
        return f"ERROR: Failed to get location from ipinfo.io: {e}"

    loc_str = ip_data.get("loc", "")
    if not loc_str:
        return "ERROR: ipinfo.io did not return location data."

    lat, lon = loc_str.split(",")
    lat = float(lat)
    lon = float(lon)

    # Use provided location or fall back to ipinfo city/region
    display_city = location or ip_data.get("city", "Unknown")
    display_region = ip_data.get("region", "")
    display_country = ip_data.get("country", "")
    display_loc = f"{display_city}, {display_region}, {display_country}"

    # --- Step 2: Fetch weather from Open-Meteo ---
    weather_text = ""
    try:
        weather_resp = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "current": (
                    "temperature_2m,relative_humidity_2m,apparent_temperature,"
                    "precipitation,weather_code,wind_speed_10m,wind_direction_10m,"
                    "surface_pressure"
                ),
                "timezone": "auto",
            },
            timeout=10,
        )
        weather_resp.raise_for_status()
        w = weather_resp.json()
        c = w.get("current", {})

        # Open-Meteo WMO weather codes -> readable descriptions
        weather_codes = {
            0: "Clear sky",
            1: "Mainly clear",
            2: "Partly cloudy",
            3: "Overcast",
            45: "Foggy",
            48: "Rime fog",
            51: "Light drizzle",
            53: "Drizzle",
            55: "Dense drizzle",
            56: "Freezing drizzle",
            57: "Dense freezing drizzle",
            61: "Slight rain",
            63: "Moderate rain",
            65: "Heavy rain",
            66: "Freezing rain",
            67: "Heavy freezing rain",
            71: "Slight snow",
            73: "Moderate snow",
            75: "Heavy snow",
            77: "Snow grains",
            80: "Slight rain showers",
            81: "Moderate rain showers",
            82: "Violent rain showers",
            85: "Slight snow showers",
            86: "Heavy snow showers",
            95: "Thunderstorm",
            96: "Thunderstorm with hail",
            99: "Thunderstorm with heavy hail",
        }
        code = c.get("weather_code", 0)
        condition = weather_codes.get(code, f"Unknown ({code})")

        # Wind direction in compass
        def _wind_dir(deg):
            dirs = [
                "N",
                "NNE",
                "NE",
                "ENE",
                "E",
                "ESE",
                "SE",
                "SSE",
                "S",
                "SSW",
                "SW",
                "WSW",
                "W",
                "WNW",
                "NW",
                "NNW",
            ]
            if deg is None:
                return "N/A"
            return dirs[round(deg / 22.5) % 16]

        temp = c.get("temperature_2m", "N/A")
        feels_like = c.get("apparent_temperature", "N/A")
        humidity = c.get("relative_humidity_2m", "N/A")
        pressure = c.get("surface_pressure", "N/A")
        wind_speed = c.get("wind_speed_10m", "N/A")
        wind_dir = _wind_dir(c.get("wind_direction_10m"))
        precip = c.get("precipitation", 0)

        weather_text = (
            f"Weather in {display_loc} (lat={lat}, lon={lon})\n"
            f"  Condition : {condition}\n"
            f"  Temp      : {temp} C (feels like {feels_like} C)\n"
            f"  Humidity  : {humidity}%\n"
            f"  Pressure  : {pressure} hPa\n"
            f"  Wind      : {wind_speed} km/h {wind_dir}\n"
            f"  Precip    : {precip} mm"
        )
    except requests.RequestException as e:
        weather_text = f"Weather: Could not fetch data - {e}"

    # --- Step 3: Fetch AQI from Open-Meteo Air Quality API ---
    aqi_text = ""
    try:
        aqi_resp = requests.get(
            "https://air-quality-api.open-meteo.com/v1/air-quality",
            params={
                "latitude": lat,
                "longitude": lon,
                "hourly": "pm10,pm2_5,nitrogen_dioxide,sulphur_dioxide,ozone,carbon_monoxide",
                "timezone": "auto",
            },
            timeout=10,
        )
        aqi_resp.raise_for_status()
        aqi_data = aqi_resp.json()
        hourly = aqi_data.get("hourly", {})

        # Find the index for the current hour
        times = hourly.get("time", [])
        now_hour_str = dt.datetime.now().strftime("%Y-%m-%dT%H:00")
        current_idx = None
        for i, t in enumerate(times):
            if t == now_hour_str:
                current_idx = i
                break

        if current_idx is None and times:
            # Fallback: use the last available hour
            current_idx = len(times) - 1

        if current_idx is not None:

            def _val(key):
                vals = hourly.get(key, [])
                if current_idx < len(vals):
                    v = vals[current_idx]
                    return f"{v}" if v is not None else "N/A"
                return "N/A"

            pm10 = _val("pm10")
            pm25 = _val("pm2_5")
            no2 = _val("nitrogen_dioxide")
            so2 = _val("sulphur_dioxide")
            o3 = _val("ozone")
            co = _val("carbon_monoxide")

            # Simple AQI estimation from PM2.5 (US EPA scale)
            def _aqi_from_pm25(v):
                try:
                    v = float(v)
                except (TypeError, ValueError):
                    return None
                if v <= 12.0:
                    return int(v * 50 / 12)
                elif v <= 35.4:
                    return int(50 + (v - 12.0) * 50 / 23.4)
                elif v <= 55.4:
                    return int(100 + (v - 35.4) * 50 / 20.0)
                elif v <= 150.4:
                    return int(150 + (v - 55.4) * 100 / 95.0)
                elif v <= 250.4:
                    return int(200 + (v - 150.4) * 100 / 100.0)
                elif v <= 350.4:
                    return int(300 + (v - 250.4) * 100 / 100.0)
                else:
                    return int(400 + (v - 350.4) * 500 / 150.0)

            def _aqi_label(aqi):
                if aqi is None:
                    return "N/A"
                if aqi <= 50:
                    return "Good"
                elif aqi <= 100:
                    return "Moderate"
                elif aqi <= 150:
                    return "Unhealthy for Sensitive Groups"
                elif aqi <= 200:
                    return "Unhealthy"
                elif aqi <= 300:
                    return "Very Unhealthy"
                else:
                    return "Hazardous"

            aqi_val = _aqi_from_pm25(pm25)
            aqi_label = _aqi_label(aqi_val)
            aqi_display = str(aqi_val) if aqi_val is not None else "N/A"

            aqi_text = (
                f"\n\nAir Quality Index (current hour)\n"
                f"  AQI        : {aqi_display} - {aqi_label}\n"
                f"  PM2.5      : {pm25} ug/m3\n"
                f"  PM10       : {pm10} ug/m3\n"
                f"  NO2        : {no2} ug/m3\n"
                f"  SO2        : {so2} ug/m3\n"
                f"  O3         : {o3} ug/m3\n"
                f"  CO         : {co} ug/m3"
            )
        else:
            aqi_text = "\n\nAQI: No hourly data available for current time."
    except requests.RequestException as e:
        aqi_text = f"\n\nAQI: Could not fetch data - {e}"

    return weather_text + aqi_text


def fetch_todays_emails(max_results: int = 10) -> str:
    """
    Fetch all Gmail messages received today, including sender, subject,
    snippet, and date.

    Args:
        max_results: Maximum number of emails to return (default 10).

    Returns:
        Formatted list of today's emails.
    """
    gmail_service = _build_service("gmail", "v1")
    user_id = "me"

    today = dt.date.today()
    after = f"{today.isoformat()}"

    messages = (
        gmail_service.users()
        .messages()
        .list(userId=user_id, q=f"after:{after}", maxResults=max_results)
        .execute()
        .get("messages", [])
    )

    if not messages:
        return f"📭 No emails received today ({after})."

    lines = [f"📬 Emails received today ({len(messages)} shown):"]
    for msg_meta in messages:
        msg = (
            gmail_service.users()
            .messages()
            .get(
                userId=user_id,
                id=msg_meta["id"],
                format="metadata",
                metadataHeaders=["From", "Subject", "Date"],
            )
            .execute()
        )
        headers = {h["name"]: h["value"] for h in msg["payload"]["headers"]}
        frm = headers.get("From", "Unknown")
        subject = headers.get("Subject", "(no subject)")
        date = headers.get("Date", "")
        snippet = msg.get("snippet", "")

        lines.append(f"\n  📧 From    : {frm}")
        lines.append(f"     Subject : {subject}")
        lines.append(f"     Date    : {date}")
        lines.append(f"     Preview : {snippet}")

    return "\n".join(lines)


def get_news(category: str = "general") -> str:
    """
    Fetch the latest headlines from around the world and India using the
    free Currents API (currentsapi.services).

    Args:
        category: One of the Currents API categories: "general", "society",
                  "science_technology", "politics_government",
                  "economy_business_finance", "arts_culture_entertainment",
                  "lifestyle_leisure", "human_interest", "sport",
                  "crime_law_justice", "education", "environment",
                  "labour", "health", "automotive", "real_estate".
                  Common aliases like "business", "technology", "sports",
                  "entertainment", "health", "science" also work.

    Returns:
        A formatted string with headlines and short descriptions.
    """
    api_key = os.getenv("CURRENTS_API_KEY")
    if not api_key:
        return "ERROR: CURRENTS_API_KEY environment variable is not set."

    base_url = "https://api.currentsapi.services/v1/latest-news"
    headers = {"Authorization": api_key}
    results: list[str] = []

    # Map common shorthand aliases to Currents category names
    category_aliases = {
        "general": "general",
        "business": "economy_business_finance",
        "technology": "science_technology",
        "sports": "sport",
        "entertainment": "arts_culture_entertainment",
        "health": "health",
        "science": "science_technology",
    }
    currents_category = category_aliases.get(category, category)

    for label, params in [
        (
            "World News",
            {"language": "en", "category": currents_category},
        ),
        (
            "India News",
            {"language": "en", "category": currents_category, "country": "IN"},
        ),
    ]:
        try:
            resp = requests.get(base_url, headers=headers, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            results.append(f"\n{label}: Failed to fetch - {e}")
            continue

        if data.get("status") != "ok":
            results.append(
                f"\n{label}: API returned error - {data.get('message', 'unknown')}"
            )
            continue

        articles = data.get("news", [])
        if not articles:
            results.append(f"\n{label}: No articles found.")
            continue

        lines = [f"\n{label}:"]
        for i, a in enumerate(articles, 1):
            title = a.get("title", "(no title)")
            desc = a.get("description", "")
            author = a.get("author", "")
            url = a.get("url", "")
            published = a.get("published", "")
            meta = f" ({author})" if author else ""
            line = f"  {i}. {title}{meta}"
            if published:
                line += f" [{published}]"
            if desc:
                line += f"\n     {desc}"
            line += f"\n     Link: {url}"
            lines.append(line)
        results.append("\n".join(lines))

    return "\n".join(results)


# ── Face DB helpers ────────────────────────────────────────────────────────────

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


# ── Tool implementations ───────────────────────────────────────────────────────


def get_weather(location: str) -> str:
    return f"Weather in {location} is sunny with a high of 25°C."


def get_image() -> str:
    image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "image.jpg")
    with open(image_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()


def capture_camera_frame() -> dict:
    print("[capture_camera_frame] called", flush=True)
    """
    Capture a frame from the camera stream (currently a local placeholder image).
    Runs DeepFace analysis (emotion, age, gender) and face identity lookup via ChromaDB.
    Returns a structured dict with the base64 image + all analysis results.
    """
    image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "image.jpg")

    with open(image_path, "rb") as f:
        raw = f.read()
    image_b64 = base64.b64encode(raw).decode()

    # ── DeepFace attribute analysis ────────────────────────────────────────────
    DeepFace = _get_deepface()
    attributes = {}
    try:
        analysis = DeepFace.analyze(
            img_path=image_path,
            actions=["age", "gender", "emotion", "race"],
            enforce_detection=True,
            detector_backend="opencv",
            silent=True,
        )
        a = analysis[0]  # first face
        attributes = {
            "age": a.get("age"),
            "gender": a.get("dominant_gender"),
            "emotion": a.get("dominant_emotion"),
            "emotion_scores": {k: round(v, 1) for k, v in a.get("emotion", {}).items()},
            "race": a.get("dominant_race"),
        }
        print(
            f"[DeepFace] Attributes: age={attributes['age']}, gender={attributes['gender']}, emotion={attributes['emotion']}"
        )
    except Exception as e:
        print(f"[DeepFace] analyze() failed: {e}")
        attributes = {"error": str(e)}

    # ── Face identity (embedding → ChromaDB) ──────────────────────────────────
    identity = {
        "recognized": False,
        "name": None,
        "familiar": False,
        "face_id": None,
        "is_new_face": False,
    }

    embedding = _embed_image(image_path)
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
                f"[FaceDB] Match found → name={identity['name']}, familiar={identity['familiar']}, dist={match['_distance']}"
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


# ── Tool schemas ───────────────────────────────────────────────────────────────


def save_to_memory(info: str, face_id: str = None, name: str = None) -> str:
    """
    Save any relevant information to the LLM's long-term memory vector database.
    Optionally link it to a face_id and name if it's about a specific person.
    """
    doc_id = str(uuid.uuid4())
    metadata = {}
    if face_id:
        metadata["face_id"] = face_id
    if name:
        metadata["name"] = name

    try:
        memory_collection.add(
            documents=[info], metadatas=[metadata] if metadata else None, ids=[doc_id]
        )
        print(f"[MemoryDB] Saved info for face_id={face_id}, name={name}: {info}")
        return f"Successfully saved memory: {info}"
    except Exception as e:
        return f"Failed to save memory: {e}"


def find_memory(query: str, face_id: str = None, name: str = None) -> list[str]:
    """
    Search the LLM's long-term memory vector database for relevant information.
    Optionally filter by face_id and/or name.
    """
    where_filter = None
    if face_id and name:
        where_filter = {"$and": [{"face_id": face_id}, {"name": name}]}
    elif face_id:
        where_filter = {"face_id": face_id}
    elif name:
        where_filter = {"name": name}

    try:
        results = memory_collection.query(
            query_texts=[query],
            n_results=5,
            where=where_filter,
        )
        if results and results.get("documents") and results["documents"][0]:
            found_memories = results["documents"][0]
            print(
                f"[MemoryDB] Found {len(found_memories)} memories for query '{query}'"
            )
            return found_memories
        else:
            print(f"[MemoryDB] No memories found for query '{query}'")
            return ["No relevant memories found."]
    except Exception as e:
        return [f"Failed to search memory: {e}"]


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a given location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The location to get the weather for.",
                    }
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather_aqi",
            "description": (
                "Get the current weather (temperature, humidity, wind speed, conditions, precipitation) "
                "and the Air Quality Index (AQI) including PM2.5, PM10, NO2, SO2, O3, CO readings. "
                "Uses ipinfo.io to detect location and Open-Meteo for weather + AQI data. "
                "No API keys required. The location parameter is optional and used only as a display "
                "label -- actual data is based on the server's IP geolocation. "
                "Use this whenever the user asks about weather, temperature, air quality, pollution, "
                "AQI, or whether it's safe to go outside."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": (
                            "Optional display label for the location, e.g. 'Mumbai', 'Home', "
                            "'Office'. If empty, the city is auto-detected from IP."
                        ),
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_news",
            "description": (
                "Fetch the latest top headlines from around the world and from India using the "
                "Currents API. Supports categories like general, business, technology, sports, "
                "entertainment, health, science, politics, environment, education, and more. "
                "Use this whenever the user asks 'what's in the news', 'latest headlines', 'India news', "
                "'world news', or any current-events question."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "enum": [
                            "general",
                            "business",
                            "technology",
                            "sports",
                            "entertainment",
                            "health",
                            "science",
                            "politics_government",
                            "economy_business_finance",
                            "science_technology",
                            "arts_culture_entertainment",
                            "lifestyle_leisure",
                            "sport",
                            "crime_law_justice",
                            "education",
                            "environment",
                            "society",
                            "human_interest",
                            "labour",
                            "automotive",
                            "real_estate",
                        ],
                        "description": (
                            "The news category to fetch. Defaults to 'general' if not specified. "
                            "Common aliases also work: 'business', 'technology', 'sports', "
                            "'entertainment', 'health', 'science'."
                        ),
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_todays_emails",
            "description": (
                "Fetch all Gmail messages received today, returning sender, subject, date, and a "
                "short preview snippet for each email. Use this when the user asks 'any new emails', "
                "'check my inbox', 'what did I get today', or 'show me today's emails'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "max_results": {
                        "type": "integer",
                        "description": (
                            "Maximum number of emails to return (default 10)."
                        ),
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_image",
            "description": "Retrieve and display a generic image.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "capture_camera_frame",
            "description": (
                "Capture a live frame from the camera stream. "
                "Automatically analyzes the frame for facial attributes (age, gender, emotion, race) "
                "using DeepFace, and checks the face against a local identity database to determine "
                "if the person is known, familiar (seen before but unnamed), or completely new. "
                "Use this when the user wants you to look at them, see who is there, "
                "analyze their expression or mood, or when visual context about the person is needed."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "register_face_name",
            "description": (
                "Store a person's name associated with their face in the local identity database. "
                "Call this after capturing a camera frame when the user tells you their name, "
                "or when you want to save a name for a recognized face_id. "
                "Requires the face_id returned from a prior capture_camera_frame call."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "face_id": {
                        "type": "string",
                        "description": "The face_id returned from capture_camera_frame's identity block.",
                    },
                    "name": {
                        "type": "string",
                        "description": "The name to associate with this face.",
                    },
                },
                "required": ["face_id", "name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_to_memory",
            "description": (
                "Save any important information, user preferences, or facts to long-term memory "
                "in the vector database. Optionally link it to a specific face_id "
                "from the camera frame tool and their name."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "info": {
                        "type": "string",
                        "description": "The piece of information to save (e.g., 'User likes coffee', 'Has a pet dog').",
                    },
                    "face_id": {
                        "type": "string",
                        "description": "The face_id of the person this info relates to, if any.",
                    },
                    "name": {
                        "type": "string",
                        "description": "The name of the person this info relates to, if any.",
                    },
                },
                "required": ["info"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_memory",
            "description": (
                "Search the long-term memory vector database for relevant information, user preferences, "
                "or facts. CRITICAL: Use this proactively whenever the user mentions themselves, "
                "their past, their preferences, or implies any personal context that you might reasonably have stored."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query (e.g., 'What did Alex ask me to remind him to buy?', 'What does Alex do?').",
                    },
                    "face_id": {
                        "type": "string",
                        "description": "Optional: Filter memories by the face_id of the person.",
                    },
                    "name": {
                        "type": "string",
                        "description": "Optional: Filter memories by the specific name of the person.",
                    },
                },
                "required": ["query"],
            },
        },
    },
]

TOOL_MAP = {
    "get_weather": get_weather,
    "get_image": get_image,
    "get_news": get_news,
    "get_weather_aqi": get_weather_aqi,
    "fetch_todays_emails": fetch_todays_emails,
    "capture_camera_frame": capture_camera_frame,
    "register_face_name": register_face_name,
    "save_to_memory": save_to_memory,
    "find_memory": find_memory,
}


# ── Streaming + tool execution ─────────────────────────────────────────────────


def stream_response(messages, tts: StreamingTTS | None = None) -> tuple[str, dict]:
    """Stream a response, returning (text, tool_calls_buffer).
    If tts is provided, tokens are fed into it sentence-by-sentence as they arrive.
    """
    stream = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
        stream=True,
        temperature=0.7,
        max_tokens=250,  # limit output tokens
    )

    full_response = ""
    tool_calls_buffer = {}

    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta

        if delta.content:
            print(delta.content, end="", flush=True)
            full_response += delta.content
            if tts is not None:
                tts.feed(delta.content)

        if delta.tool_calls:
            for tool_call in delta.tool_calls:
                idx = tool_call.index
                if idx not in tool_calls_buffer:
                    tool_calls_buffer[idx] = {
                        "id": tool_call.id,
                        "name": "",
                        "arguments": "",
                    }
                if tool_call.function.name:
                    tool_calls_buffer[idx]["name"] += tool_call.function.name
                if tool_call.function.arguments:
                    tool_calls_buffer[idx]["arguments"] += tool_call.function.arguments

    # Flush any remaining partial sentence
    if tts is not None and not tool_calls_buffer:
        tts.flush()
        tts.wait()

    return full_response, tool_calls_buffer


def execute_tools(tool_calls_buffer: dict) -> list[dict]:
    """Run all tool calls and return the result messages to append."""
    new_messages = []

    # Append the assistant's tool-use turn
    new_messages.append(
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": t["id"],
                    "type": "function",
                    "function": {"name": t["name"], "arguments": t["arguments"]},
                }
                for t in tool_calls_buffer.values()
            ],
        }
    )

    for tool in tool_calls_buffer.values():
        fn = TOOL_MAP.get(tool["name"])
        if fn is None:
            print(f"\n⚠️ Unknown tool: {tool['name']}")
            continue

        try:
            args = json.loads(tool["arguments"])
        except json.JSONDecodeError:
            print(f"\n⚠️ Failed to parse arguments for {tool['name']}")
            continue

        result = fn(**args)

        # ── camera frame: send image as vision + JSON context as text ─────────
        if tool["name"] == "capture_camera_frame":
            image_b64 = result.pop("image_b64")
            context_json = json.dumps(result, indent=2, default=str)
            print(
                f"\n[Tool: capture_camera_frame] → face_count={result['face_count']}, identity={result['identity']}"
            )

            # Pass the structured data back as the tool result string
            new_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool["id"],
                    "content": (
                        "Camera frame captured. DeepFace analysis and identity lookup results:\n"
                        f"{context_json}\n\n"
                        "Use the identity.face_id if you need to register a name via register_face_name."
                    ),
                }
            )

            # To allow the LLM to analyze the image (vision), we must pass the image in a user message.
            # Tool messages strictly require string content in OpenAI/Groq specs.
            new_messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Here is the visual frame captured from the camera for your analysis:",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                        },
                    ],
                }
            )

        # ── generic image tool ─────────────────────────────────────────────────
        elif tool["name"] == "get_image":
            print(f"\n[Tool: get_image] → image loaded ({len(result)} base64 chars)")
            new_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool["id"],
                    "content": "Image loaded successfully. See the attached user message for the image.",
                }
            )
            new_messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Here is the requested image:",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{result}"},
                        },
                    ],
                }
            )

        # ── all other tools: plain text ────────────────────────────────────────
        else:
            print(f"\n[Tool: {tool['name']}({args})] → {result}")
            new_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool["id"],
                    "content": str(result),
                }
            )

    return new_messages


# ── Chatbot ────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are ARIA, an earnest, warm, and highly concise AI with a camera stream. Use minimal tokens.
CRITICAL: DO NOT use internal reasoning, Chain of Thought, or <thought> tags. Output ONLY the final response.
1. Camera: ALWAYS call `capture_camera_frame` on the first turn. Greet known users warmly by name.
2. New/Familiar: Welcome them and politely ask/register their name.
3. Memory: CRITICAL - If a user says "you know me", mentions their line of work, or implies past context, ALWAYS call `find_memory` first with broad queries like "[Name]'s work" or "[Name]'s background".
4. Call `save_to_memory` to store new facts/preferences (include face_id/name).
5. Emotion: Respond naturally to their detected mood. Never expose raw JSON or face IDs.
"""


class Chatbot:
    def __init__(self):
        print("[Chatbot] init", flush=True)
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    def chat(self, user_input: str):
        self.messages.append({"role": "user", "content": user_input})

        # Memory Window: Keep only the system prompt + last 10 messages (5 turns)
        # That drastically saves tokens by not repeating endless history.
        if len(self.messages) > 11:
            self.messages = [self.messages[0]] + self.messages[-10:]

        # One StreamingTTS instance per chat turn - reused across the agentic loop
        # tts.feed() is only called when delta.content tokens arrive, so tool-call-only
        # turns (like capture_camera_frame) stay silent automatically.
        tts = StreamingTTS()

        print("Assistant:", end=" ", flush=True)
        text, tool_calls = stream_response(self.messages, tts=tts)

        # Agentic loop: keep executing tools until the model stops calling them
        while tool_calls:
            tool_messages = execute_tools(tool_calls)
            self.messages.extend(tool_messages)

            print("\nAssistant:", end=" ", flush=True)
            text, tool_calls = stream_response(self.messages, tts=tts)

        self.messages.append({"role": "assistant", "content": text})
        print("\n")

        tts.stop()


if __name__ == "__main__":
    print("Starting...", flush=True)

    # ── Eager initialization: load heavy models at startup ─────────────────────
    print("[Startup] Pre-loading DeepFace model...", flush=True)
    _get_deepface()
    print("[Startup] Pre-loading TTS model...", flush=True)
    from tts_engine_alt import get_tts

    get_tts()
    print("[Startup] All models loaded.\n", flush=True)

    bot = Chatbot()
    print("ARIA online. Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ["exit", "quit"]:
            break
        bot.chat(user_input)
