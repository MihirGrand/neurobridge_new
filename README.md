# Nuro - Voice-Driven AI Desktop Assistant

A real-time, conversational desktop AI assistant with facial recognition, voice interaction, and integrated Google services. Powered by LLM streaming, Silero VAD, faster-whisper STT, and Kokoro TTS with **barge-in** (interrupt) support.

---

## Overview

**Nuro** is a warm, concise AI assistant that:
- Listens for voice input via HTTP PCM stream (from ESP32 or similar device)
- Detects speech boundaries using **Silero VAD**
- Transcribes speech in real-time using **faster-whisper** (GPU-accelerated)
- Responds with conversational LLM output via **Kokoro TTS** with pipelined synthesis
- Supports **barge-in**: user can interrupt TTS by speaking, immediate response to speech detection
- Recognizes and greets familiar faces via **DeepFace** facial recognition
- Maintains long-term memory via **ChromaDB** vector database
- Calls integrated tools for weather, news, email, calendar, tasks, Google Docs, and more

---

## Architecture

```
HTTP PCM Stream (16-bit, 16kHz mono)
        ↓
   Silero VAD
   (speech detection)
        ↓
   faster-whisper STT
   (CUDA-accelerated)
        ↓
   OpenAI-compatible LLM
   (tool calling + streaming)
        ↓
   Kokoro TTS (pipelined)
   + speaker playback
```

### Barge-In Flow
```
User speaking while TTS active
        ↓
VAD detects speech_start
        ↓
TTS.interrupt() called
        ↓
Audio playback stopped immediately
LLM stream aborted
        ↓
New utterance transcribed & processed
```

---

## Technologies & Libraries

### Core Audio/Speech
- **Silero VAD** - Real-time speech detection (PyTorch-based)
- **faster-whisper** - GPU-accelerated speech-to-text (CTranslate2 backend, `small.en` model)
- **Kokoro (82M)** - Apache-licensed TTS, 10x faster than XTTS-v2
- **sounddevice** - Audio playback with minimal latency
- **requests** - HTTP PCM stream ingestion

### Computer Vision
- **DeepFace** - Face detection, embedding extraction, and identity matching
- **Chroma** - Persistent vector database for face embeddings and memory

### LLM & APIs
- **OpenAI Python SDK** - OpenAI-compatible API (any LLM provider: Groq, Together, etc.)
- **Google API Client** - Gmail, Calendar, Google Tasks, Google Docs
- **Currents API** - News headlines

### Infrastructure
- **PyTorch** - Deep learning framework (for VAD, DeepFace)
- **NumPy** - Numerical operations
- **threading** - Multi-threaded VAD, STT, synthesis, and playback pipelines
- **dotenv** - Environment configuration

---

## Core Components

### 1. **Speech-to-Text (STT)**
- Model: `faster-whisper` (small.en)
- Hardware: CUDA GPU (RTX 2050+, ~2GB VRAM)
- Accuracy: Good balance of speed vs accuracy for real-time use
- Computation type: `float16` (half-precision for efficiency)
- Input: Raw PCM float32 at 16 kHz

### 2. **Text-to-Speech (TTS)**
- Model: **Kokoro** (82M parameters, Apache-licensed)
- Voice: `af_heart` (warm, expressive American female, customizable)
- Features:
  - **Pipelined synthesis** - Sentence N+1 synthesizes while sentence N plays (no inter-sentence gaps)
  - **Sentence auto-splitting** - Automatically detects sentence boundaries (`.`, `!`, `?`)
  - **Interrupt support** - Barge-in immediately stops playback and synthesis
  - **Text normalization** - Converts symbols, units, decimals, and Roman numerals to spoken English
- Output: 24 kHz PCM audio

### 3. **Voice Activity Detection (VAD)**
- Model: **Silero VAD** (pre-trained PyTorch model)
- Frame size: 512 samples (32ms at 16 kHz)
- Threshold: 0.5 (probability of speech)
- Speech confirmation: 100ms of continuous speech required
- Silence timeout: 1200ms of sustained silence → triggers transcription
- Features:
  - Low false-positive rate (ignores clicks, environmental noise)
  - Survives natural pauses (commas, thought breaks up to 700ms)
  - Enables **pre-buffering** to capture speech onset before VAD confirmation

### 4. **Facial Recognition**
- Model: **DeepFace** (cascade of CNNs for detection + embedding extraction)
- Capabilities:
  - Real-time face detection from camera frame
  - Face embedding extraction (512-dimensional vectors)
  - Emotion & age estimation
  - Identity lookup via ChromaDB
- Integration: `capture_camera_frame()` tool returns face count, detected emotions, and identity matches
- Storage: Face embeddings stored in ChromaDB with optional name/ID metadata

### 5. **Memory System (Vector DB)**
- Backend: **ChromaDB** (persistent, local vector database)
- Embedding space: Cosine similarity (for semantic search)
- Use cases:
  - Store user preferences, facts, background context
  - Link memories to specific faces (via `face_id`)
  - Retrieve context via `find_memory(query)` with optional filtering by name/face_id
- Example: "Mihir works in AI" stored with face_id → queried later when Mihir returns

### 6. **LLM & Tool Calling**
- API: OpenAI-compatible (supports Groq, Together, OpenRouter, etc.)
- Streaming: Token-by-token with immediate TTS feeding
- Tool integration: **Function calling** - LLM decides which tools to invoke
- Message management:
  - System prompt with behavioral guidelines (warmth, conciseness, barge-in priority)
  - Memory window trimming (keeps system + last 10 messages to save tokens)
  - Sanitization of orphaned/broken tool calls from interrupted streams
- Interruption handling: `_llm_interrupt` event set during barge-in → aborts stream gracefully

---

## Tools

### 📸 **Vision & Identity**
- **`capture_camera_frame()`** - Detect faces, recognize identities, estimate emotion/age
- **`get_image(path)`** - Retrieve and display images
- **`register_face_name(face_id, name)`** - Link a detected face to a name

### 💾 **Memory**
- **`find_memory(query, face_id?, name?)`** - Search long-term memory via ChromaDB (semantic + filtering)
- **`save_to_memory(info, face_id?, name?)`** - Store facts/preferences linked to optional face/name

### 🌐 **Information**
- **`get_weather(location)`** - Current weather and forecast
- **`get_news(category?)`** - Latest headlines (general, business, tech, sports, etc.)

### 📧 **Email (Gmail)**
- **`fetch_todays_emails()`** - Inbox emails from today
- **`search_emails(query)`** - Full-text search
- **`draft_email(to, subject, body)`** - Compose unsent message
- **`send_email(to, subject, body)`** - Send immediately
- **`reply_email(message_id, body)`** - Reply to existing thread
- **`draft_reply_email(message_id, body)`** - Compose unsent reply

### 📅 **Calendar (Google Calendar)**
- **`get_todays_events()`** - Today's schedule
- **`find_event(query)`** - Search for events
- **`add_calendar_event(title, time, description?)`** - Create new event
- **`update_calendar_event(event_id, updates)`** - Modify existing event

### ✅ **Tasks (Google Tasks)**
- **`get_todays_tasks()`** - Tasks due today
- **`get_pending_tasks()`** - All incomplete tasks
- **`add_google_task(title, notes?)`** - Create new task
- **`find_task(query)`** - Search tasks
- **`update_google_task(task_id, updates)`** - Modify task

### 📝 **Google Docs**
- **`search_google_docs(query)`** - Find documents by title/content
- **`read_google_doc(doc_id)`** - Fetch document content
- **`create_google_doc(title, content?)`** - Create new document
- **`update_google_doc(doc_id, content)`** - Modify document

---

## Key Features

### ⚡ **Barge-In (Interrupt)**
- User can speak anytime, even while TTS is playing
- VAD immediately detects speech onset
- TTS and LLM stream abort instantly
- New utterance processed without delay
- Prevents awkward "sorry, I didn't catch that" loops

### 🔄 **Streaming Pipeline**
- LLM tokens arrive one-by-one
- Each token fed to TTS sentence-by-sentence
- Kokoro synthesizes sentence N+1 while sentence N plays (zero inter-sentence gaps)
- **Agentic loop**: After response, LLM can call tools → execute → continue conversation

### 📍 **Multi-Modal Interaction**
- **Voice input** (primary) - VAD + STT
- **Keyboard fallback** - For testing, accessibility
- **Camera input** - Face detection + recognition on first turn
- **Text output** - Streamed through TTS speaker (not displayed text)

### 🧠 **Context Management**
- System prompt emphasizes warmth, conciseness, no internal reasoning
- Automatic greeting for recognized faces
- Memory queries to build context on repeat visitors
- Token-efficient message windowing (last 10 messages only)

### 🛡️ **Robustness**
- Reconnection logic for HTTP PCM stream (auto-retry on disconnect)
- Whisper VAD filtering for extra speech detection confidence
- Pre-buffering to capture speech onset missed by initial VAD frames
- Orphaned tool-call sanitization for interrupted streams

---

## Configuration

### PCM Stream
```python
PCM_URL = "http://192.168.0.250/audio"
PCM_SAMPLE_RATE = 16000  # Hz
PCM_SAMPLE_WIDTH = 2     # bytes (16-bit)
PCM_CHANNELS = 1         # mono
```

### VAD
```python
VAD_THRESHOLD = 0.5              # Speech probability threshold
SPEECH_PAD_MS = 100              # Min speech duration (ignore pops)
SILENCE_DURATION_MS = 1000       # Sustained silence → transcribe
```

### STT (faster-whisper)
```python
WHISPER_MODEL_SIZE = "small.en"
WHISPER_DEVICE = "cuda"
WHISPER_COMPUTE_TYPE = "float16"
```

### TTS (Kokoro)
```python
VOICE = "af_heart"  # or af_bella, am_adam, etc.
SPEED = 1.1         # 10% faster than default
SAMPLE_RATE = 24000 # Output Hz
```

---

## Deployment

### Requirements
- Python 3.11+
- CUDA 11.8+ (for GPU acceleration)
- RTX 2050+ (or similar 2GB+ VRAM GPU)
- ESP32 or similar device streaming PCM to HTTP endpoint
- Google OAuth credentials (Gmail, Calendar, Tasks, Docs)
- LLM API key (OpenAI, Groq, Together, etc.)

### Installation
```bash
pip install -r requirements.txt
```

### Running
```bash
python main_4.py
```

Then speak into the microphone. Nuro listens for speech onset, transcribes, and responds.

---

## Files

| File | Purpose |
|------|---------|
| `main_4.py` | Main event loop, VAD pipeline, agentic LLM loop, barge-in coordination |
| `tts_engine_alt_3.py` | Kokoro TTS pipeline with pipelined synthesis + interrupt support |
| `tools/` | Tool implementations (weather, email, calendar, memory, face recognition, etc.) |
| `tools/schemas.py` | TOOLS list (OpenAI function definitions) + TOOL_MAP (name → function) |
| `face_db/` | ChromaDB persistent storage (embeddings + memory) |

---

## RAG (Retrieval-Augmented Generation)

Nuro implements **light RAG** via:
1. **Memory recall** - `find_memory(query)` retrieves relevant context before LLM processes user input
2. **Vector embeddings** - ChromaDB stores facts/preferences as semantic vectors → similarity search
3. **Face-linked memory** - Memories tied to specific faces/names for personalization
4. **Dynamic context injection** - System prompt updated with recalled memories before streaming

This allows Nuro to "remember" past conversations, user preferences, and relevant context without explicit prompt engineering.

---

## Performance Notes

- **VAD latency**: ~32ms per frame (512 samples @ 16kHz)
- **STT latency**: ~1–3s for typical 5–10s utterance (GPU-accelerated)
- **TTS latency**: ~300–600ms synthesis per sentence + realtime playback
- **Total end-to-end**: ~2–5 seconds user speech → AI response audio starts

Pipelining (synthesis during playback) reduces user-perceived latency significantly.
