"""Voice-driven Nuro with STT, VAD, and barge-in.

Reads raw PCM audio from http://192.168.0.250/audio, runs Silero VAD
to detect speech boundaries, transcribes with faster-whisper (small model,
CUDA), and sends the transcript to the LLM.  If the TTS is currently
speaking and the user starts talking, the TTS is interrupted immediately
(barge-in).

Architecture:
  ┌─────────────────────┐
  │  HTTP PCM stream    │  192.168.0.250/audio
  │  (raw 16-bit mono)  │
  └─────────────────────┘
  ┌─────────────────────┐
  │ VAD (Silero)        │  Detects voice activity in real-time
  │  → speech_start     │  Triggers barge-in (interrupts TTS)
  │  → speech_end       │  After sustained silence → transcribe
  └─────────────────────┘
  ┌─────────────────────┐
  │ STT (faster-whisper)│  GPU-accelerated, CTranslate2
  │  small model        │  Good balance: accuracy vs speed
  └─────────────────────┘
  ┌─────────────────────┐
  │ LLM + TTS pipeline  │  Same agentic loop as main_3.py
  └─────────────────────┘

PCM stream config (from ESP32):
  - 16-bit signed little-endian (int16)
  - 16000 Hz sample rate (standard for speech)
  - Mono

If your stream uses different settings, adjust PCM_* constants below.
"""

import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import json
import logging

logging.getLogger("tensorflow").setLevel(logging.ERROR)

import collections
import datetime
import io
import struct
import threading
import time
from typing import Optional

import numpy as np
import requests
import torch
from dotenv import load_dotenv
from openai import OpenAI

from tools import TOOL_MAP, TOOLS
from tools.face import _get_deepface
from tts_engine_alt_3 import StreamingTTS

load_dotenv()

API_KEY = os.getenv("LLM_API_KEY")
BASE_URL = os.getenv("LLM_BASE_URL")
MODEL = os.getenv("LLM_MODEL")

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# ── PCM Stream Configuration ──────────────────────────────────────────────────
# Adjust these if your ESP32/mic sends a different format.
PCM_URL = "http://192.168.0.250/audio"
PCM_SAMPLE_RATE = 16000  # Hz (standard for speech / Silero VAD)
PCM_SAMPLE_WIDTH = 2  # bytes per sample (16-bit = 2)
PCM_CHANNELS = 1  # mono

# ── VAD Configuration ─────────────────────────────────────────────────────────
# These tune how "speech end" is detected.  The goal is to NOT trigger on
# natural pauses (which can be 300–700ms) but DO trigger when the user
# has genuinely finished speaking.
VAD_THRESHOLD = 0.5  # Silero VAD probability threshold for "speech"
SPEECH_PAD_MS = 100  # Min speech duration to count (ignores clicks/pops)
SILENCE_DURATION_MS = 1000  # Sustained silence after speech → "done speaking"
# 1200ms is long enough to survive natural pauses (commas, thought breaks)
# but short enough to feel responsive.  Adjust to taste.

# ── STT Configuration ─────────────────────────────────────────────────────────
WHISPER_MODEL_SIZE = "small.en"  # good accuracy/speed on RTX 2050 (~2GB VRAM)
WHISPER_DEVICE = "cuda"
WHISPER_COMPUTE_TYPE = "float16"  # RTX 2050 supports fp16

# ── VAD frame size ─────────────────────────────────────────────────────────────
# Silero VAD requires sr/num_samples <= 31.25 → min 512 samples at 16kHz (32ms)
VAD_FRAME_SAMPLES = 512  # 32ms at 16kHz
VAD_FRAME_BYTES = VAD_FRAME_SAMPLES * PCM_SAMPLE_WIDTH  # 1024 bytes


# ── Silero VAD loader ─────────────────────────────────────────────────────────
_vad_model = None


def get_vad_model():
    """Load Silero VAD model (lazy, cached)."""
    global _vad_model
    if _vad_model is None:
        print("[VAD] Loading Silero VAD model...", flush=True)
        model, _utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            onnx=False,
        )
        _vad_model = model
        print("[VAD] Silero VAD ready.", flush=True)
    return _vad_model


# ── faster-whisper loader ──────────────────────────────────────────────────────
_whisper_model = None


def get_whisper_model():
    """Load faster-whisper model (lazy, cached)."""
    global _whisper_model
    if _whisper_model is None:
        print(
            f"[STT] Loading faster-whisper ({WHISPER_MODEL_SIZE}) on {WHISPER_DEVICE}...",
            flush=True,
        )
        from faster_whisper import WhisperModel

        _whisper_model = WhisperModel(
            WHISPER_MODEL_SIZE,
            device=WHISPER_DEVICE,
            compute_type=WHISPER_COMPUTE_TYPE,
        )
        print("[STT] faster-whisper ready.", flush=True)
    return _whisper_model


def transcribe_audio(audio_np: np.ndarray) -> str:
    """Transcribe a numpy float32 array (16kHz mono) → text string."""
    model = get_whisper_model()
    segments, info = model.transcribe(
        audio_np,
        beam_size=5,
        language="en",
        vad_filter=True,  # extra VAD pass inside whisper for robustness
        vad_parameters=dict(
            min_silence_duration_ms=500,
        ),
    )
    text = " ".join(seg.text.strip() for seg in segments).strip()
    return text


# ── Audio Listener (PCM stream → VAD → STT) ───────────────────────────────────


class AudioListener:
    """Connects to a raw PCM HTTP stream, runs Silero VAD, and emits
    transcribed utterances via a callback.

    Supports barge-in: if `tts` is provided and is currently speaking,
    the listener will interrupt it when the user starts talking.
    """

    def __init__(
        self,
        on_utterance,  # callback(text: str) called when user finishes speaking
        tts: StreamingTTS | None = None,
        pcm_url: str = PCM_URL,
    ):
        self.on_utterance = on_utterance
        self.tts = tts
        self.pcm_url = pcm_url

        self._running = False
        self._thread: threading.Thread | None = None

        # VAD state
        self._vad_model = None
        self._in_speech = False
        self._speech_frames: list[bytes] = []  # accumulate raw PCM during speech
        self._silence_counter = 0  # how many consecutive silence frames
        self._speech_counter = 0  # how many consecutive speech frames

        # Pre-buffer: rolling ring buffer that keeps the last ~300ms of audio
        # so we capture the speech onset that happens BEFORE VAD confirms speech.
        _pre_buffer_frames = int(
            (300 / 1000.0) * PCM_SAMPLE_RATE / VAD_FRAME_SAMPLES
        )  # ~9 frames at 32ms each
        self._pre_buffer: collections.deque[bytes] = collections.deque(
            maxlen=_pre_buffer_frames
        )

        # How many silence frames = SILENCE_DURATION_MS
        self._silence_frames_needed = int(
            (SILENCE_DURATION_MS / 1000.0) * PCM_SAMPLE_RATE / VAD_FRAME_SAMPLES
        )
        # Min speech frames to count as real speech (ignore pops)
        self._min_speech_frames = int(
            (SPEECH_PAD_MS / 1000.0) * PCM_SAMPLE_RATE / VAD_FRAME_SAMPLES
        )

        # Lock to prevent concurrent transcription + chat
        self._processing = threading.Lock()

    def start(self):
        """Start listening in a background thread."""
        if self._running:
            return
        self._running = True
        self._vad_model = get_vad_model()
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()
        print("[AudioListener] Listening for speech...", flush=True)

    def stop(self):
        """Stop listening."""
        self._running = False

    def _listen_loop(self):
        """Main loop: connect to HTTP PCM stream and process frames."""
        while self._running:
            try:
                print(f"[AudioListener] Connecting to {self.pcm_url}...", flush=True)
                response = requests.get(self.pcm_url, stream=True, timeout=10)
                response.raise_for_status()
                print("[AudioListener] Connected. Streaming audio.", flush=True)

                # Read in fixed-size chunks for VAD
                leftover = b""
                for chunk in response.iter_content(chunk_size=VAD_FRAME_BYTES * 4):
                    if not self._running:
                        break

                    data = leftover + chunk
                    offset = 0

                    while offset + VAD_FRAME_BYTES <= len(data):
                        frame = data[offset : offset + VAD_FRAME_BYTES]
                        offset += VAD_FRAME_BYTES
                        self._process_vad_frame(frame)

                    leftover = data[offset:]

            except requests.exceptions.ConnectionError:
                if self._running:
                    print(
                        "[AudioListener] Connection lost. Reconnecting in 3s...",
                        flush=True,
                    )
                    time.sleep(3)
            except requests.exceptions.Timeout:
                if self._running:
                    print(
                        "[AudioListener] Connection timed out. Retrying in 3s...",
                        flush=True,
                    )
                    time.sleep(3)
            except Exception as e:
                if self._running:
                    print(
                        f"[AudioListener] Stream error: {e}. Retrying in 3s...",
                        flush=True,
                    )
                    time.sleep(3)

    def _process_vad_frame(self, frame_bytes: bytes):
        """Process a single VAD frame (32ms of 16kHz int16 PCM)."""
        # Convert raw PCM bytes → float32 tensor for Silero
        int16_arr = np.frombuffer(frame_bytes, dtype=np.int16)
        float32_arr = int16_arr.astype(np.float32) / 32768.0
        tensor = torch.from_numpy(float32_arr)

        # Run VAD
        speech_prob = self._vad_model(tensor, PCM_SAMPLE_RATE).item()
        is_speech = speech_prob > VAD_THRESHOLD

        if is_speech:
            self._silence_counter = 0
            self._speech_counter += 1

            if not self._in_speech and self._speech_counter >= self._min_speech_frames:
                # ── Speech started ─────────────────────────────────────────
                self._in_speech = True
                print("\n[VAD] 🎙️ Speech detected", flush=True)

                # Prepend the pre-buffer so we capture the speech onset
                # that happened BEFORE VAD had enough frames to confirm.
                self._speech_frames = list(self._pre_buffer)
                self._pre_buffer.clear()

                # Barge-in: interrupt TTS if it's speaking
                if self.tts is not None and self.tts.is_speaking:
                    self.tts.interrupt()

            if self._in_speech:
                self._speech_frames.append(frame_bytes)
            else:
                # Not yet confirmed as speech - keep in pre-buffer
                self._pre_buffer.append(frame_bytes)

        else:
            self._speech_counter = 0

            if self._in_speech:
                self._speech_frames.append(frame_bytes)
                self._silence_counter += 1

                if self._silence_counter >= self._silence_frames_needed:
                    # ── Speech ended (sustained silence) ───────────────────
                    print("[VAD] 🔇 Speech ended - transcribing...", flush=True)
                    self._in_speech = False
                    self._silence_counter = 0

                    # Extract audio and transcribe
                    audio_bytes = b"".join(self._speech_frames)
                    self._speech_frames = []

                    # Transcribe in a separate thread so we don't block
                    # the VAD pipeline
                    threading.Thread(
                        target=self._handle_utterance,
                        args=(audio_bytes,),
                        daemon=True,
                    ).start()
            else:
                # Silence outside of speech - feed pre-buffer
                self._pre_buffer.append(frame_bytes)

    def _handle_utterance(self, audio_bytes: bytes):
        """Transcribe audio and call the utterance callback."""
        if not self._processing.acquire(blocking=False):
            print(
                "[AudioListener] Skipping - still processing previous utterance.",
                flush=True,
            )
            return

        try:
            # Convert raw PCM to float32
            int16_arr = np.frombuffer(audio_bytes, dtype=np.int16)
            float32_arr = int16_arr.astype(np.float32) / 32768.0

            # Skip very short clips (< 0.5s) - probably noise
            if len(float32_arr) < PCM_SAMPLE_RATE * 0.5:
                print("[STT] Audio too short, skipping.", flush=True)
                return

            text = transcribe_audio(float32_arr)
            if text and text.strip():
                print(f'\n[STT] "{text}"', flush=True)
                try:
                    self.on_utterance(text.strip())
                except Exception as e:
                    print(f"[Chat] Error processing utterance: {e}", flush=True)
            else:
                print("[STT] (empty transcription, ignoring)", flush=True)
        except Exception as e:
            print(f"[STT] Transcription error: {e}", flush=True)
        finally:
            self._processing.release()


# ── Streaming + tool execution ─────────────────────────────────────────────────


# Shared interrupt event - set by barge-in to abort current LLM stream
_llm_interrupt = threading.Event()


def stream_response(
    messages, tts: StreamingTTS | None = None
) -> tuple[str, dict, bool]:
    """Stream a response, returning (text, tool_calls_buffer, was_interrupted).
    If tts is provided, tokens are fed into it sentence-by-sentence as they arrive.
    Respects _llm_interrupt to abort mid-stream.
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
    was_interrupted = False

    for chunk in stream:
        # Check for barge-in interrupt
        if _llm_interrupt.is_set():
            was_interrupted = True
            print("\n[LLM] ⚡ Stream interrupted (barge-in)", flush=True)
            try:
                stream.close()
            except Exception:
                pass
            break

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
                idx = tool_call.index if tool_call.index is not None else 0
                # Some providers (e.g. Gemini) reuse index=0 for all tool
                # calls.  Detect a *new* tool by checking for a fresh id
                # at an already-occupied index.
                if (
                    tool_call.id is not None
                    and idx in tool_calls_buffer
                    and tool_calls_buffer[idx]["id"] is not None
                    and tool_calls_buffer[idx]["id"] != tool_call.id
                ):
                    idx = (
                        max(
                            (k for k in tool_calls_buffer if isinstance(k, int)),
                            default=-1,
                        )
                        + 1
                    )
                if idx not in tool_calls_buffer:
                    tool_calls_buffer[idx] = {
                        "id": tool_call.id,
                        "name": "",
                        "arguments": "",
                    }
                if tool_call.id and not tool_calls_buffer[idx]["id"]:
                    tool_calls_buffer[idx]["id"] = tool_call.id
                if tool_call.function.name:
                    tool_calls_buffer[idx]["name"] += tool_call.function.name
                if tool_call.function.arguments:
                    tool_calls_buffer[idx]["arguments"] += tool_call.function.arguments

    # Flush any remaining partial sentence
    if tts is not None and not tool_calls_buffer and not was_interrupted:
        tts.flush()
        tts.wait()

    return full_response, tool_calls_buffer, was_interrupted


def execute_tools(tool_calls_buffer: dict) -> list[dict]:
    """Run all tool calls and return the result messages to append."""
    new_messages = []

    # Filter out incomplete tool calls (e.g. from interrupted streams)
    # Gemini rejects tool calls with empty names.
    valid_tools = {
        k: v for k, v in tool_calls_buffer.items() if v.get("name") and v.get("id")
    }
    if not valid_tools:
        return new_messages

    # Append the assistant's tool-use turn
    new_messages.append(
        {
            "role": "assistant",
            "content": None,  # explicit null for Gemini / other providers
            "tool_calls": [
                {
                    "id": t["id"],
                    "type": "function",
                    "function": {"name": t["name"], "arguments": t["arguments"]},
                }
                for t in valid_tools.values()
            ],
        }
    )

    for tool in valid_tools.values():
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
                    "name": tool["name"],
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
                    "name": tool["name"],
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
                    "name": tool["name"],
                    "content": str(result),
                }
            )

    return new_messages


# ── Chatbot ────────────────────────────────────────────────────────────────────

# SYSTEM_PROMPT = """You are Nuro, an earnest, warm, and highly concise AI with a camera stream. Use minimal tokens.
# CRITICAL: DO NOT use internal reasoning, Chain of Thought, or <thought> tags. Output ONLY the final response.
# 1. When you need to call tools, ALWAYS say a brief natural acknowledgment FIRST before the tool calls or capture_camera_frame so the user hears something immediately. Examples: "Sure, let me check.", "One moment, pulling that up.", "Let me grab your schedule." Keep it under one sentence. This is especially important when calling multiple tools like for daily briefings.
# 2. Camera: ALWAYS call `capture_camera_frame` on the first turn. Greet known users warmly by name.
# 3. New/Familiar: Welcome them and politely ask/register their name.
# 4. Memory: CRITICAL - If a user says "you know me", mentions their line of work, or implies past context, ALWAYS call `find_memory` first with broad queries like "[Name]'s work" or "[Name]'s background".
# 5. Call `save_to_memory` to store new facts/preferences (include face_id/name).
# 6. Emotion: Respond naturally to their detected mood. Never expose raw JSON or face IDs.
# 7. Daily Briefing: If user asks for their agenda, day plan, or to start their day, call: get_weather, get_news, fetch_todays_emails, get_todays_events, get_todays_tasks, get_pending_tasks. Return a very concise summary.
# 8. Formatting: Your output is spoken aloud via TTS, NOT displayed visually. NEVER use asterisks, hashes, bullet markers, dashes as list markers, markdown, or any special formatting characters. Write in plain, natural spoken English only. Use commas and periods for structure. Write numbers and units as spoken words when short (e.g. "thirty five degrees Celsius", "Artemis the second").
# """

SYSTEM_PROMPT = """You are Nuro, an earnest, warm, and concise AI with a camera stream and speaker. Use minimal tokens. Do not think out loud. Do not use <think> or <thought> tags.
1. CRITICAL - User feedback: You are a desk assistant with a mic and speaker. The user CANNOT see text - they can only HEAR you. Before calling ANY tools, you MUST speak a short acknowledgment so the user is not left waiting in silence. Examples: "Sure, one moment.", "Let me check.", "Pulling that up." After the tools return, go straight to the answer - do NOT repeat the acknowledgment.
2. Camera & Greetings: On the very first turn, call `capture_camera_frame`. If the user is recognized, greet them warmly by name. For new users, welcome them and ask their name.
3. Memory: If a user implies past context or says "you know me", call `find_memory` first with broad queries like "[Name]'s background". Call `save_to_memory` to store new facts, always including name or face_id.
4. Emotion: Respond naturally to detected mood. Never expose raw JSON or face IDs.
5. Daily Briefing: If asked for agenda, day plan, or to start the day, acknowledge the user FIRST, then call: get_weather, get_news, fetch_todays_emails, get_todays_events, get_todays_tasks, get_pending_tasks. Return a concise spoken summary.
6. Formatting: Output is spoken via TTS. No asterisks, hashes, dashes, bullet markers, or markdown. Plain spoken English only. Spell out short numbers and units.
"""


class Chatbot:
    def __init__(self):
        print("[Chatbot] init", flush=True)
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        self._chat_lock = threading.Lock()
        self.tts: StreamingTTS | None = None  # set by main loop

    def chat(self, user_input: str):
        """Process a user message (from keyboard or STT)."""
        with self._chat_lock:
            self._chat_inner(user_input)

    def _sanitize_messages(self):
        """Remove broken / orphaned tool messages from history.

        Two failure modes:
          1. Interrupted LLM streams leave partial tool calls (empty names).
          2. Memory-window trimming slices away the assistant tool_calls
             message but leaves the orphaned tool-result messages behind.
        Gemini rejects both with 'function_response.name: Name cannot be empty'.
        """
        # Pass 1: collect all valid tool_call IDs defined by assistant messages
        defined_ids: set[str] = set()
        for msg in self.messages:
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    fn_name = tc.get("function", {}).get("name", "")
                    tc_id = tc.get("id", "")
                    if fn_name and tc_id:
                        defined_ids.add(tc_id)

        # Pass 2: filter out broken messages
        clean = []
        for msg in self.messages:
            role = msg.get("role")

            # Drop tool result messages with empty name OR orphaned
            # (no matching assistant tool_call in history)
            if role == "tool":
                if not msg.get("name"):
                    continue
                if msg.get("tool_call_id") not in defined_ids:
                    continue

            # Drop assistant tool-call messages that have any call with empty name
            if role == "assistant" and msg.get("tool_calls"):
                bad = any(
                    not (tc.get("function", {}).get("name")) for tc in msg["tool_calls"]
                )
                if bad:
                    continue

            clean.append(msg)
        self.messages = clean

    def _chat_inner(self, user_input: str):
        # Clear any lingering interrupt flag
        _llm_interrupt.clear()

        current_time = datetime.datetime.now().strftime("%I:%M %p, %A, %B %d, %Y")
        structured_input = f"[Current Time: {current_time}]\n{user_input}"
        self.messages.append({"role": "user", "content": structured_input})

        # Memory Window: Keep only the system prompt + last 10 messages (5 turns)
        # That drastically saves tokens by not repeating endless history.
        if len(self.messages) > 11:
            self.messages = [self.messages[0]] + self.messages[-10:]

        # Scrub any broken tool messages left by interrupted streams
        self._sanitize_messages()

        # One StreamingTTS instance per chat turn - reused across the agentic loop
        # tts.feed() is only called when delta.content tokens arrive, so tool-call-only
        # turns (like capture_camera_frame) stay silent automatically.
        tts = StreamingTTS()
        self.tts = tts  # expose for barge-in

        print("Assistant:", end=" ", flush=True)
        text, tool_calls, interrupted = stream_response(self.messages, tts=tts)

        if interrupted:
            # Don't continue the agentic loop - user interrupted
            if text:
                self.messages.append(
                    {"role": "assistant", "content": text + " [interrupted]"}
                )
            tts.stop()
            self.tts = None
            print("\n")
            return

        # Agentic loop: keep executing tools until the model stops calling them
        while tool_calls:
            if _llm_interrupt.is_set():
                tts.stop()
                self.tts = None
                print("\n")
                return

            tool_messages = execute_tools(tool_calls)
            self.messages.extend(tool_messages)

            print("\nAssistant:", end=" ", flush=True)
            text, tool_calls, interrupted = stream_response(self.messages, tts=tts)

            if interrupted:
                if text:
                    self.messages.append(
                        {"role": "assistant", "content": text + " [interrupted]"}
                    )
                tts.stop()
                self.tts = None
                print("\n")
                return

        self.messages.append({"role": "assistant", "content": text})
        print("\n")

        tts.stop()
        self.tts = None


def on_barge_in(bot: Chatbot):
    """Called when VAD detects speech while TTS is active.
    Interrupts both the TTS and the LLM stream."""
    _llm_interrupt.set()
    if bot.tts is not None:
        bot.tts.interrupt()


if __name__ == "__main__":
    print("Starting...", flush=True)

    # ── Eager initialization: load heavy models at startup ─────────────────────
    print("[Startup] Pre-loading DeepFace model...", flush=True)
    _get_deepface()

    print("[Startup] Pre-loading TTS engine (Kokoro)...", flush=True)
    from tts_engine_alt_3 import warmup as tts_warmup

    tts_warmup()

    print("[Startup] Pre-loading Silero VAD...", flush=True)
    get_vad_model()

    print("[Startup] Pre-loading faster-whisper...", flush=True)
    get_whisper_model()

    print("[Startup] All models loaded.\n", flush=True)

    bot = Chatbot()

    # ── Set up audio listener with barge-in ────────────────────────────────────
    def handle_utterance(text: str):
        """Called by AudioListener when user finishes speaking."""
        # Barge-in: interrupt any ongoing TTS
        on_barge_in(bot)
        # Small delay to let interrupt propagate
        time.sleep(0.1)
        # Send to chatbot
        bot.chat(text)

    listener = AudioListener(
        on_utterance=handle_utterance,
        tts=None,  # We dynamically check bot.tts in barge-in
    )

    # Override the barge-in check in the listener to use bot.tts
    original_process_vad = listener._process_vad_frame

    def _process_vad_frame_with_bargein(frame_bytes: bytes):
        """Wrapper that sets the listener's tts reference dynamically."""
        listener.tts = bot.tts
        original_process_vad(frame_bytes)

    listener._process_vad_frame = _process_vad_frame_with_bargein

    listener.start()

    print("NURO online. Voice input active. You can also type below.\n")
    print("Type 'exit' to quit.\n")

    # ── Main loop: keyboard fallback + voice runs in background ────────────────
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user_input:
            continue
        if user_input.lower() in ["exit", "quit"]:
            break
        # Interrupt any TTS in progress when user types
        on_barge_in(bot)
        time.sleep(0.05)
        bot.chat(user_input)

    listener.stop()
    print("Shutting down...")
