"""Streaming TTS engine using Kokoro (82M param, Apache-licensed).

Advantages over XTTS-v2:
  • ~10x faster synthesis - 82M params vs XTTS-v2's 467M
  • Superior pronunciation via espeak-ng G2P fallback
  • Natural, expressive voices (af_heart, af_bella, am_adam, etc.)
  • Built-in sentence splitting - no manual regex needed
  • Pipelined architecture: synthesis of sentence N+1 overlaps
    playback of sentence N, eliminating inter-sentence pauses

Requirements:
  pip install kokoro sounddevice numpy
  Also install espeak-ng (used for English OOD/fallback):
    Windows: download .msi from https://github.com/espeak-ng/espeak-ng/releases
    Linux:   sudo apt install espeak-ng
"""

import queue
import re
import threading

import numpy as np
import sounddevice as sd

# ── Configuration ──────────────────────────────────────────────────────────────
# Voices: https://huggingface.co/hexgrad/Kokoro-82M
#   af_heart    – warm, expressive American female (default)
#   af_bella    – clear, friendly American female
#   af_sarah    – professional American female
#   am_adam     – natural American male
#   am_michael  – clear American male
#   bf_emma     – British female
#   bm_george   – British male
VOICE = "af_heart"
LANG_CODE = "a"  # 'a' = American English, 'b' = British English
SPEED = 1.1  # slightly faster than default (1.0)
SAMPLE_RATE = 24000  # Kokoro outputs 24 kHz audio

SENTENCE_END = re.compile(r"(?<=[.!?])\s+")

# ── Text normalisation for TTS ─────────────────────────────────────────────────
# Kokoro can mispronounce symbols and abbreviations (e.g. reads "II" as two
# letter-I's, "°C" as "degrees c").  This function rewrites them as natural
# spoken English before synthesis.

# Roman numeral map (up to 39 to cover realistic use-cases)
_ROMAN_MAP = [
    ("XXXIX", "the thirty ninth"),
    ("XXXVIII", "the thirty eighth"),
    ("XXXVII", "the thirty seventh"),
    ("XXXVI", "the thirty sixth"),
    ("XXXV", "the thirty fifth"),
    ("XXXIV", "the thirty fourth"),
    ("XXXIII", "the thirty third"),
    ("XXXII", "the thirty second"),
    ("XXXI", "the thirty first"),
    ("XXX", "the thirtieth"),
    ("XXIX", "the twenty ninth"),
    ("XXVIII", "the twenty eighth"),
    ("XXVII", "the twenty seventh"),
    ("XXVI", "the twenty sixth"),
    ("XXV", "the twenty fifth"),
    ("XXIV", "the twenty fourth"),
    ("XXIII", "the twenty third"),
    ("XXII", "the twenty second"),
    ("XXI", "the twenty first"),
    ("XX", "the twentieth"),
    ("XIX", "the nineteenth"),
    ("XVIII", "the eighteenth"),
    ("XVII", "the seventeenth"),
    ("XVI", "the sixteenth"),
    ("XV", "the fifteenth"),
    ("XIV", "the fourteenth"),
    ("XIII", "the thirteenth"),
    ("XII", "the twelfth"),
    ("XI", "the eleventh"),
    ("X", "the tenth"),
    ("IX", "the ninth"),
    ("VIII", "the eighth"),
    ("VII", "the seventh"),
    ("VI", "the sixth"),
    ("V", "the fifth"),
    ("IV", "the fourth"),
    ("III", "the third"),
    ("II", "the second"),
    # ("I", "the first"),
]

# Pre-compile a single pattern that matches any Roman numeral as a whole word
# (preceded by a word char or space, not inside another word).
# Longest-first ordering in the alternation guarantees greedy correct matching.
_ROMAN_RE = re.compile(
    r"(?<=\b)(" + "|".join(re.escape(r) for r, _ in _ROMAN_MAP) + r")(?=\b)",
)
_ROMAN_DICT = dict(_ROMAN_MAP)


def _normalize_for_speech(text: str) -> str:
    """Rewrite symbols, abbreviations, and numerals into spoken English."""

    # ── Strip markdown formatting artifacts ───────────────────────────────────
    text = re.sub(r"\*+", "", text)  # asterisks (bold/italic)
    text = re.sub(r"#{1,6}\s*", "", text)  # heading markers
    text = re.sub(r"`", "", text)  # inline code backticks
    text = re.sub(r"^\s*[-•]\s+", "", text, flags=re.MULTILINE)  # bullet markers

    # ── Decimal numbers → "point" (must run BEFORE degree/unit subs) ──────────
    # Converts "34.7" → "34 point 7", prevents Kokoro treating '.' as sentence end.
    text = re.sub(r"(\d)\.(\d)", r"\1 point \2", text)

    # ── Degree symbols → spoken form ──────────────────────────────────────────
    text = re.sub(r"°C\b", " degrees Celsius", text)
    text = re.sub(r"°F\b", " degrees Fahrenheit", text)
    text = re.sub(r"°", " degrees", text)

    # ── Common unit symbols ───────────────────────────────────────────────────
    text = re.sub(r"(\d)\s*%", r"\1 percent", text)
    text = re.sub(r"(\d)\s*km/h\b", r"\1 kilometers per hour", text)
    text = re.sub(r"(\d)\s*mph\b", r"\1 miles per hour", text)
    text = re.sub(r"(\d)\s*km\b", r"\1 kilometers", text)
    text = re.sub(r"(\d)\s*kg\b", r"\1 kilograms", text)
    text = re.sub(r"(\d)\s*lbs?\b", r"\1 pounds", text)
    text = re.sub(r"(\d)\s*ft\b", r"\1 feet", text)
    text = re.sub(r"(\d)\s*m\b", r"\1 meters", text)

    # ── Roman numerals (whole words only) ─────────────────────────────────────
    # Must run AFTER other substitutions so "°C" etc. are already gone.
    text = _ROMAN_RE.sub(lambda m: _ROMAN_DICT[m.group(0)], text)

    # ── Clean up stray whitespace ─────────────────────────────────────────────
    text = re.sub(r"  +", " ", text).strip()

    return text


# ── Module-level pipeline cache ────────────────────────────────────────────────
_pipeline = None


def get_pipeline():
    """Load the Kokoro pipeline (downloads model on first run, ~80MB)."""
    global _pipeline
    if _pipeline is None:
        print("[TTS] Loading Kokoro pipeline...", flush=True)
        from kokoro import KPipeline

        _pipeline = KPipeline(lang_code=LANG_CODE)
        print("[TTS] Kokoro ready.", flush=True)
    return _pipeline


def warmup():
    """
    Pre-load the Kokoro model + voice at startup.
    Also runs a tiny synthesis to JIT-compile any remaining code paths.
    """
    pipe = get_pipeline()
    # Warm-up synthesis (result is discarded)
    try:
        for _, _, _ in pipe("Hello.", voice=VOICE, speed=SPEED):
            break  # only need the first chunk
    except Exception:
        pass
    print("[TTS] Warm-up complete.", flush=True)


def _synth_and_play(text: str):
    """Synthesize a single chunk of text and play it."""
    text = _normalize_for_speech(text.strip())
    if not text:
        return
    pipe = get_pipeline()
    try:
        for _gs, _ps, audio in pipe(text, voice=VOICE, speed=SPEED):
            if audio is not None and len(audio) > 0:
                wav_np = np.array(audio, dtype=np.float32)
                sd.play(wav_np, samplerate=SAMPLE_RATE)
                sd.wait()
    except Exception as e:
        print(f"[TTS] Synthesis error: {e}")


class StreamingTTS:
    """
    Two-stage pipelined TTS:

        Sentence Queue  →  [Synth Worker]  →  Audio Queue  →  [Play Worker]

    Synthesis of sentence N+1 overlaps playback of sentence N,
    eliminating the dead gap between sentences.
    """

    def __init__(self):
        self._buffer = ""
        self._sentence_q: queue.Queue[str | None] = queue.Queue()
        self._audio_q: queue.Queue[tuple[np.ndarray, int] | None] = queue.Queue()

        self._synth_thread = threading.Thread(target=self._synth_worker, daemon=True)
        self._play_thread = threading.Thread(target=self._play_worker, daemon=True)
        self._synth_thread.start()
        self._play_thread.start()

    # ── Workers ────────────────────────────────────────────────────────────────

    def _synth_worker(self):
        """Take sentences, synthesize with Kokoro, push PCM to audio queue."""
        pipe = get_pipeline()

        while True:
            text = self._sentence_q.get()
            if text is None:
                self._audio_q.put(None)  # signal play worker to stop
                self._sentence_q.task_done()
                break

            text = _normalize_for_speech(text.strip())
            if not text:
                self._sentence_q.task_done()
                continue

            try:
                # Kokoro's generator yields (graphemes, phonemes, audio) per sentence.
                # For short text it's usually a single chunk.
                for _gs, _ps, audio in pipe(text, voice=VOICE, speed=SPEED):
                    if audio is not None and len(audio) > 0:
                        wav_np = np.array(audio, dtype=np.float32)
                        self._audio_q.put((wav_np, SAMPLE_RATE))
            except Exception as e:
                print(f"[TTS] Synth error: {e}")

            self._sentence_q.task_done()

    def _play_worker(self):
        """Play audio chunks back-to-back with minimal gap."""
        while True:
            item = self._audio_q.get()
            if item is None:
                self._audio_q.task_done()
                break
            samples, sr = item
            sd.play(samples, samplerate=sr)
            sd.wait()
            self._audio_q.task_done()

    # ── Public API (same interface as tts_engine_alt.py) ───────────────────────

    def feed(self, token: str):
        """Feed a streamed token. Sentences are auto-detected and queued."""
        self._buffer += token
        parts = SENTENCE_END.split(self._buffer)
        if len(parts) > 1:
            for sentence in parts[:-1]:
                if sentence.strip():
                    self._sentence_q.put(sentence.strip())
            self._buffer = parts[-1]

    def flush(self):
        """Flush remaining buffered text as a final utterance."""
        if self._buffer.strip():
            self._sentence_q.put(self._buffer.strip())
            self._buffer = ""

    def wait(self):
        """Block until all queued audio has finished playing."""
        self._sentence_q.join()
        self._audio_q.join()

    def stop(self):
        """Signal workers to shut down after finishing current work."""
        self._sentence_q.put(None)
