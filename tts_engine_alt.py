import queue
import re
import threading

import numpy as np
import sounddevice as sd
import torch
from TTS.api import TTS

MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SENTENCE_END = re.compile(r"(?<=[.!?])\s+")

_tts = None


def get_tts():
    global _tts
    if _tts is None:
        print(f"[TTS] Loading XTTS-v2 on {DEVICE}...")
        _tts = TTS(model_name=MODEL_NAME, progress_bar=False)
        if DEVICE == "cuda":
            _tts = _tts.to(DEVICE)
        print("[TTS] Ready.")
    return _tts


def _synth_and_play(text: str):
    text = text.strip()
    if not text:
        return
    tts = get_tts()
    wav = tts.tts(text=text, speaker="Daisy Studious", language="en")
    wav_np = np.array(wav, dtype=np.float32)
    sd.play(wav_np, samplerate=tts.synthesizer.output_sample_rate)
    sd.wait()


def speak(text: str):
    _synth_and_play(text)


class StreamingTTS:
    def __init__(self):
        self._buffer = ""
        self._audio_queue = queue.Queue()
        self._worker = threading.Thread(target=self._audio_worker, daemon=True)
        self._worker.start()

    def _audio_worker(self):
        while True:
            chunk = self._audio_queue.get()
            if chunk is None:
                break
            _synth_and_play(chunk)
            self._audio_queue.task_done()

    def feed(self, token: str):
        self._buffer += token
        parts = SENTENCE_END.split(self._buffer)
        if len(parts) > 1:
            for sentence in parts[:-1]:
                if sentence.strip():
                    self._audio_queue.put(sentence.strip())
            self._buffer = parts[-1]

    def flush(self):
        if self._buffer.strip():
            self._audio_queue.put(self._buffer.strip())
            self._buffer = ""

    def wait(self):
        self._audio_queue.join()

    def stop(self):
        self._audio_queue.put(None)
