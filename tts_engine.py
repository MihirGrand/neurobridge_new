import numpy as np
import sounddevice as sd
import torch
from TTS.api import TTS

# Recommended model: good quality, fits in 4GB VRAM, real-time capable
# MODEL_NAME = "tts_models/en/ljspeech/tacotron2-DDC"
# For multilingual / more natural voice (uses ~2GB VRAM, ~1-2s latency):
MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
#
# MODEL_NAME = "tts_models/en/vctk/vits"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[TTS] Using device: {DEVICE}")

_tts = None  # lazy-load


def get_tts() -> TTS:
    global _tts
    if _tts is None:
        print(f"[TTS] Loading model '{MODEL_NAME}'...")
        _tts = TTS(model_name=MODEL_NAME, progress_bar=False).to(DEVICE)
        print("[TTS] Model loaded.")
    return _tts


# def speak(text: str, speaker: str = None):
#     """
#     Synthesize text and play it immediately via sounddevice (no file written).
#     For XTTS-v2, pass speaker="Ana Florence" or similar.
#     """
#     tts = get_tts()

#     kwargs = {"text": text}
#     if speaker and "xtts" in MODEL_NAME:
#         kwargs["speaker"] = speaker

#     # Returns a numpy float32 array at model's sample rate
#     wav = tts.tts(**kwargs)
#     wav_np = np.array(wav, dtype=np.float32)

#     sample_rate = tts.synthesizer.output_sample_rate
#     sd.play(wav_np, samplerate=sample_rate)
#     sd.wait()  # block until done


def speak(text: str):
    tts = get_tts()
    wav = tts.tts(
        text=text,
        speaker="Ana Florence",  # or "Claribel Dervla", "Daisy Studious", "Gracie Wise"
        language="en",
    )
    wav_np = np.array(wav, dtype=np.float32)
    sd.play(wav_np, samplerate=tts.synthesizer.output_sample_rate)
    sd.wait()


# def speak(text: str):
#     tts = get_tts()
#     wav = tts.tts(text=text, speaker="p273")
#     wav_np = np.array(wav, dtype=np.float32)
#     sd.play(wav_np, samplerate=tts.synthesizer.output_sample_rate)
#     sd.wait()
