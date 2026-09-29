"""
voice_engine.py -- speech capture and transcription for the Room-
Temperature Resistance Measurement tab.

Two transcription engines behind one common interface, so they can be
A/B tested directly against each other on real audio before picking
one for the live loop:

  WhisperEngine -- faster-whisper (CTranslate2-optimized Whisper).
      Much stronger general-purpose accuracy, handles open vocabulary
      well, normalizes spoken numbers to digits in its own output.
      Needs to download model weights once (~75-500MB depending on
      size) the first time it runs, then works fully offline. Not a
      true streaming engine -- transcribes one complete audio buffer
      at a time (fine here: SilenceSegmenter below already hands it
      one full utterance at a time, it never needs word-by-word
      partial results).

  VoskEngine -- Vosk (Kaldi-based). Smaller (~40MB for the small
      English model), noticeably faster on CPU, and genuinely
      streaming (feeds audio in small chunks, can report partial
      results as the user is still talking). Its real advantage here:
      Vosk supports a CONSTRAINED GRAMMAR -- a fixed list of words/
      phrases it's allowed to recognize -- and this tab's entire
      vocabulary really is closed (digits, "point", "kilohm", "open",
      "short", a handful of commands). Constraining recognition to
      exactly that list tends to beat general-purpose Whisper on
      accuracy for THIS specific narrow domain, while also being
      faster and lighter. RESISTANCE_GRAMMAR below is that word list.

Recommendation to start from: Vosk with RESISTANCE_GRAMMAR as the
live, hands-free engine (fast, streaming, and the closed grammar
should make it very hard for it to mishear a number as some other
word); Whisper as an optional slower high-accuracy re-check for any
utterance you're not sure about, or for open-ended commands beyond
this fixed vocabulary. Validate this recommendation against real
audio in your own measurement environment (see README.md) before
relying on it in production -- model weights for both engines are
downloaded separately (see each engine's own documentation) and are
not bundled with this repository.

Both engines expect mono 16kHz audio, matching what SilenceSegmenter
below captures.
"""

from __future__ import annotations

import json
import queue
import time
from dataclasses import dataclass
from typing import Callable, Iterator, List, Optional, Protocol

import numpy as np

SAMPLE_RATE = 16000

# The closed vocabulary this tab will ever actually hear -- used to
# build Vosk's constrained grammar, and doubles as documentation of
# exactly what resistance_parser.py needs to be able to parse.
RESISTANCE_GRAMMAR: List[str] = [
    "zero", "oh", "one", "two", "three", "four", "five", "six", "seven",
    "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen",
    "fifteen", "sixteen", "seventeen", "eighteen", "nineteen",
    "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
    "hundred", "thousand", "point", "and",
    "ohm", "ohms", "kilohm", "kilohms", "megaohm", "megaohms",
    "open", "short", "shorted", "overload",
    "undo", "redo", "skip", "next", "repeat", "cancel", "stop", "pause",
    "[unk]",  # Vosk convention: catch-all bucket for anything outside the grammar
]


class TranscriptionEngine(Protocol):
    """Common interface both engines implement, so the rest of the
    tab (and any test/comparison harness) doesn't care which one is
    active."""

    def transcribe(self, audio_int16: np.ndarray) -> str:
        """audio_int16: mono int16 PCM samples at SAMPLE_RATE. Returns
        the transcript text (empty string if nothing recognizable)."""
        ...


class WhisperEngine:
    def __init__(self, model_size: str = "base.en", device: str = "cpu",
                 compute_type: str = "int8",
                 initial_prompt: str = (
                     "Resistance measurements read from a multimeter, in ohms "
                     "or kilohms. Examples: twelve point four kilohms, "
                     "forty one point eight, open, short."
                 )):
        # Imported here, not at module load, so importing this file
        # doesn't require faster-whisper to be installed if you end up
        # only using VoskEngine (or vice versa for vosk below).
        from faster_whisper import WhisperModel
        self._model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self._initial_prompt = initial_prompt

    def transcribe(self, audio_int16: np.ndarray) -> str:
        audio_f32 = (audio_int16.astype(np.float32) / 32768.0)
        segments, _info = self._model.transcribe(
            audio_f32, language="en", beam_size=5,
            initial_prompt=self._initial_prompt,
            vad_filter=False,  # SilenceSegmenter already did the segmentation
        )
        return " ".join(seg.text for seg in segments).strip()


class VoskEngine:
    def __init__(self, model_path: str, grammar: Optional[List[str]] = RESISTANCE_GRAMMAR):
        from vosk import Model, KaldiRecognizer
        self._model = Model(model_path)
        self._grammar = grammar
        self._KaldiRecognizer = KaldiRecognizer

    def _new_recognizer(self):
        if self._grammar:
            return self._KaldiRecognizer(self._model, SAMPLE_RATE, json.dumps(self._grammar))
        return self._KaldiRecognizer(self._model, SAMPLE_RATE)

    def transcribe(self, audio_int16: np.ndarray) -> str:
        rec = self._new_recognizer()  # fresh recognizer per utterance -- SilenceSegmenter
                                       # already hands us one complete utterance at a time,
                                       # so there's no cross-utterance state to preserve
        rec.AcceptWaveform(audio_int16.tobytes())
        result = json.loads(rec.FinalResult())
        return result.get("text", "").strip()


# --- hands-free capture: continuous mic listening, segmented into utterances ---

@dataclass
class SegmenterConfig:
    """Tune these against YOUR actual probe-station environment (fan/
    pump hum, room echo, mic distance) -- these starting values are
    reasonable defaults, not measured against your real setup."""
    silence_rms_threshold: float = 500.0     # int16 RMS below this = silence
    min_speech_ms: int = 200                 # ignore blips shorter than this
    silence_hangover_ms: int = 600           # how long a pause must last before
                                              # the utterance is considered finished
    max_utterance_ms: int = 6000             # hard cap, in case something goes wrong
    block_ms: int = 30                       # audio callback block size


class _SegmenterCore:
    """The actual speech/silence state machine, deliberately split out
    from SilenceSegmenter below so it can be unit-tested (see
    test_segmenter_core.py) by feeding it synthetic blocks directly --
    no microphone, no sounddevice/PortAudio needed. SilenceSegmenter is
    a thin wrapper that feeds this real audio blocks from the mic;
    everything about WHEN an utterance is considered finished lives
    here and only here, so testing this class is testing the real
    logic, not a simplified stand-in for it."""

    def __init__(self, config: SegmenterConfig):
        self.config = config
        self._buffer: List[np.ndarray] = []
        self._in_speech = False
        self._speech_ms = 0
        self._silence_ms = 0

    def push_block(self, block: np.ndarray) -> Optional[np.ndarray]:
        """Feed one block of int16 mono samples. Returns a completed
        utterance (concatenated int16 array) if this block finished
        one, otherwise None."""
        cfg = self.config
        rms = float(np.sqrt(np.mean(block.astype(np.float64) ** 2))) if block.size else 0.0

        if rms >= cfg.silence_rms_threshold:
            self._buffer.append(block)
            self._speech_ms += cfg.block_ms
            self._silence_ms = 0
            self._in_speech = True
        elif self._in_speech:
            self._buffer.append(block)  # keep trailing silence in the clip -- helps the
                                         # ASR engine hear the word actually finish
            self._silence_ms += cfg.block_ms

        total_ms = self._speech_ms + self._silence_ms
        should_cut = self._in_speech and (
            (self._silence_ms >= cfg.silence_hangover_ms and self._speech_ms >= cfg.min_speech_ms)
            or total_ms >= cfg.max_utterance_ms
        )
        if not should_cut:
            return None

        utterance = np.concatenate(self._buffer) if self._buffer else np.array([], dtype=np.int16)
        self._buffer = []
        self._in_speech = False
        self._speech_ms = 0
        self._silence_ms = 0
        return utterance if utterance.size > 0 else None


class SilenceSegmenter:
    """Wraps a live microphone stream and yields one int16 numpy array
    per detected utterance (voice, then a long-enough pause), via
    _SegmenterCore above. If it proves too sensitive to the room's
    actual noise floor (fan/pump hum in a real fab is a realistic
    concern), the same interface can be swapped to a proper VAD model
    (e.g. silero-vad) without changing anything else in the tab --
    this class's job is just "produce one utterance's worth of audio
    at a time," however it decides where the boundaries are.

    Usage:
        seg = SilenceSegmenter()
        for utterance_audio in seg.listen():
            text = engine.transcribe(utterance_audio)
            ...
    (listen() is a blocking generator -- run it in a background thread
    when wiring this into a Qt GUI, and hand results back to the main
    thread via a Qt signal rather than touching widgets directly from
    that thread.)
    """

    def __init__(self, config: SegmenterConfig = SegmenterConfig()):
        self.config = config
        self._stop = False

    def stop(self):
        self._stop = True

    def listen(self) -> Iterator[np.ndarray]:
        import sounddevice as sd

        cfg = self.config
        block_samples = int(SAMPLE_RATE * cfg.block_ms / 1000)
        audio_q: "queue.Queue[np.ndarray]" = queue.Queue()
        core = _SegmenterCore(cfg)

        def callback(indata, frames, time_info, status):
            audio_q.put(indata.copy())

        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16",
                             blocksize=block_samples, callback=callback):
            while not self._stop:
                try:
                    block = audio_q.get(timeout=0.5)
                except queue.Empty:
                    continue
                utterance = core.push_block(block.reshape(-1))
                if utterance is not None:
                    yield utterance
