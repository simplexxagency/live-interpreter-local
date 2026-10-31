from collections import deque
import threading
import time

import numpy as np
import sounddevice as sd
import webrtcvad

API_RATE = 24000
DEVICE_RATE = 48000
BLOCK_MS = 20
DEVICE_FRAMES = DEVICE_RATE * BLOCK_MS // 1000


def downsample_48_to_24(raw):
    return np.frombuffer(raw, np.int16)[::2].astype(np.int16, copy=False).tobytes()


def upsample_24_to_48(raw):
    return np.repeat(np.frombuffer(raw, np.int16), 2).astype(np.int16, copy=False).tobytes()


class BufferedOutput:
    def __init__(self, device_id):
        self.device_id = device_id
        self.buf = bytearray()
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = None

    def push_api24(self, raw):
        self.push_pcm48(upsample_24_to_48(raw))

    def push_pcm48(self, raw):
        with self.lock:
            self.buf.extend(raw)
            max_bytes = DEVICE_RATE * 2 * 8
            if len(self.buf) > max_bytes:
                del self.buf[:-max_bytes]

    def _take(self, n):
        with self.lock:
            k = min(n, len(self.buf))
            data = bytes(self.buf[:k])
            del self.buf[:k]
        return data + b"\x00" * (n - k)

    def start(self):
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        nbytes = DEVICE_FRAMES * 2
        with sd.RawOutputStream(
            samplerate=DEVICE_RATE, blocksize=DEVICE_FRAMES, device=self.device_id,
            channels=2, dtype="int16", latency="low"
        ) as stream:
            while not self.stop_event.is_set():
                mono = np.frombuffer(self._take(nbytes), np.int16)
                stereo = np.repeat(mono[:, None], 2, axis=1).reshape(-1).astype(np.int16)
                stream.write(stereo.tobytes())

    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=1.5)


class HeadphoneMixer:
    def __init__(self, device_id, gain_getter):
        self.device_id = device_id
        self.gain_getter = gain_getter
        self.translation = bytearray()
        self.original = bytearray()
        self.lock = threading.Lock()
        self.last_translation = 0.0
        self.stop_event = threading.Event()
        self.thread = None

    def start(self):
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def push_translation_api24(self, raw):
        self.push_translation_48(upsample_24_to_48(raw))

    def push_translation_48(self, raw):
        with self.lock:
            self.translation.extend(raw)
            self.last_translation = time.monotonic()

    def push_original_48(self, raw):
        with self.lock:
            self.original.extend(raw)
            max_bytes = DEVICE_RATE * 2 * 3
            if len(self.original) > max_bytes:
                del self.original[:-max_bytes]

    @staticmethod
    def _take(buf, n):
        k = min(n, len(buf))
        data = bytes(buf[:k])
        del buf[:k]
        return data + b"\x00" * (n - k)

    def _run(self):
        nbytes = DEVICE_FRAMES * 2
        with sd.RawOutputStream(
            samplerate=DEVICE_RATE, blocksize=DEVICE_FRAMES, device=self.device_id,
            channels=2, dtype="int16", latency="low"
        ) as stream:
            while not self.stop_event.is_set():
                with self.lock:
                    tr = self._take(self.translation, nbytes)
                    orig = self._take(self.original, nbytes)
                    tr_active = time.monotonic() - self.last_translation < 0.35
                tr = np.frombuffer(tr, np.int16).astype(np.int32)
                orig = np.frombuffer(orig, np.int16).astype(np.int32)
                gain = float(self.gain_getter()) * (0.35 if tr_active else 1.0)
                mixed = np.clip(tr + orig * gain, -32768, 32767).astype(np.int16)
                stereo = np.repeat(mixed[:, None], 2, axis=1).reshape(-1).astype(np.int16)
                stream.write(stereo.tobytes())

    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=1.5)


class Capture:
    def __init__(
        self, device_id, send_cb, passthrough=None, gate=True,
        min_threshold=180.0, pre_roll_ms=300, hangover_ms=700
    ):
        self.device_id = device_id
        self.send_cb = send_cb
        self.passthrough = passthrough
        self.stream = None
        self.gate = gate
        self.min_threshold = float(min_threshold)
        self.noise_floor = max(40.0, self.min_threshold / 3.0)
        self.pre_roll = deque(maxlen=max(1, pre_roll_ms // BLOCK_MS))
        self.hangover_frames = max(1, hangover_ms // BLOCK_MS)
        self.active = False
        self.silence_frames = 0
        self.sent_seconds = 0.0

    def _rms(self, raw):
        x = np.frombuffer(raw, np.int16).astype(np.float32)
        if not len(x):
            return 0.0
        return float(np.sqrt(np.mean(x * x)))

    def _send(self, raw):
        self.send_cb(downsample_48_to_24(raw))
        self.sent_seconds += len(raw) / (DEVICE_RATE * 2.0)

    def _process_gate(self, raw):
        rms = self._rms(raw)
        threshold = max(self.min_threshold, self.noise_floor * 2.8)
        voice = rms >= threshold

        if self.active:
            self._send(raw)
            if voice:
                self.silence_frames = 0
            else:
                self.silence_frames += 1
                if self.silence_frames >= self.hangover_frames:
                    self.active = False
                    self.silence_frames = 0
                    self.pre_roll.clear()
            return

        self.pre_roll.append(raw)
        if voice:
            self.active = True
            self.silence_frames = 0
            while self.pre_roll:
                self._send(self.pre_roll.popleft())
        else:
            capped = min(rms, self.min_threshold * 1.5)
            self.noise_floor = self.noise_floor * 0.985 + capped * 0.015

    def start(self):
        def callback(indata, frames, time_info, status):
            raw = bytes(indata)
            if self.passthrough:
                self.passthrough(raw)
            if self.gate:
                self._process_gate(raw)
            else:
                self._send(raw)

        self.stream = sd.RawInputStream(
            samplerate=DEVICE_RATE, blocksize=DEVICE_FRAMES, device=self.device_id,
            channels=1, dtype="int16", latency="low", callback=callback
        )
        self.stream.start()

    def stop(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

class UtteranceCapture:
    """Local VAD that emits complete 48 kHz mono PCM utterances."""

    def __init__(
        self, device_id, utterance_cb, passthrough=None,
        min_threshold=160.0, pre_roll_ms=300, end_silence_ms=760,
        start_voice_ms=100, max_utterance_ms=15000,
        vad_aggressiveness=2, min_utterance_ms=400,
    ):
        self.device_id = device_id
        self.utterance_cb = utterance_cb
        self.passthrough = passthrough
        self.min_threshold = float(min_threshold)
        self.noise_floor = max(35.0, self.min_threshold / 3.0)
        self.pre_roll = deque(maxlen=max(1, pre_roll_ms // BLOCK_MS))
        self.end_frames = max(1, end_silence_ms // BLOCK_MS)
        self.start_frames = max(1, start_voice_ms // BLOCK_MS)
        self.max_frames = max(1, max_utterance_ms // BLOCK_MS)
        self.min_utterance_seconds = max(0.1, min_utterance_ms / 1000.0)
        self.vad = webrtcvad.Vad(int(vad_aggressiveness))
        self.stream = None
        self.active = False
        self.voice_streak = 0
        self.silence_frames = 0
        self.frames_in_utterance = 0
        self.utterance = bytearray()

    @staticmethod
    def _rms(raw):
        x = np.frombuffer(raw, np.int16).astype(np.float32)
        return float(np.sqrt(np.mean(x * x))) if len(x) else 0.0

    def _reset(self):
        self.active = False
        self.voice_streak = 0
        self.silence_frames = 0
        self.frames_in_utterance = 0
        self.utterance.clear()
        self.pre_roll.clear()

    def _emit(self):
        if not self.utterance:
            self._reset()
            return
        raw = bytes(self.utterance)
        duration = len(raw) / (DEVICE_RATE * 2.0)
        self._reset()
        if duration >= self.min_utterance_seconds:
            self.utterance_cb(raw)

    def _process(self, raw):
        rms = self._rms(raw)
        threshold = max(self.min_threshold, self.noise_floor * 2.2)
        try:
            speech_like = self.vad.is_speech(raw, DEVICE_RATE)
        except Exception:
            speech_like = True
        voice = speech_like and rms >= threshold

        if self.active:
            self.utterance.extend(raw)
            self.frames_in_utterance += 1
            if voice:
                self.silence_frames = 0
            else:
                self.silence_frames += 1
            if self.silence_frames >= self.end_frames or self.frames_in_utterance >= self.max_frames:
                self._emit()
            return

        self.pre_roll.append(raw)
        if voice:
            self.voice_streak += 1
        else:
            self.voice_streak = 0
            capped = min(rms, self.min_threshold * 1.5)
            self.noise_floor = self.noise_floor * 0.985 + capped * 0.015

        if self.voice_streak >= self.start_frames:
            self.active = True
            self.silence_frames = 0
            self.frames_in_utterance = len(self.pre_roll)
            while self.pre_roll:
                self.utterance.extend(self.pre_roll.popleft())

    def start(self):
        def callback(indata, frames, time_info, status):
            raw = bytes(indata)
            if self.passthrough:
                self.passthrough(raw)
            self._process(raw)

        self.stream = sd.RawInputStream(
            samplerate=DEVICE_RATE, blocksize=DEVICE_FRAMES, device=self.device_id,
            channels=1, dtype="int16", latency="low", callback=callback
        )
        self.stream.start()

    def stop(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        if self.active and self.utterance:
            self._emit()
