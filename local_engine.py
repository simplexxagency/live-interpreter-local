import io
import json
import os
import queue
import re
import threading
import time
import wave

import numpy as np
import torch
from scipy.signal import resample_poly
from piper import PiperVoice, SynthesisConfig
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    M2M100ForConditionalGeneration,
    M2M100Tokenizer,
)

ASR_FOLDER = "whisper-large-v3-turbo"
MT_FOLDER = "m2m100_418M"
MADLAD_FOLDER = "madlad400-3b-ct2"
MADLAD_MIN_MODEL_BYTES = 2_900_000_000
DEVICE_RATE = 48000
ASR_RATE = 16000


def pcm48_to_float16k(raw):
    x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if not len(x):
        return x

    # Remove DC offset, then use a proper anti-aliasing low-pass filter before
    # reducing 48 kHz audio to Whisper's native 16 kHz. The previous x[::3]
    # decimation could fold high-frequency speech energy back into the voice band.
    x = x - float(np.mean(x))
    y = resample_poly(x, up=1, down=3).astype(np.float32, copy=False)

    # Give quiet but valid utterances a modest lift. Use a robust percentile
    # instead of the absolute peak so a single click/pop does not set the gain.
    level = float(np.percentile(np.abs(y), 95)) if len(y) else 0.0
    if 0.0 < level < 0.08:
        gain = min(3.0, 0.12 / level)
        y = y * gain

    return np.clip(y, -0.98, 0.98).astype(np.float32, copy=False)


def resample_pcm16(raw, source_rate, target_rate=DEVICE_RATE):
    x = np.frombuffer(raw, dtype=np.int16)
    if not len(x) or source_rate == target_rate:
        return raw
    count = max(1, int(round(len(x) * target_rate / source_rate)))
    source_pos = np.arange(len(x), dtype=np.float64)
    target_pos = np.linspace(0, len(x) - 1, count, dtype=np.float64)
    y = np.interp(target_pos, source_pos, x.astype(np.float64))
    return np.clip(y, -32768, 32767).astype(np.int16).tobytes()


class LocalModels:
    def __init__(self, base_dir, status_cb):
        self.base_dir = base_dir
        self.status_cb = status_cb
        self.asr_lock = threading.Lock()
        self.glossary = self._load_glossary()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self._load()

    def _load_glossary(self):
        path = os.path.join(self.base_dir, "glossary.json")
        try:
            with open(path, "r", encoding="utf-8") as handle:
                return json.load(handle)
        except Exception:
            return {"uk_to_en_output": {}, "en_to_uk_output": {}}

    def _apply_glossary(self, text, source_lang, target_lang):
        key = f"{source_lang}_to_{target_lang}_output"
        for old, new in self.glossary.get(key, {}).items():
            text = re.sub(rf"\b{re.escape(old)}\b", new, text, flags=re.IGNORECASE)
        return text

    @staticmethod
    def _dedupe_sentences(text):
        parts = re.split(r"(?<=[.!?])\s+", text.strip())
        kept = []
        for part in parts:
            normalized = re.sub(r"\s+", " ", part).strip().casefold()
            if not normalized:
                continue
            if kept:
                prev = re.sub(r"\s+", " ", kept[-1]).strip().casefold()
                if normalized == prev:
                    continue
            kept.append(part.strip())
        return " ".join(kept)

    def _prepare_tts_text(self, text, language):
        if language != "uk":
            return text
        for old, new in self.glossary.get("uk_tts", {}).items():
            text = re.sub(re.escape(old), new, text, flags=re.IGNORECASE)
        # Ukrainian Piper voice has no Latin phonemes. Spell any leftovers softly
        # instead of silently dropping them.
        latin_names = {
            "a": "ей", "b": "бі", "c": "сі", "d": "ді", "e": "і",
            "f": "еф", "g": "джі", "h": "ейч", "i": "ай", "j": "джей",
            "k": "кей", "l": "ел", "m": "ем", "n": "ен", "o": "оу",
            "p": "пі", "q": "к'ю", "r": "ар", "s": "ес", "t": "ті",
            "u": "ю", "v": "ві", "w": "дабл ю", "x": "екс",
            "y": "вай", "z": "зі",
        }
        def spell(match):
            return " ".join(latin_names.get(ch.lower(), ch) for ch in match.group(0))
        return re.sub(r"[A-Za-z]+", spell, text)

    def _load(self):
        models_root = os.path.join(self.base_dir, "models")
        asr_path = os.path.join(models_root, ASR_FOLDER)
        mt_path = os.path.join(models_root, MT_FOLDER)

        self.status_cb(f"Loading Whisper on {self.device.upper()}...")
        self.asr_processor = AutoProcessor.from_pretrained(
            asr_path, local_files_only=True
        )
        self.asr_model = AutoModelForSpeechSeq2Seq.from_pretrained(
            asr_path,
            dtype=self.dtype,
            local_files_only=True,
        ).to(self.device)
        self.asr_model.eval()

        self._load_translator(models_root, mt_path)

        voices = os.path.join(models_root, "piper")
        en_voice = os.path.join(voices, "en_US-ryan-medium.onnx")
        uk_voice = os.path.join(voices, "uk_UA-ukrainian_tts-medium.onnx")
        self.status_cb("Loading local voices...")
        self.voice_en = PiperVoice.load(en_voice)
        self.voice_uk = PiperVoice.load(uk_voice)
        self.status_cb(
            f"Local AI ready | ASR: {self.device.upper()} | MT: {self.mt_backend} | TTS: CPU"
        )

    def _load_translator(self, models_root, m2m_path):
        self.mt_lock = threading.Lock()
        self.mt_backend = "M2M100"
        self.mt_model = None
        self.mt_tokenizers = {}
        self.madlad_translator = None
        self.madlad_tokenizer = None
        self._dll_handles = []

        madlad_path = os.path.join(models_root, MADLAD_FOLDER)
        madlad_bin = os.path.join(madlad_path, "model.bin")
        madlad_ready = (
            os.path.isfile(madlad_bin)
            and os.path.getsize(madlad_bin) >= MADLAD_MIN_MODEL_BYTES
            and os.path.isfile(os.path.join(madlad_path, "sentencepiece.model"))
        )

        if madlad_ready:
            try:
                self.status_cb("Loading MADLAD-400 3B translator...")
                if os.name == "nt":
                    torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
                    if os.path.isdir(torch_lib):
                        self._dll_handles.append(os.add_dll_directory(torch_lib))

                import ctranslate2
                from sentencepiece import SentencePieceProcessor

                self.madlad_tokenizer = SentencePieceProcessor()
                self.madlad_tokenizer.load(
                    os.path.join(madlad_path, "sentencepiece.model")
                )

                if self.device == "cuda":
                    self.madlad_translator = ctranslate2.Translator(
                        madlad_path,
                        device="cuda",
                        compute_type="int8_float16",
                    )
                    self.mt_backend = "MADLAD-3B CUDA int8"
                else:
                    self.madlad_translator = ctranslate2.Translator(
                        madlad_path,
                        device="cpu",
                        compute_type="int8",
                    )
                    self.mt_backend = "MADLAD-3B CPU int8"
                return
            except Exception as exc:
                self.status_cb(f"MADLAD load failed, using M2M100: {exc}")

        self.status_cb(f"Loading M2M100 translator on {self.device.upper()}...")
        self.mt_model = M2M100ForConditionalGeneration.from_pretrained(
            m2m_path,
            dtype=self.dtype,
            local_files_only=True,
        ).to(self.device)
        self.mt_model.eval()
        self.mt_model.generation_config.early_stopping = False
        for lang in ("uk", "en"):
            tok = M2M100Tokenizer.from_pretrained(
                m2m_path, local_files_only=True
            )
            tok.src_lang = lang
            self.mt_tokenizers[lang] = tok

    def _detect_language_confidence(self, features, attention_mask):
        config = self.asr_model.generation_config
        decoder_ids = torch.full(
            (features.shape[0], 1),
            int(config.decoder_start_token_id),
            device=self.device,
            dtype=torch.long,
        )
        logits = self.asr_model(
            input_features=features,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_ids,
            use_cache=False,
        ).logits[:, -1]

        lang_items = list(config.lang_to_id.items())
        lang_ids = [token_id for _, token_id in lang_items]
        lang_logits = logits[0, lang_ids].float()
        probs = torch.softmax(lang_logits, dim=-1)
        best_index = int(torch.argmax(probs).item())
        token, _ = lang_items[best_index]
        language = token.replace("<|", "").replace("|>", "")
        confidence = float(probs[best_index].item())
        return language, confidence

    def transcribe(self, raw48, language, enforce_language=True, min_confidence=0.80):
        audio = pcm48_to_float16k(raw48)
        if len(audio) < ASR_RATE // 5:
            return ""
        inputs = self.asr_processor(
            audio,
            sampling_rate=ASR_RATE,
            return_tensors="pt",
            return_attention_mask=True,
        )
        features = inputs.input_features.to(self.device, dtype=self.dtype)
        attention_mask = inputs.attention_mask.to(self.device)
        with self.asr_lock, torch.inference_mode():
            detected_language, confidence = self._detect_language_confidence(
                features, attention_mask
            )
            if enforce_language and (
                detected_language != language or confidence < min_confidence
            ):
                return ""

            ids = self.asr_model.generate(
                features,
                attention_mask=attention_mask,
                language=language,
                task="transcribe",
                max_new_tokens=192,
                num_beams=5,
                do_sample=False,
                early_stopping=True,
            )
        text = self.asr_processor.batch_decode(
            ids, skip_special_tokens=True
        )[0].strip()

        # Common Whisper hallucinations triggered by residual non-speech.
        normalized = re.sub(r"[^\w]+", " ", text.lower(), flags=re.UNICODE).strip()
        hallucinations = {
            "thank you",
            "thanks",
            "thanks for watching",
            "thank you for watching",
            "дякую",
            "дякую за перегляд",
        }
        if normalized in hallucinations and confidence < 0.95:
            return ""
        return text

    def translate(self, text, source_lang, target_lang):
        if target_lang not in ("uk", "en") or source_lang not in ("uk", "en"):
            raise ValueError(f"Unsupported language pair: {source_lang}->{target_lang}")

        if self.madlad_translator is not None:
            input_tokens = self.madlad_tokenizer.encode(
                f"<2{target_lang}> {text}",
                out_type=str,
            )
            with self.mt_lock:
                result = self.madlad_translator.translate_batch(
                    [input_tokens],
                    batch_type="tokens",
                    max_batch_size=1024,
                    beam_size=4,
                    max_decoding_length=256,
                    no_repeat_ngram_size=2,
                    repetition_penalty=1.2,
                )[0]
            translated = self.madlad_tokenizer.decode(
                result.hypotheses[0]
            ).strip()
            translated = self._dedupe_sentences(translated)
            return self._apply_glossary(
                translated, source_lang, target_lang
            )

        tokenizer = self.mt_tokenizers[source_lang]
        batch = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=384,
        )
        batch = {k: v.to(self.device) for k, v in batch.items()}
        target_id = tokenizer.get_lang_id(target_lang)

        with self.mt_lock, torch.inference_mode():
            out = self.mt_model.generate(
                **batch,
                forced_bos_token_id=target_id,
                max_new_tokens=256,
                num_beams=1,
                do_sample=False,
            )
        translated = tokenizer.batch_decode(
            out, skip_special_tokens=True
        )[0].strip()
        return self._apply_glossary(translated, source_lang, target_lang)

    @staticmethod
    def _voice_to_pcm48(voice, text, speaker_id=None, length_scale=0.94):
        mem = io.BytesIO()
        syn_config = SynthesisConfig(
            speaker_id=speaker_id,
            length_scale=length_scale,
            normalize_audio=True,
            volume=1.0,
        )
        with wave.open(mem, "wb") as wav_file:
            voice.synthesize_wav(
                text,
                wav_file,
                syn_config=syn_config,
            )
        mem.seek(0)
        with wave.open(mem, "rb") as wav_file:
            rate = wav_file.getframerate()
            raw = wav_file.readframes(wav_file.getnframes())
        return resample_pcm16(raw, rate)

    def synthesize(self, text, language):
        if language == "en":
            return self._voice_to_pcm48(self.voice_en, text, length_scale=1.30)
        if language == "uk":
            tts_text = self._prepare_tts_text(text, "uk").lower()
            return self._voice_to_pcm48(
                self.voice_uk, tts_text, speaker_id=1, length_scale=0.94
            )
        raise ValueError(f"Unsupported TTS language: {language}")


class DirectionWorker:
    def __init__(
        self, name, models, source_lang, target_lang,
        output_cb, status_cb, event_cb, source_event, translation_event,
    ):
        self.name = name
        self.models = models
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.output_cb = output_cb
        self.status_cb = status_cb
        self.event_cb = event_cb
        self.source_event = source_event
        self.translation_event = translation_event
        self.queue = queue.Queue(maxsize=3)
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def submit(self, raw48):
        try:
            self.queue.put_nowait((time.monotonic(), raw48))
        except queue.Full:
            try:
                self.queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.queue.put_nowait((time.monotonic(), raw48))
            except queue.Full:
                pass

    def _run(self):
        while not self.stop_event.is_set():
            try:
                queued_at, raw = self.queue.get(timeout=0.2)
            except queue.Empty:
                continue
            started = time.monotonic()
            try:
                self.status_cb("recognizing…")
                source = self.models.transcribe(raw, self.source_lang)
                if not source:
                    self.status_cb("listening")
                    continue
                self.event_cb(self.source_event, source + "\n")

                self.status_cb("translating…")
                translated = self.models.translate(
                    source, self.source_lang, self.target_lang
                )
                self.event_cb(self.translation_event, translated + "\n")

                self.status_cb("speaking…")
                audio = self.models.synthesize(translated, self.target_lang)
                self.output_cb(audio)
                elapsed = time.monotonic() - started
                queued = started - queued_at
                self.status_cb(f"listening | last {elapsed:.2f}s | queue {queued:.2f}s")
            except Exception as exc:
                self.status_cb("error")
                self.event_cb("error", f"{self.name}: {exc}\n")

    def stop(self):
        self.stop_event.set()
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)


class TextDirectionWorker:
    """Transcribe + translate only. Voice output is always a manual user action."""

    def __init__(
        self, name, models, source_lang, target_lang,
        status_cb, result_cb, error_cb,
    ):
        self.name = name
        self.models = models
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.status_cb = status_cb
        self.result_cb = result_cb
        self.error_cb = error_cb
        self.queue = queue.Queue(maxsize=6)
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def submit(self, raw48):
        try:
            self.queue.put_nowait((time.monotonic(), raw48))
        except queue.Full:
            try:
                self.queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.queue.put_nowait((time.monotonic(), raw48))
            except queue.Full:
                pass

    def _run(self):
        while not self.stop_event.is_set():
            try:
                queued_at, raw = self.queue.get(timeout=0.2)
            except queue.Empty:
                continue

            started = time.monotonic()
            try:
                self.status_cb("recognizing…")
                source = self.models.transcribe(
                    raw,
                    self.source_lang,
                    enforce_language=False,
                )
                if not source:
                    self.status_cb("listening")
                    continue

                self.status_cb("translating…")
                translated = self.models.translate(
                    source,
                    self.source_lang,
                    self.target_lang,
                )
                elapsed = time.monotonic() - started
                queued = started - queued_at
                self.result_cb(source, translated, elapsed, queued)
                self.status_cb(
                    f"listening | last {elapsed:.2f}s | queue {queued:.2f}s"
                )
            except Exception as exc:
                self.status_cb("error")
                self.error_cb(f"{self.name}: {exc}")

    def stop(self):
        self.stop_event.set()
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)
