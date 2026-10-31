"""Download model weights into the ignored models/ folder (never into Git)."""
from pathlib import Path
import shutil
from huggingface_hub import hf_hub_download, snapshot_download

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
MODELS.mkdir(parents=True, exist_ok=True)

def snapshot(repo: str, folder: str, patterns: list[str]) -> None:
    path = MODELS / folder
    print(f"Downloading {repo} to {path} (resumable)...", flush=True)
    snapshot_download(repo_id=repo, local_dir=str(path), allow_patterns=patterns)

snapshot(
    "openai/whisper-large-v3-turbo",
    "whisper-large-v3-turbo",
    [
        "config.json", "generation_config.json", "preprocessor_config.json",
        "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
        "added_tokens.json", "vocab.json", "merges.txt", "normalizer.json",
        "model.safetensors",
    ],
)
snapshot(
    "facebook/m2m100_418M",
    "m2m100_418M",
    [
        "config.json", "generation_config.json", "pytorch_model.bin",
        "sentencepiece.bpe.model", "special_tokens_map.json",
        "tokenizer_config.json", "vocab.json",
    ],
)
voices = [
    "en/en_US/ryan/medium/en_US-ryan-medium.onnx",
    "en/en_US/ryan/medium/en_US-ryan-medium.onnx.json",
    "uk/uk_UA/ukrainian_tts/medium/uk_UA-ukrainian_tts-medium.onnx",
    "uk/uk_UA/ukrainian_tts/medium/uk_UA-ukrainian_tts-medium.onnx.json",
]
piper = MODELS / "piper"
piper.mkdir(parents=True, exist_ok=True)
for item in voices:
    target = piper / Path(item).name
    if target.exists() and target.stat().st_size > 100:
        print(f"Already downloaded: {target.name}", flush=True)
        continue
    print(f"Downloading Piper voice: {target.name}", flush=True)
    source = hf_hub_download(repo_id="rhasspy/piper-voices", filename=item)
    shutil.copy2(source, target)

required = [
    MODELS / "whisper-large-v3-turbo" / "model.safetensors",
    MODELS / "m2m100_418M" / "pytorch_model.bin",
    piper / "en_US-ryan-medium.onnx",
    piper / "uk_UA-ukrainian_tts-medium.onnx",
]
missing = [str(p) for p in required if not p.is_file()]
if missing:
    raise SystemExit("Missing model files:\n" + "\n".join(missing))
print("\nALL_LOCAL_MODELS_READY. You can now run run_local.bat.", flush=True)
