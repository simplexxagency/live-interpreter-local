# Live Interpreter LOCAL 🎧

An experimental **offline English ↔ Ukrainian meeting assistant** for people who understand English but sometimes miss details when other participants speak quickly. Built with Python/Tkinter, Whisper, M2M100, Piper, and VoiceMeeter Banana.

It transcribes speech and displays editable translation cards. Outgoing synthesized speech is **only sent to the meeting when you press the button**. It is **not** a zero-latency, professional simultaneous interpreter.

## What it does

- English → Ukrainian: capture meeting speech, display the original and its Ukrainian translation.
- Ukrainian → English: transcribe your speech, show and edit the translation, optionally preview it, and **manually** send synthesized English speech to the call.
- Keeps speech processing and translation **on your computer after model download**, with no per-minute AI API.
- Can use an optional locally supplied MADLAD-400 CTranslate2 model; the standard installer uses M2M100, which is much easier to set up.

## Requirements

- **Windows 10/11**, **Python 3.11–3.12** (64-bit), and working microphone/headphones.
- **VoiceMeeter Banana** installed for audio routing. The routing helper assumes the Banana layout; check your routes before joining a real call.
- NVIDIA GPU with compatible driver recommended; CPU installation is possible but speech processing will be significantly slower.
- Internet access and several GB of free disk space **for the initial package/model download**. Models are **not** included in this repository. Keep additional space for PyTorch, Python packages and download caches.

## Install

Download this repository (GitHub `Code → Download ZIP` or `git clone`), extract it, open **PowerShell in its folder**, and run:

```powershell
powershell -ExecutionPolicy Bypass -File .\setup_local.ps1
```

This creates `.venv-local`, installs a CUDA 12.8 PyTorch build and Python requirements, then **automatically downloads** Whisper large-v3-turbo, M2M100-418M, and two Piper voices from Hugging Face into `models/`. It can be rerun to resume interrupted model downloads.

CPU-only installation:

```powershell
powershell -ExecutionPolicy Bypass -File .\setup_local.ps1 -Torch cpu
```

If you already have the correct models installed or want to download them separately:

```powershell
powershell -ExecutionPolicy Bypass -File .\setup_local.ps1 -SkipModels
.\.venv-local\Scripts\python.exe .\download_models.py
```

**Start** by double-clicking `run_local.bat`. It launches in offline mode and expects the model files to exist in `models/`.

## Google Meet and VoiceMeeter Banana

1. Configure your real microphone and headset in the app's four dropdowns. Make sure to select the correct **Voicemeeter Out B1** capture device and **Voicemeeter AUX Input** playback device for the translated voice.
2. In **Google Meet**, set **Speaker = Voicemeeter Input** and **Microphone = Voicemeeter Out B2**.
3. Check the VoiceMeeter Banana inputs, outputs and monitoring before starting. Click **START LISTENING** in the app.
4. Read incoming translated cards. For outgoing cards, edit the English text if necessary, then click **Send to Meet**. Voice is *never* auto-sent.
5. Test routing in a private call before using the app in meetings. **Do not record or share others' conversations without consent.**

Audio device labels differ between PCs. The application lists detected input/output devices, so select the correct ones manually.

## Model sources / licenses

The downloader retrieves weights **directly from their respective model publishers**, not from this GitHub repository:
- [OpenAI Whisper large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo)
- [Meta M2M100 418M](https://huggingface.co/facebook/m2m100_418M)
- [Piper voices: Ryan (EN) and Ukrainian TTS (UK)](https://huggingface.co/rhasspy/piper-voices)

Review the upstream model cards, model/voice licensing terms and package licenses before redistribution. Model files belong to their respective authors.

## Limitations

- Some phrases may be recognized or translated incorrectly; it takes time to detect phrase boundaries and run inference.
- Whisper and M2M100 can use substantial RAM/VRAM. First startup may take longer; CPU inference can be too slow for comfortable meetings.
- Only English ↔ Ukrainian is configured and VoiceMeeter Banana routing is Windows-specific.
- Experimental software. No warranty. The source is provided without bundled model weights.

## Development

Start the GUI with `run_local.bat`. App entry point: `launch_local.pyw` → `app_manual.py`. Models: `local_engine.py`. Audio: `audio_engine.py`. VoiceMeeter: `voicemeeter_control.py`.

Developed by Simplex Agency. Contributions and feedback welcome.
