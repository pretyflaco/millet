# Requirements

> Part of [meetscribe](README.md) -- fully local meeting transcription.

The default configuration runs entirely on your machine: WhisperX +
pyannote-audio for transcription and diarization, Ollama for AI
summaries.  No cloud APIs needed for the local-first path.

The default summary backend is DeepSeek V4.1 Flash inside a Tinfoil
hardware-attested TEE (needs `TINFOIL_API_KEY`); local Ollama remains
available for a fully offline path.  Both are private — the cloud backends
that could read meeting content were removed in 0.19.0.  See *Summary
backends* in the [README](README.md) for the trust model of each.

## Hardware

| Component | Minimum | Recommended | Notes |
|-----------|---------|-------------|-------|
| GPU | NVIDIA with 4GB+ VRAM | NVIDIA RTX with 8GB+ VRAM | CUDA required for fast transcription |
| RAM | 8 GB | 16 GB+ | WhisperX + pyannote load into RAM alongside GPU VRAM |
| Storage | 5 GB free | 10 GB+ free | Model weights (~2 GB) + recordings (~30 MB/hr WAV) |
| Mic | Any input device | USB headset or webcam mic | Dual-channel capture needs both mic and system audio |

CPU-only mode works (`--device cpu`) but is significantly slower.

## Operating System

- **Linux** with PulseAudio or PipeWire (PulseAudio compat layer) for
  recording (`meet record`, `meet run`). Tested on Pop!_OS 22.04 (GNOME, X11)
  with PipeWire 1.0.3. Should work on Ubuntu 22.04+, Fedora 38+, Arch, etc.
- **macOS Apple Silicon** is supported for the post-capture pipeline
  (transcription, labeling, summarization, sync) as of v0.6.0. Install with
  `pip install 'meetscribe-offline[mlx]'`. Audio capture on macOS is not
  supported by `meet record`; feed audio in via `meet transcribe <file>`,
  or use [vezir](https://github.com/pretyflaco/vezir) to run a Mac as a
  transcription server with Linux/Android clients.
- **Windows is not supported.**

## Software Dependencies

### System packages

| Package | Purpose | Install |
|---------|---------|---------|
| Python 3.10+ | Runtime | `sudo apt install python3` |
| ffmpeg 4.4+ | Audio capture & format conversion | `sudo apt install ffmpeg` |
| PulseAudio tools | Audio device enumeration | `sudo apt install pulseaudio-utils` (usually pre-installed) |
| GTK3 | GUI widget (optional) | `sudo apt install gir1.2-gtk-3.0` (usually pre-installed) |

### Python packages

Core (required):
```
whisperx          # ASR + alignment + diarization pipeline
faster-whisper    # CTranslate2-based Whisper inference (installed with whisperx)
pyannote.audio    # Speaker diarization
torch             # PyTorch with CUDA support
click             # CLI framework
reportlab         # PDF generation
requests          # HTTP client (for Ollama API)
```

Install whisperx (which pulls in most dependencies):
```bash
pip install whisperx
pip install reportlab requests click
```

### Ollama (for AI meeting summaries)

| Requirement | Details |
|-------------|---------|
| Ollama | Install from https://ollama.com |
| Default model | `qwen3.5:9b` (6.6 GB download) |
| Alternative models | `gemma3:12b` (fastest), `qwen3:14b`, `glm-4.7-flash` (best quality, 19 GB) |

Pull the default model:
```bash
ollama pull qwen3.5:9b
```

Ollama is optional. Without it, transcription still works but summaries are skipped.

## API Keys / Tokens

| Token | Purpose | How to get it |
|-------|---------|---------------|
| HuggingFace (`HF_TOKEN`) | Required for pyannote speaker diarization models | 1. Create account at https://huggingface.co <br> 2. Accept terms at https://huggingface.co/pyannote/speaker-diarization-community-1 <br> 3. Generate token at https://huggingface.co/settings/tokens |

Set it in your shell:
```bash
export HF_TOKEN=hf_your_token_here
# Or add to ~/.bashrc for persistence
echo 'export HF_TOKEN=hf_your_token_here' >> ~/.bashrc
```

Without `HF_TOKEN`, transcription works but speaker diarization is disabled (all text attributed to one speaker).

## Model Downloads

On first run, these models are downloaded automatically (requires internet):

| Model | Size | Purpose |
|-------|------|---------|
| `openai/whisper-large-v3-turbo` | ~1.6 GB (CTranslate2) | Speech recognition |
| `pyannote/speaker-diarization-community-1` | ~50 MB | Speaker diarization |
| `wav2vec2` alignment model | ~360 MB | Word-level timestamp alignment |

After the first run, everything works offline.

## Quick Verification

```bash
meet check
```

This validates ffmpeg, PulseAudio, CUDA, whisperx, and HF_TOKEN are all set up correctly.
