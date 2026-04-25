# 🧠 AI Meeting Intelligence System

> **Upload a meeting recording → Get transcript, summary, action items & deadlines — 100% local, no paid APIs.**

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Gradio](https://img.shields.io/badge/UI-Gradio-orange?logo=gradio)
![Whisper](https://img.shields.io/badge/STT-Whisper-green)
![Ollama](https://img.shields.io/badge/LLM-Ollama-purple)

---

## 📋 Features

| Feature | Description |
|---------|-------------|
| 🎙️ **Speech-to-Text** | Whisper (CTranslate2/C++ engine) with timestamps |
| 📝 **Smart Summary** | LLM-generated concise meeting summary |
| ✅ **Action Items** | Auto-extracted tasks with assignee + priority |
| ⏰ **Deadline Detection** | Finds time-bound commitments from discussion |
| 🎛️ **Audio Preprocessing** | FFmpeg normalization (any format → 16kHz WAV) |
| 🔒 **100% Local** | Everything runs on your machine — no data leaves |

---

## 🏗️ Project Structure

```
ai-meeting-intelligence/
├── app.py                          # Gradio UI + entry point
├── pipeline/
│   ├── __init__.py
│   ├── audio_preprocessor.py       # FFmpeg audio preprocessing
│   ├── transcriber.py              # Whisper speech-to-text
│   ├── llm_analyzer.py             # Ollama LLM analysis + prompts
│   └── meeting_pipeline.py         # End-to-end orchestrator
├── models/
│   ├── __init__.py
│   └── schemas.py                  # Data models (dataclasses)
├── utils/
│   ├── __init__.py
│   ├── validators.py               # Input validation + dependency checks
│   └── formatters.py               # Output formatting for UI
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites

You need **3 external tools** installed on your system:

#### 1. Python 3.10+
```bash
python --version  # Should be 3.10 or higher
```

#### 2. FFmpeg
FFmpeg is used for audio preprocessing (format conversion, normalization).

**Windows:**
```bash
winget install FFmpeg
# OR download from: https://ffmpeg.org/download.html
# Add to PATH after downloading
```

**macOS:**
```bash
brew install ffmpeg
```

**Linux:**
```bash
sudo apt update && sudo apt install ffmpeg
```

**Verify:**
```bash
ffmpeg -version
```

#### 3. Ollama
Ollama runs LLMs locally for meeting analysis.

**Windows / macOS / Linux:**
```bash
# Download from: https://ollama.com/download
# Or on Linux:
curl -fsSL https://ollama.com/install.sh | sh
```

**Start Ollama and pull a model:**
```bash
# Start the server (if not auto-started)
ollama serve

# Pull the default model (in a new terminal)
ollama pull llama3.2

# Verify
ollama list
```

> **Alternative lightweight models:** `phi3`, `mistral`, `gemma2:2b`
> Change the model with: `LLM_MODEL=phi3 python app.py`

---

### Installation

```bash
# 1. Navigate to the project directory
cd ai-meeting-intelligence

# 2. Create a virtual environment (recommended)
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate

# 3. Install Python dependencies
pip install -r requirements.txt
```

---

### Run the App

```bash
python app.py
```

The Gradio UI will open at: `http://localhost:7860`

---

## ⚙️ Configuration

All settings can be configured via **environment variables**:

| Variable | Default | Description |
|----------|---------|-------------|
| `WHISPER_MODEL` | `base` | Whisper model size: `tiny`, `base`, `small`, `medium`, `large-v3` |
| `WHISPER_BACKEND` | `faster-whisper` | Backend: `faster-whisper` or `whisper-cpp` |
| `WHISPER_CPP_PATH` | `None` | Path to whisper.cpp binary (only for `whisper-cpp` backend) |
| `LLM_MODEL` | `llama3.2` | Ollama model name |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL |
| `DEVICE` | `auto` | Compute device: `auto`, `cpu`, `cuda` |
| `LANGUAGE` | `None` | Audio language (None = auto-detect) |

**Example — use a larger Whisper model and different LLM:**
```bash
WHISPER_MODEL=small LLM_MODEL=mistral python app.py
```

**Windows (PowerShell):**
```powershell
$env:WHISPER_MODEL="small"
$env:LLM_MODEL="mistral"
python app.py
```

---

## 🎯 Using whisper.cpp Backend

If you prefer the original whisper.cpp:

```bash
# 1. Clone and build whisper.cpp
git clone https://github.com/ggerganov/whisper.cpp.git
cd whisper.cpp
make

# 2. Download a model
bash models/download-ggml-model.sh base

# 3. Run the app with whisper.cpp backend
WHISPER_BACKEND=whisper-cpp WHISPER_CPP_PATH=./whisper.cpp/main python app.py
```

---

## 📖 Sample Input & Output

### Input
Any meeting audio file in `.mp3`, `.wav`, `.m4a`, `.ogg`, `.flac`, `.wma`, `.aac`, or `.webm` format.

**Example scenario:** A 5-minute product team standup meeting recording.

### Expected Output

**Transcript:**
```
[00:00 → 00:15] Good morning everyone, let's get started with our weekly standup.
[00:16 → 00:45] Sarah, can you give us an update on the frontend redesign?
[00:46 → 01:20] Sure. I've completed the new dashboard layout and the responsive
                 design is working. I still need to integrate the charting library
                 and that should be done by Wednesday.
[01:21 → 01:55] Great. Mike, what about the API endpoints?
[01:56 → 02:30] The user authentication endpoints are done. I'm working on the
                 data export API now. I'll have it ready by Friday for testing.
[02:31 → 03:00] Perfect. We need to have everything ready for the client demo
                 next Monday. Let's make sure all features are tested by end of day Friday.
```

**Summary:**
> The weekly standup covered progress on the frontend redesign and API development.
> Sarah reported completing the dashboard layout with responsive design and plans
> to integrate the charting library by Wednesday. Mike finished user authentication
> endpoints and is working on the data export API, targeting Friday completion.
> The team agreed that all features must be tested by Friday for the client demo
> scheduled for next Monday.

**Action Items:**
| Priority | Assignee | Task | Level |
|:--------:|----------|------|:-----:|
| 🔴 | **Sarah** | Integrate charting library into dashboard | `HIGH` |
| 🔴 | **Mike** | Complete data export API endpoints | `HIGH` |
| 🟡 | **Team** | Test all features before client demo | `MEDIUM` |

**Deadlines:**
| Task | Due Date | Assignee |
|------|----------|----------|
| Charting library integration | Wednesday | Sarah |
| Data export API ready for testing | Friday | Mike |
| All features tested | End of day Friday | Team |
| Client demo | Next Monday | Team |

---

## 🔧 Troubleshooting

| Issue | Solution |
|-------|----------|
| `FFmpeg not found` | Install FFmpeg and add to PATH. Verify with `ffmpeg -version` |
| `Ollama connection refused` | Start Ollama: `ollama serve`. Verify at `http://localhost:11434` |
| `Model not found` | Pull the model: `ollama pull llama3.2` |
| `CUDA out of memory` | Set `DEVICE=cpu` or use smaller Whisper model (`tiny`) |
| `Empty transcript` | Audio may be silent, too noisy, or in unsupported language |
| `Slow processing` | Use `WHISPER_MODEL=tiny` and lightweight LLM like `phi3` |

---

## 🏛️ Architecture

```
┌──────────────┐     ┌──────────────────┐     ┌───────────────┐
│  Audio File  │────▶│  FFmpeg Preproc.  │────▶│  16kHz Mono   │
│  (.mp3/.wav) │     │  (normalize,      │     │  WAV File     │
└──────────────┘     │   resample)       │     └───────┬───────┘
                     └──────────────────┘             │
                                                      ▼
                     ┌──────────────────┐     ┌───────────────┐
                     │  Whisper (C++)   │◀────│  Transcriber  │
                     │  CTranslate2     │     │  Module       │
                     └──────────────────┘     └───────┬───────┘
                                                      │
                            Transcript                │
                                                      ▼
┌──────────────┐     ┌──────────────────┐     ┌───────────────┐
│  Summary     │◀────│  Ollama LLM      │◀────│  LLM Analyzer │
│  Actions     │     │  (llama3.2)      │     │  Module       │
│  Deadlines   │     └──────────────────┘     └───────────────┘
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Gradio UI   │
│  (Tabs View) │
└──────────────┘
```

---

## 📄 License

This project is open-source. Use it freely for personal or commercial purposes.
