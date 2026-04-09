# CLAUDE.md

## Project Overview

**Omni Notes** — an AI-powered notetaking app that transcribes audio (file upload or mic), then generates summaries and extracts named entities. Built with Gradio web UI, LiveKit Agents framework, Silero VAD, and Qwen3-Omni for both ASR and LLM tasks.

## Running the App

```bash
python app.py
# Opens at http://127.0.0.1:7860
```

Other scripts:
- `verify_pipeline.py` — tests pipeline with mock transcript lines
- `test_api.py` — tests Qwen3-Omni API endpoints directly

## Architecture

```
Audio Input → NotetakingPipeline → TranscriberAgent (VAD + STT)
                                          ↓ every 5 lines
                             SummarizerAgent + EntityExtractorAgent (parallel, LLM)
                                          ↓
                                     Gradio UI (streaming)
```

### Key Files

| File | Purpose |
|------|---------|
| `app.py` | Gradio web UI, entry point |
| `pipeline.py` | Orchestrates all agents, reads WAV into 20ms frames |
| `qwen_stt.py` | Custom LiveKit STT node wrapping Qwen3-Omni via OpenAI-compatible API |
| `core/state.py` | Central `TranscriptState` — stores transcript, summaries, entities; triggers background tasks every 5 lines |
| `agents/transcriber.py` | Silero VAD + QwenSTT → timestamped transcript lines |
| `agents/summarizer.py` | LLM-based summarization of transcript segments |
| `agents/entities_extractor.py` | LLM-based named entity extraction (people, orgs, locations, dates, key terms, products) |

## Configuration

Credentials and model endpoint live in `.env`:

```env
LLM_BASE_URL="http://<host>/v1"   # Qwen3-Omni OpenAI-compatible endpoint
LLM_API_KEY="sk-..."
LLM_MODEL="qwen3-omni"
QWEN_ASR_URL="http://<host>/v1"   # Qwen ASR endpoint
QWEN_ASR_API_KEY="sk-..."
QWEN_ASR_MODEL="Qwen/Qwen3-ASR-1.7B"
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Web UI | Gradio |
| Backend | Python 3.14, asyncio |
| Speech pipeline | LiveKit Agents, Silero VAD |
| ASR + LLM | Qwen3-Omni (OpenAI-compatible API) |
| Audio I/O | scipy, numpy, wave |
| API client | AsyncOpenAI |

## Key Patterns

- **QwenSTT** (`qwen_stt.py`) buffers audio frames into a WAV, then sends to `/v1/chat/completions` with the audio attached — not the standard `/v1/audio/transcriptions` endpoint.
- **Background processing** is triggered by `TranscriptState` callbacks after every 5 new transcript lines; summarizer and entity extractor run concurrently via `asyncio.gather`.
- **Gradio streaming** uses an async generator in `app.py` that yields UI updates as transcript lines and LLM results arrive.

## Dependencies

Install with:
```bash
pip install -r requirements.txt
```

Key packages: `gradio`, `livekit-agents`, `livekit-plugins-silero`, `livekit-plugins-openai`, `openai`, `dashscope`, `python-dotenv`, `numpy`, `scipy`.
