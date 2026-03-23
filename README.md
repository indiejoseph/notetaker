# Omni Notes

An AI-powered notetaking app that transcribes audio (file upload or mic), generates incremental summaries, extracts named entities, and produces a final refined transcript and consolidated summary.

Built with Gradio, LiveKit Agents, Silero VAD, and Qwen3-Omni.

## Features

- **Verbatim ASR**: Transcribes speech in the original language using Qwen3-Omni's Chat Completions API with audio input.
- **Context-aware transcription**: Each speech segment is transcribed with the last 5 lines, latest summary, and all known entities as context.
- **Incremental summarization**: Summary updated every 5 transcript lines, concurrently with entity extraction.
- **Entity extraction**: Tracks people, organizations, locations, dates, key terms, and products — every 5 lines.
- **Transcript refinement**: After transcription, lines are corrected in batches of 10 using the full entity and summary context.
- **Final summary**: One consolidated meeting summary generated from all incremental summaries and the refined transcript.
- **Gradio UI**: File upload or live microphone input.

## Pipeline

```mermaid
flowchart TD
    A([Audio Input]) --> B[Silero VAD]

    B -->|END_OF_SPEECH + frames| C[Qwen3-Omni ASR\nChat Completions API]

    C -->|text + timestamp| D[(TranscriptState\nsorted by timestamp)]

    D -->|every 5 lines\nconcurrently| E[SummarizerAgent\nincremental]
    D -->|every 5 lines\nconcurrently| F[EntityExtractorAgent\nincremental]

    E -->|state.summaries| D
    F -->|state.entities| D

    D -->|context: last 5 lines\n+ latest summary\n+ all entities| C

    D -->|after all lines\n+ summarizer done\n+ extractor done| G[TranscriptRefinerAgent\n10 lines at a time]

    G -->|refined lines| D

    D -->|refined transcript\n+ all summaries\n+ all entities| H[SummarizerAgent\nfinalize]

    H -->|state.final_summary| I([Gradio UI])
    D -->|state.lines\nstate.summaries\nstate.entities| I
```

## Architecture

| File | Role |
|------|------|
| `app.py` | Gradio web UI, entry point |
| `pipeline.py` | Orchestrates all agents, drives the end-to-end flow |
| `core/state.py` | `TranscriptState` — shared store, timestamp-sorted insertion, 5-line trigger |
| `agents/transcriber.py` | Silero VAD → Qwen3-Omni ASR, self-contained with context prompt |
| `agents/summarizer.py` | Incremental LLM summarization + `finalize()` for final summary |
| `agents/entities_extractor.py` | Incremental LLM entity extraction |
| `agents/refiner.py` | Post-processing refinement pass in batches of 10 |

## Installation

```bash
pip install -r requirements.txt
```

Create a `.env` file:

```env
QWEN_BASE_URL=http://<host>/v1
QWEN_API_KEY=sk-...
QWEN_MODEL=qwen3-omni
```

## Usage

```bash
# Web UI
python app.py

# Offline verification
python verify_pipeline.py
```
