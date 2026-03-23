import os
import asyncio
from livekit.agents import llm
from livekit.plugins import openai
from core.state import TranscriptState


class SummarizerAgent:
    """
    Summarizer Agent: Takes the full transcript and produces a concise summary.
    Uses standalone LLM with .collect().
    """

    def __init__(self):
        self._lock = asyncio.Lock()
        self._llm = openai.LLM(
            model=os.environ.get("QWEN_MODEL", "qwen3-omni"),
            api_key=os.environ.get("QWEN_API_KEY", "sk-xxx"),
            base_url=os.environ.get("QWEN_BASE_URL", "http://localhost/v1"),
        )
        self.instructions = (
            "You are a Summarizer Agent. Provide a concise, highly readable summary "
            "of the given transcript. Highlight the main points discussed."
        )

    async def run(self, state: TranscriptState):
        """
        Background task triggered by TranscriptState.
        """
        async with self._lock:
            if len(state.lines) == state.summary_processed:
                return

            # Snapshot the count now — lines may keep arriving while the LLM call is awaited
            snapshot_len = len(state.lines)
            transcript = state.get_transcript_snapshot()
            timestamp = state.format_timestamp(state.line_timestamps[-1])

            user_content = f"You are taking notes for a meeting. Please summarize the following transcript:\n\n{transcript}"
            entity_snapshot = state.get_entity_snapshot()
            if entity_snapshot:
                user_content += (
                    f"\n\nKnown entities extracted so far:\n{entity_snapshot}"
                )

            chat_ctx = llm.ChatContext()
            chat_ctx.add_message(role="system", content=self.instructions)
            chat_ctx.add_message(role="user", content=user_content)

            print(f"[SummarizerAgent] Running LLM chat for {timestamp}...")
            try:
                response = await self._llm.chat(chat_ctx=chat_ctx).collect()
                print(f"[SummarizerAgent] LLM Response received: {response}")
                summary_text = response.text

                state.summaries.append((timestamp, summary_text))
                state.summary_processed = snapshot_len
                print(f"[SummarizerAgent] Summary updated at {timestamp}")
            except Exception as e:
                print(f"[SummarizerAgent] Error: {e}")

    async def finalize(self, state: TranscriptState):
        """
        Produces a single consolidated summary from all incremental summaries,
        the full refined transcript, and all extracted entities.
        Runs once at the end of the pipeline.
        """
        if not state.summaries and not state.lines:
            return

        incremental = state.get_summary_snapshot()
        transcript = state.get_transcript_snapshot()
        entity_snapshot = state.get_entity_snapshot()

        user_content = (
            "You are taking notes for a meeting. "
            "Using the incremental summaries and the full transcript below, "
            "write one final, comprehensive meeting summary.\n\n"
        )
        if incremental:
            user_content += f"### Incremental summaries\n{incremental}\n\n"
        user_content += f"### Full transcript\n{transcript}"
        if entity_snapshot:
            user_content += f"\n\n### Known entities\n{entity_snapshot}"

        chat_ctx = llm.ChatContext()
        chat_ctx.add_message(role="system", content=self.instructions)
        chat_ctx.add_message(role="user", content=user_content)

        print("[SummarizerAgent] Generating final summary...")
        try:
            response = await self._llm.chat(chat_ctx=chat_ctx).collect()
            state.final_summary = response.text
            print("[SummarizerAgent] Final summary complete.")
        except Exception as e:
            print(f"[SummarizerAgent] Error generating final summary: {e}")
