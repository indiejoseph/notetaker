import os
import asyncio
from livekit.agents import llm
from livekit.plugins import openai
from core.state import TranscriptState


class EntityExtractorAgent:
    """
    Entity Extractor Agent: Extracts people, organizations, etc. from transcript.
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
            "Extract: People, Organizations, Locations, Dates, Key terms, Products. "
            "Output only the extracted entities as a bulleted list grouped by category."
        )

    async def run(self, state: TranscriptState):
        """
        Background task triggered by TranscriptState.
        """
        async with self._lock:
            if len(state.lines) == state.entities_processed:
                return

            # Snapshot the count now — lines may keep arriving while the LLM call is awaited
            snapshot_len = len(state.lines)
            transcript = state.get_transcript_snapshot()
            timestamp = state.format_timestamp(state.line_timestamps[-1])

            chat_ctx = llm.ChatContext()
            chat_ctx.add_message(role="system", content=self.instructions)
            chat_ctx.add_message(
                role="user", content=f"Extract entities from:\n\n{transcript}"
            )

            print(f"[EntityExtractorAgent] Running LLM chat for {timestamp}...")
            try:
                response = await self._llm.chat(chat_ctx=chat_ctx).collect()
                print(f"[EntityExtractorAgent] LLM Response received: {response}")
                entity_text = response.text

                state.entities.append((timestamp, entity_text))
                state.entities_processed = snapshot_len
                print(f"[EntityExtractorAgent] Entities updated at {timestamp}")
            except Exception as e:
                print(f"[EntityExtractorAgent] Error: {e}")
