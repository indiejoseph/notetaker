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
            model=os.environ.get("LLM_MODEL", "qwen3-omni"),
            api_key=os.environ.get("LLM_API_KEY", "sk-xxx"),
            base_url=os.environ.get("LLM_BASE_URL", "http://localhost/v1"),
        )
        self.instructions = (
            "擷取以下類別的實體：人物、組織、地點、日期、關鍵術語、產品。"
            "僅輸出擷取到的實體，以分類的項目符號列表呈現。所有輸出必須使用繁體中文。"
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
            transcript = state.get_transcript_snapshot(add_speaker=False)
            timestamp = state.format_timestamp(state.line_timestamps[-1])

            chat_ctx = llm.ChatContext()
            chat_ctx.add_message(role="system", content=self.instructions)
            chat_ctx.add_message(
                role="user", content=f"請從以下逐字稿中擷取實體：\n\n{transcript}"
            )

            print(f"[EntityExtractorAgent] Running LLM chat for {timestamp}...")
            state.status = "Entity Extractor: extracting entities..."
            try:
                response = await self._llm.chat(chat_ctx=chat_ctx).collect()
                print(f"[EntityExtractorAgent] LLM Response received: {response}")
                entity_text = response.text

                state.entities.append((timestamp, entity_text))
                print(f"[EntityExtractorAgent] Entities updated at {timestamp}")
            except Exception as e:
                print(f"[EntityExtractorAgent] Error: {e}")
            finally:
                state.entities_processed = snapshot_len
                state.status = ""
