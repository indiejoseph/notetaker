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
            model=os.environ.get("LLM_MODEL", "qwen3-omni"),
            api_key=os.environ.get("LLM_API_KEY", "sk-xxx"),
            base_url=os.environ.get("LLM_BASE_URL", "http://localhost/v1"),
        )
        self.instructions = (
            "你是一個摘要助手。請用繁體中文提供簡潔、易讀的逐字稿摘要，"
            "重點標示討論的主要議題。所有輸出必須使用繁體中文。"
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

            user_content = (
                f"你正在為會議做筆記。請用繁體中文摘要以下逐字稿：\n\n{transcript}"
            )

            chat_ctx = llm.ChatContext()
            chat_ctx.add_message(role="system", content=self.instructions)
            chat_ctx.add_message(role="user", content=user_content)

            print(f"[SummarizerAgent] Running LLM chat for {timestamp}...")
            state.status = "Summarizer: generating summary..."
            try:
                response = await self._llm.chat(chat_ctx=chat_ctx).collect()
                print(f"[SummarizerAgent] LLM Response received: {response}")
                summary_text = response.text

                state.summaries.append((timestamp, summary_text))
                print(f"[SummarizerAgent] Summary updated at {timestamp}")
            except Exception as e:
                print(f"[SummarizerAgent] Error: {e}")
            finally:
                state.summary_processed = snapshot_len
                state.status = ""

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
            "你正在為會議做筆記。"
            "請根據以下的階段性摘要與完整逐字稿，"
            "用繁體中文撰寫一份最終的完整會議摘要。\n\n"
        )
        if incremental:
            user_content += f"### 階段性摘要\n{incremental}\n\n"
        user_content += f"### 完整逐字稿\n{transcript}"
        if entity_snapshot:
            user_content += f"\n\n### 已擷取的實體\n{entity_snapshot}"

        chat_ctx = llm.ChatContext()
        chat_ctx.add_message(role="system", content=self.instructions)
        chat_ctx.add_message(role="user", content=user_content)

        print("[SummarizerAgent] Generating final summary...")
        state.status = "Summarizer: generating final summary..."
        try:
            response = await self._llm.chat(chat_ctx=chat_ctx).collect()
            state.final_summary = response.text
            print("[SummarizerAgent] Final summary complete.")
        except Exception as e:
            print(f"[SummarizerAgent] Error generating final summary: {e}")
        finally:
            state.status = ""
