import os
import asyncio
from livekit.agents import llm
from livekit.plugins import openai
from core.state import TranscriptState


class SummarizerAgent:
    """
    Summarizer Agent: Uses rolling summarization with 50-line windows.
    Prevents context limit overflow on long audio.
    """

    # Rolling window size: max lines per summary batch
    WINDOW_SIZE = 50
    # How many previous rolling summaries to include as context
    PRIOR_SUMMARIES = 3

    def __init__(self):
        self._lock = asyncio.Lock()
        self._llm = openai.LLM(
            model=os.environ.get("LLM_MODEL", "qwen3-omni"),
            api_key=os.environ.get("LLM_API_KEY", "sk-xxx"),
            base_url=os.environ.get("LLM_BASE_URL", "http://localhost/v1"),
        )
        self.instructions = (
            "你係一個摘要助手。請用繁體中文提供簡潔、易讀嘅摘要，"
            "重點標示討論嘅主要議題。所有輸出必須使用繁體中文。"
        )

    async def run(self, state: TranscriptState):
        """
        Background task triggered by TranscriptState.
        Uses rolling window: only summarize the last WINDOW_SIZE lines.
        """
        async with self._lock:
            current_line_count = len(state.lines)
            if current_line_count <= state.summary_processed:
                return

            # Get the last WINDOW_SIZE lines (or fewer if transcript is shorter)
            window_start = max(0, current_line_count - self.WINDOW_SIZE)
            window_lines = state.lines[window_start:current_line_count]

            if not window_lines:
                state.summary_processed = current_line_count
                return

            # Build transcript snippet for this window using state lists
            window_transcript = "\n".join(
                (
                    f"{state.format_timestamp(state.line_timestamps[i])} (Speaker {state.line_speakers[i]}): {state.lines[i]}"
                    if state.line_speakers[i] >= 1
                    else f"{state.format_timestamp(state.line_timestamps[i])} {state.lines[i]}"
                )
                for i in range(window_start, current_line_count)
            )

            window_end_time = state.format_timestamp(
                state.line_timestamps[current_line_count - 1]
            )

            # Include the last PRIOR_SUMMARIES rolling summaries as context, if available
            prior_count = min(self.PRIOR_SUMMARIES, len(state.summaries))
            if prior_count > 0:
                prior_snippets = []
                for s in state.summaries[-prior_count:]:
                    prior_snippets.append(
                        f"時間: {s['timestamp']} (行 {s['window_start']+1}-{s['window_end']}):\n{s['text']}"
                    )
                prior_text = "\n\n".join(prior_snippets)
                user_content = (
                    f"以下為最近 {prior_count} 段階段性摘要，請參考以保持摘要連貫性：\n\n{prior_text}\n\n"
                    f"請用繁體中文摘要以下逐字稿片段（共 {current_line_count - window_start} 行）：\n\n{window_transcript}"
                )
            else:
                user_content = f"請用繁體中文摘要以下逐字稿片段（共 {current_line_count - window_start} 行）：\n\n{window_transcript}"

            chat_ctx = llm.ChatContext()
            chat_ctx.add_message(role="system", content=self.instructions)
            chat_ctx.add_message(role="user", content=user_content)

            print(
                f"[SummarizerAgent] Running rolling window summary at {window_end_time} (lines {window_start+1}-{current_line_count})..."
            )
            state.status = "Summarizer: generating rolling summary..."
            try:
                response = await self._llm.chat(chat_ctx=chat_ctx).collect()
                print(f"[SummarizerAgent] LLM Response received: {response}")
                summary_text = response.text

                # Store rolling summary with window boundaries
                state.summaries.append(
                    {
                        "timestamp": window_end_time,
                        "window_start": window_start,
                        "window_end": current_line_count,
                        "text": summary_text,
                    }
                )
                print(
                    f"[SummarizerAgent] Rolling summary saved (window: {window_start+1}-{current_line_count})"
                )
            except Exception as e:
                print(f"[SummarizerAgent] Error: {e}")
            finally:
                state.summary_processed = current_line_count
                state.status = ""

    async def finalize(self, state: TranscriptState):
        """
        Produces a single consolidated summary from all rolling summaries.
        Runs once at the end of the pipeline.
        """
        if not state.summaries and not state.lines:
            return

        # Build final context from rolling summaries
        rolling_summaries_text = ""
        if state.summaries:
            rolling_summaries_text = "### 分段摘要\n"
            for summary in state.summaries:
                rolling_summaries_text += f"\n**時間: {summary['timestamp']}** (行 {summary['window_start']+1}-{summary['window_end']}):\n{summary['text']}\n"

        transcript = state.get_transcript_snapshot()
        entity_snapshot = state.get_entity_snapshot()

        user_content = (
            "你正在為會議做筆記。"
            "請根據以下的分段摘要與完整逐字稿，"
            "用繁體中文撰寫一份最終的完整會議摘要。\n\n"
        )
        if rolling_summaries_text:
            user_content += rolling_summaries_text + "\n\n"
        user_content += f"### 完整逐字稿\n{transcript}"
        if entity_snapshot:
            user_content += f"\n\n### 已擷取的實體\n{entity_snapshot}"

        chat_ctx = llm.ChatContext()
        chat_ctx.add_message(role="system", content=self.instructions)
        chat_ctx.add_message(role="user", content=user_content)

        print("[SummarizerAgent] Generating final consolidated summary...")
        state.status = "Summarizer: generating final summary..."
        try:
            response = await self._llm.chat(chat_ctx=chat_ctx).collect()
            state.final_summary = response.text
            print("[SummarizerAgent] Final summary complete.")
        except Exception as e:
            print(f"[SummarizerAgent] Error generating final summary: {e}")
        finally:
            state.status = ""
