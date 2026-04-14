import os
from openai import AsyncOpenAI
from core.state import TranscriptState

BATCH_SIZE = 40


class TranscriptRefinerAgent:
    """
    Refinement Agent: Post-processes the final transcript in batches of 10 lines,
    using accumulated summaries and entities to fix inconsistencies — names, jargon,
    code-switching — while preserving the original language and meaning.
    """

    def __init__(self):
        self._client = AsyncOpenAI(
            api_key=os.environ.get("LLM_API_KEY", "sk-xxx"),
            base_url=os.environ.get("LLM_BASE_URL", "http://localhost/v1"),
        )
        self._model = os.environ.get("LLM_MODEL", "qwen3-omni")

    async def run(self, state: TranscriptState):
        if not state.lines:
            return

        summary_context = state.get_summary_snapshot()
        entity_context = state.get_entity_snapshot()

        print(
            f"[RefinerAgent] Refining {len(state.lines)} lines in batches of {min(BATCH_SIZE, len(state.lines))}..."
        )

        total_batches = (len(state.lines) + BATCH_SIZE - 1) // BATCH_SIZE
        for batch_idx, batch_start in enumerate(range(0, len(state.lines), BATCH_SIZE)):
            state.status = (
                f"Refiner: processing batch {batch_idx + 1}/{total_batches}..."
            )
            batch_lines = state.lines[batch_start : batch_start + BATCH_SIZE]
            refined = await self._refine_batch(
                batch_lines, summary_context, entity_context
            )
            if refined:
                state.lines[batch_start : batch_start + len(refined)] = refined

        state.status = ""
        print("[RefinerAgent] Refinement complete.")

    async def _refine_batch(
        self,
        lines: list[str],
        summary_context: str,
        entity_context: str,
    ) -> list[str]:
        numbered = "\n".join(f"{i + 1}. {line}" for i, line in enumerate(lines))

        context_block = ""
        if summary_context:
            context_block += f"### 完整對話摘要\n{summary_context}\n\n"
        if entity_context:
            context_block += f"### 已知實體（人物、地點、術語）\n{entity_context}\n\n"

        prompt = (
            f"{context_block}"
            f"### 需要修正的逐字稿\n{numbered}\n\n"
            "修正以上逐字稿中的轉錄錯誤：\n"
            "- 根據已知實體修正人名、地名和專業術語\n"
            "- 根據上下文修正明顯的聽錯或亂碼\n"
            "- 所有簡體中文必須轉換為繁體中文\n"
            "- 保留粵語口語用詞，不要改為書面語\n"
            "- 保持原意和語氣不變\n\n"
            f"請回傳正好 {len(lines)} 行，以相同編號格式（1. 2. 3. …）。"
            "不要輸出任何其他內容。"
        )

        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "你係一個逐字稿編輯助手。你根據提供嘅上下文修正轉錄錯誤。"
                            "所有輸出必須使用繁體中文，將任何簡體中文轉換為繁體中文。"
                            "你唔會翻譯、摘要或添加內容。/no_think"
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                extra_body={"enable_thinking": False},
            )

            raw = response.choices[0].message.content or ""
            refined = self._parse_numbered_lines(raw, expected=len(lines))
            print(f"[RefinerAgent] Refined batch of {len(lines)} lines.")
            return refined

        except Exception as e:
            print(f"[RefinerAgent] Error refining batch: {e}")
            return lines  # fall back to original on error

    @staticmethod
    def _parse_numbered_lines(text: str, expected: int) -> list[str]:
        """Extract numbered lines (1. ... 2. ...) from LLM output."""
        lines = []
        for raw_line in text.splitlines():
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            # Strip leading "N." or "N) " prefix
            for sep in (". ", ") "):
                parts = raw_line.split(sep, 1)
                if len(parts) == 2 and parts[0].isdigit():
                    lines.append(parts[1].strip())
                    break
            else:
                lines.append(raw_line)  # no prefix found, keep as-is

        # If parsing went wrong, fall back to splitting by newline as-is
        if len(lines) != expected:
            fallback = [l.strip() for l in text.splitlines() if l.strip()]
            return fallback[:expected] if len(fallback) >= expected else lines
        return lines
