import os
from openai import AsyncOpenAI
from core.state import TranscriptState

BATCH_SIZE = 10


class TranscriptRefinerAgent:
    """
    Refinement Agent: Post-processes the final transcript in batches of 10 lines,
    using accumulated summaries and entities to fix inconsistencies — names, jargon,
    code-switching — while preserving the original language and meaning.
    """

    def __init__(self):
        self._client = AsyncOpenAI(
            api_key=os.environ.get("QWEN_API_KEY", "sk-xxx"),
            base_url=os.environ.get("QWEN_BASE_URL", "http://localhost/v1"),
        )
        self._model = os.environ.get("QWEN_MODEL", "qwen3-omni")

    async def run(self, state: TranscriptState):
        if not state.lines:
            return

        summary_context = state.get_summary_snapshot()
        entity_context = state.get_entity_snapshot()

        print(f"[RefinerAgent] Refining {len(state.lines)} lines in batches of {BATCH_SIZE}...")

        for batch_start in range(0, len(state.lines), BATCH_SIZE):
            batch_lines = state.lines[batch_start : batch_start + BATCH_SIZE]
            refined = await self._refine_batch(batch_lines, summary_context, entity_context)
            if refined:
                state.lines[batch_start : batch_start + len(refined)] = refined

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
            context_block += f"### Summary of the full conversation\n{summary_context}\n\n"
        if entity_context:
            context_block += f"### Known entities (people, places, terms)\n{entity_context}\n\n"

        prompt = (
            f"{context_block}"
            f"### Transcript lines to refine\n{numbered}\n\n"
            "Fix transcription errors in the lines above:\n"
            "- Correct names, places, and technical terms using the known entities\n"
            "- Fix obvious mishearings or garbled words based on context\n"
            "- Preserve the original language of each line (do NOT translate)\n"
            "- Keep the meaning and tone unchanged\n\n"
            f"Return exactly {len(lines)} lines, numbered the same way (1. 2. 3. …). "
            "Output nothing else."
        )

        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a transcript editor. You correct transcription errors "
                            "using provided context. You never translate, summarise, or add content."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
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
