import asyncio
import bisect


class TranscriptState:
    """
    Central shared state for the notetaking pipeline.
    Coordinates between Transcriber, Summarizer, and Entity Extractor.
    """

    def __init__(self):
        self.lines: list[str] = []
        self.line_timestamps: list[float] = []
        self.line_speakers: list[int] = []  # Speaker ID for each line
        self.summaries: list[tuple[str, str]] = []  # (timestamp, text)
        self.entities: list[tuple[str, str]] = []  # (timestamp, text)

        self.summary_processed = 0
        self.entities_processed = 0
        self.final_summary: str = ""
        self.on_update_trigger = None

    def set_update_trigger(self, callback):
        """
        Register a callback to be triggered when new content is added.
        """
        self.on_update_trigger = callback

    async def add_line(self, text: str, timestamp: float, speaker_id: int = -1):
        """
        Append a finalized transcript line and check if background updates should trigger.

        Args:
            text: Transcript text
            timestamp: Timestamp in seconds
            speaker_id: Speaker ID (-1 if unknown)
        """
        if not text.strip():
            return

        print(f"[TranscriptState] Adding line [speaker {speaker_id}]: {text}")
        idx = bisect.bisect_right(self.line_timestamps, timestamp)
        self.line_timestamps.insert(idx, timestamp)
        self.lines.insert(idx, text)
        self.line_speakers.insert(idx, speaker_id)

        # Trigger incremental summary/entities every 5 lines
        if len(self.lines) % 5 == 0 and self.on_update_trigger:
            print(
                f"[TranscriptState] Triggering background updates at line {len(self.lines)}"
            )
            asyncio.create_task(self.on_update_trigger(self))

    def get_transcript_snapshot(self, add_speaker: bool = True) -> str:
        """
        Returns the full transcript as a formatted string with timestamps and speaker IDs.
        """
        formatted_lines = []
        for i, line in enumerate(self.lines):
            ts = self.format_timestamp(self.line_timestamps[i])
            speaker_id = self.line_speakers[i]
            if add_speaker and speaker_id >= 1:
                formatted_lines.append(f"{ts}(speaker {speaker_id}) {line}")
            else:
                formatted_lines.append(f"{ts} {line}")
        return "\n".join(formatted_lines)

    def format_timestamp(self, seconds: float) -> str:
        """
        Formats seconds into [MM:SS] format.
        """
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"[{mins:02d}:{secs:02d}]"

    def get_summary_snapshot(self) -> str:
        """
        Returns all summaries concatenated.
        """
        return "\n\n".join([f"Summary at {ts}:\n{text}" for ts, text in self.summaries])

    def get_entity_snapshot(self) -> str:
        """
        Returns all extracted entities concatenated.
        """
        return "\n\n".join([f"Entities at {ts}:\n{text}" for ts, text in self.entities])
