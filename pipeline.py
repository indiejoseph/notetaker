import asyncio
import numpy as np
import scipy.io.wavfile as wavfile
from livekit import rtc
from agents.transcriber import TranscriberAgent
from agents.summarizer import SummarizerAgent
from agents.entities_extractor import EntityExtractorAgent
from agents.refiner import TranscriptRefinerAgent
from core.state import TranscriptState

class NotetakingPipeline:
    """
    Orchestrates the Transcriber, Summarizer, Entity Extractor, and Refiner agents.
    """

    def __init__(self):
        print("[Pipeline] Initializing...")
        self.state = TranscriptState()
        self.transcriber = TranscriberAgent()
        self.summarizer = SummarizerAgent()
        self.extractor = EntityExtractorAgent()
        self.refiner = TranscriptRefinerAgent()

        # Connect state trigger to background workers
        self.state.set_update_trigger(self._trigger_background_tasks)

    async def _trigger_background_tasks(self, state: TranscriptState):
        """
        Runs summarizer and extractor concurrently in the background.
        """
        print("[Pipeline] Triggering background tasks...")
        # Create tasks but don't await them here to avoid blocking transcription
        asyncio.create_task(self.summarizer.run(state))
        asyncio.create_task(self.extractor.run(state))

    async def process_file(self, wav_path: str):
        """
        Processes an offline WAV file through the live pipeline.
        Useful for verification and file-upload features.
        """
        if not wav_path.endswith(".wav"):
            raise ValueError("Only .wav files are supported")

        print(f"[Pipeline] Processing file: {wav_path}")
        sample_rate, audio_data = wavfile.read(wav_path)

        # Ensure mono
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)

        # Normalize to int16 if needed
        if audio_data.dtype != np.int16:
            audio_data = (audio_data * 32767).astype(np.int16)

        # Chunk audio into 20ms frames (LiveKit standard)
        samples_per_frame = int(sample_rate * 0.02)
        
        async def frame_gen():
            for i in range(0, len(audio_data), samples_per_frame):
                chunk = audio_data[i : i + samples_per_frame]
                if len(chunk) < samples_per_frame:
                    continue
                yield rtc.AudioFrame(
                    data=chunk.tobytes(),
                    sample_rate=sample_rate,
                    num_channels=1,
                    samples_per_channel=len(chunk),
                )

        # Run transcriber
        await self.transcriber.run(frame_gen(), self.state)

        # Wait for summarizer and extractor to finish all pending batches
        while (
            len(self.state.lines) > self.state.summary_processed
            or len(self.state.lines) > self.state.entities_processed
        ):
            print("[Pipeline] Waiting for background tasks to catch up...")
            await asyncio.sleep(0.5)

        # Run final refinement pass now that summaries and entities are complete
        await self.refiner.run(self.state)

        # Generate consolidated final summary over the refined transcript
        await self.summarizer.finalize(self.state)

        print("[Pipeline] Finished processing file.")
