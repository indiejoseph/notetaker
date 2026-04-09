import asyncio
import numpy as np
import scipy.io.wavfile as wavfile
from scipy.signal import resample_poly
from math import gcd
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

    def __init__(self, max_speakers: int = 0):
        print("[Pipeline] Initializing...")
        self.state = TranscriptState()
        self.transcriber = TranscriberAgent(max_speakers=max_speakers)
        self.summarizer = SummarizerAgent()
        self.extractor = EntityExtractorAgent()
        self.refiner = TranscriptRefinerAgent()

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

        # Normalize to int16
        if np.issubdtype(audio_data.dtype, np.floating):
            max_val = np.max(np.abs(audio_data))
            if max_val > 1.0:
                # Float values in int16 range (e.g. from .mean() on int16 data)
                audio_data = np.clip(audio_data, -32768, 32767).astype(np.int16)
            else:
                # Float values in [-1, 1] range (e.g. from float WAV)
                audio_data = (audio_data * 32767).astype(np.int16)
        elif audio_data.dtype != np.int16:
            audio_data = audio_data.astype(np.int16)

        # Resample to 16kHz if needed (Silero VAD and Qwen ASR expect 16kHz)
        target_rate = 16000
        if sample_rate != target_rate:
            print(f"[Pipeline] Resampling from {sample_rate}Hz to {target_rate}Hz")
            g = gcd(sample_rate, target_rate)
            audio_data = resample_poly(audio_data, target_rate // g, sample_rate // g).astype(np.int16)
            sample_rate = target_rate

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

        # 1. Transcriber
        await self.transcriber.run(frame_gen(), self.state)

        # 2. Refiner
        await self.refiner.run(self.state)

        # 3. Summarizer
        await self.summarizer.run(self.state)
        await self.summarizer.finalize(self.state)

        # 4. Extractor
        await self.extractor.run(self.state)

        print("[Pipeline] Finished processing file.")
