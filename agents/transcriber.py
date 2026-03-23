import asyncio
import os
import wave
import tempfile
import base64
import numpy as np
from typing import AsyncIterator
from openai import AsyncOpenAI
from livekit import rtc
from livekit.agents import vad
from livekit.plugins import silero
from core.state import TranscriptState


class TranscriberAgent:
    """
    Transcriber Agent: Processes audio via Silero VAD, then transcribes each
    speech segment directly using the Qwen3-Omni Chat Completions API.
    Self-contained — no external STT adapter needed.
    """

    def __init__(self):
        self._vad = silero.VAD.load()
        self._client = AsyncOpenAI(
            api_key=os.environ.get("QWEN_API_KEY", "sk-xxx"),
            base_url=os.environ.get("QWEN_BASE_URL", "http://localhost/v1"),
        )
        self._model = os.environ.get("QWEN_MODEL", "qwen3-omni")

    async def run(
        self, audio_source: AsyncIterator[rtc.AudioFrame], state: TranscriptState
    ):
        """
        Consumes audio frames, runs VAD, and transcribes each speech segment.
        Updates TranscriptState with finalized lines.
        """
        vad_stream = self._vad.stream()
        pending_transcriptions: list[asyncio.Task] = []

        async def process_vad():
            async for event in vad_stream:
                if event.type == vad.VADEventType.END_OF_SPEECH and event.frames:
                    # event.timestamp = cumulative audio time at this inference window
                    # event.silence_duration = trailing silence that triggered end-of-speech
                    # event.speech_duration = duration of the speech segment itself
                    speech_start = max(
                        0.0,
                        event.timestamp
                        - event.silence_duration
                        - event.speech_duration,
                    )
                    print(
                        f"[TranscriberAgent] END_OF_SPEECH, speech started at {speech_start:.2f}s"
                    )

                    context_parts = []
                    if state.lines:
                        recent = "\n".join(state.lines[-5:])
                        context_parts.append(f"Recent conversation:\n{recent}")
                    if state.summaries:
                        context_parts.append(
                            f"Recent summary:\n{state.summaries[-1][1]}"
                        )
                    if state.entities:
                        # Merge all entity batches so even early batches contribute
                        all_entities = "\n---\n".join(text for _, text in state.entities)
                        context_parts.append(f"Known entities:\n{all_entities}")
                    prompt = "\n\n".join(context_parts)

                    task = asyncio.create_task(
                        self._transcribe_and_add(
                            event.frames, state, speech_start, prompt
                        )
                    )
                    pending_transcriptions.append(task)

        vad_task = asyncio.create_task(process_vad())

        last_sample_rate = 16000
        try:
            async for frame in audio_source:
                last_sample_rate = frame.sample_rate
                vad_stream.push_frame(frame)
                await asyncio.sleep(0)

            # Append silence to flush any speech that hasn't seen enough trailing silence.
            # Default min_silence_duration is 0.55s; we push 0.7s to be safe.
            silence_samples = int(0.7 * last_sample_rate)
            vad_stream.push_frame(rtc.AudioFrame(
                data=bytes(silence_samples * 2),
                sample_rate=last_sample_rate,
                num_channels=1,
                samples_per_channel=silence_samples,
            ))
            await asyncio.sleep(0)
        finally:
            vad_stream.end_input()
            await vad_task
            # Wait for all in-flight transcription API calls to complete
            if pending_transcriptions:
                await asyncio.gather(*pending_transcriptions, return_exceptions=True)

    async def _transcribe_and_add(
        self,
        frames: list[rtc.AudioFrame],
        state: TranscriptState,
        timestamp: float,
        prompt: str,
    ):
        """
        Converts speech frames to WAV, sends to Qwen3-Omni, and stores the result.
        """
        try:
            sample_rate = frames[0].sample_rate
            audio_data = bytearray()
            for frame in frames:
                audio_data.extend(np.frombuffer(frame.data, dtype=np.int16).tobytes())

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                tmp_path = tmp_file.name
                with wave.open(tmp_file, "wb") as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)
                    wav_file.setframerate(sample_rate)
                    wav_file.writeframes(audio_data)

            with open(tmp_path, "rb") as f:
                audio_b64 = base64.b64encode(f.read()).decode("utf-8")
            os.remove(tmp_path)

            messages: list[dict] = [
                {
                    "role": "system",
                    "content": (
                        "You are a professional transcriber. Your ONLY task is to transcribe "
                        "the input audio verbatim in the original spoken language. "
                        "Do NOT translate. Do NOT add any commentary. "
                        "Output ONLY the raw transcript exactly as spoken."
                    ),
                },
            ]
            if prompt:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"Context from prior transcription session:\n{prompt}\n\n"
                            "Use this context to improve accuracy. Now transcribe the audio below."
                        ),
                    }
                )
                messages.append(
                    {
                        "role": "assistant",
                        "content": "Understood. I will use this context for accurate transcription.",
                    }
                )
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio",
                            "input_audio": {"data": audio_b64, "format": "wav"},
                        }
                    ],
                }
            )

            print(
                f"[TranscriberAgent] Sending {len(audio_data)} bytes to Qwen at {timestamp:.2f}s"
            )
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=0.0,
            )

            text = response.choices[0].message.content
            if text and text.lower().strip() != "none" and text.strip():
                await state.add_line(text.strip(), timestamp)
                print(
                    f"[TranscriberAgent] [{state.format_timestamp(timestamp)}] {text.strip()}"
                )

        except Exception as e:
            print(f"[TranscriberAgent] Transcription error: {e}")
