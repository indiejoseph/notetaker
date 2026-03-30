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
from agents.speaker_diarizer import SpeakerDiarizer


class TranscriberAgent:
    """
    Transcriber Agent: Processes audio via Silero VAD, then transcribes each
    speech segment directly using the Qwen3-Omni Chat Completions API.
    Self-contained — no external STT adapter needed.
    """

    def __init__(self):
        self._vad = silero.VAD.load()
        self._model = os.environ.get("QWEN_OMNI_MODEL", "qwen3-omni")
        # Optional dedicated ASR endpoint (preferred if configured)
        self._asr_base = os.environ.get("QWEN_ASR_URL")
        self._asr_api_key = os.environ.get("QWEN_ASR_API_KEY")
        self._asr_model = os.environ.get("QWEN_ASR_MODEL")
        self._client = AsyncOpenAI(
            api_key=self._asr_api_key,
            base_url=self._asr_base,
        )
        # Initialize speaker diarizer (PyTorch version)
        self._speaker_diarizer = SpeakerDiarizer()

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
                        all_entities = "\n---\n".join(
                            text for _, text in state.entities
                        )
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
            vad_stream.push_frame(
                rtc.AudioFrame(
                    data=bytes(silence_samples * 2),
                    sample_rate=last_sample_rate,
                    num_channels=1,
                    samples_per_channel=silence_samples,
                )
            )
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
        Converts speech frames to WAV, extracts speaker ID, sends to Qwen3-Omni, and stores the result.
        """
        try:
            sample_rate = frames[0].sample_rate
            audio_data = bytearray()
            for frame in frames:
                audio_data.extend(np.frombuffer(frame.data, dtype=np.int16).tobytes())

            # Extract speaker embedding
            speaker_id = await self._speaker_diarizer.get_speaker_id(frames)

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                tmp_path = tmp_file.name
                with wave.open(tmp_file, "wb") as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)
                    wav_file.setframerate(sample_rate)
                    wav_file.writeframes(audio_data)

            print(
                f"[TranscriberAgent] Sending {len(audio_data)} bytes to ASR endpoint at {timestamp:.2f}s (speaker {speaker_id})"
            )
            with open(tmp_path, "rb") as f:
                audio_b64 = base64.b64encode(f.read()).decode("utf-8")
            os.remove(tmp_path)

            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "audio_url",
                            "audio_url": {"url": f"data:audio/wav;base64,{audio_b64}"},
                        }
                    ],
                }
            ]

            messages: list[dict] = []

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
                    "role": "user",
                    "content": [
                        {
                            "type": "audio_url",
                            "audio_url": {"url": f"data:audio/wav;base64,{audio_b64}"},
                        }
                    ],
                }
            )

            try:
                response = await self._client.chat.completions.create(
                    model=self._asr_model, messages=messages
                )
            except Exception as e:
                print(f"[TranscriberAgent] ASR chat request failed: {e}")
                response = None

            text = None
            if response is not None:
                try:
                    text = response.choices[0].message.content
                    text = text.strip().split("<asr_text>")[-1]
                except Exception:
                    # fallback for dict-shaped response
                    if isinstance(response, dict):
                        # try common keys
                        text = response.get("text") or response.get("data")
            if text and text.lower().strip() != "none" and text.strip():
                await state.add_line(text.strip(), timestamp, speaker_id)
                print(
                    f"[TranscriberAgent] [{state.format_timestamp(timestamp)}](speaker {speaker_id}) {text.strip()}"
                )

        except Exception as e:
            print(f"[TranscriberAgent] Transcription error: {e}")
