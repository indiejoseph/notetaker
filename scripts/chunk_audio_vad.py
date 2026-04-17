#!/usr/bin/env python3
"""
Chunk a WAV file using the project's Silero VAD and save speech segments.

Usage:
  python scripts/chunk_audio_vad.py /path/to/audio.wav --out tmp/chunks

This script mirrors the pipeline's framing (20ms, 16kHz) and uses
the same LiveKit VAD streaming API as `agents/transcriber.py`.
"""
import argparse
import asyncio
import os
import wave
from math import gcd

import numpy as np
from scipy.signal import resample_poly
from scipy.io import wavfile

from livekit import rtc
from livekit.agents import vad as vad_module
from livekit.plugins import silero


async def chunk_file(input_wav: str, out_dir: str, silence_duration: float = 0.35):
    os.makedirs(out_dir, exist_ok=True)

    sample_rate, audio_data = wavfile.read(input_wav)

    # Ensure mono
    if len(audio_data.shape) > 1:
        audio_data = audio_data.mean(axis=1)

    # Normalize to int16
    if np.issubdtype(audio_data.dtype, np.floating):
        max_val = np.max(np.abs(audio_data))
        if max_val > 1.0:
            audio_data = np.clip(audio_data, -32768, 32767).astype(np.int16)
        else:
            audio_data = (audio_data * 32767).astype(np.int16)
    elif audio_data.dtype != np.int16:
        audio_data = audio_data.astype(np.int16)

    # Resample to 16kHz if needed
    target_rate = 16000
    if sample_rate != target_rate:
        g = gcd(sample_rate, target_rate)
        audio_data = resample_poly(
            audio_data, target_rate // g, sample_rate // g
        ).astype(np.int16)
        sample_rate = target_rate

    samples_per_frame = int(sample_rate * 0.02)  # 20ms frames

    vad = silero.VAD.load(min_silence_duration=silence_duration)
    vad_stream = vad.stream()

    chunk_index = 0

    async def process_vad():
        nonlocal chunk_index
        async for event in vad_stream:
            if event.type == vad_module.VADEventType.END_OF_SPEECH and event.frames:
                # compute start and end timestamps
                start = max(
                    0.0,
                    event.timestamp - event.silence_duration - event.speech_duration,
                )
                end = start + event.speech_duration

                # concatenate frame bytes
                audio_bytes = bytearray()
                for f in event.frames:
                    audio_bytes.extend(f.data)

                out_name = f"chunk_{chunk_index:04d}_{start:.2f}-{end:.2f}.wav"
                out_path = os.path.join(out_dir, out_name)

                with wave.open(out_path, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(sample_rate)
                    wf.writeframes(bytes(audio_bytes))

                print(f"Wrote {out_path} ({start:.2f}s - {end:.2f}s)")
                chunk_index += 1

    # start VAD event processor
    proc_task = asyncio.create_task(process_vad())

    # push frames
    try:
        for i in range(0, len(audio_data), samples_per_frame):
            chunk = audio_data[i : i + samples_per_frame]
            if len(chunk) < samples_per_frame:
                continue
            frame = rtc.AudioFrame(
                data=chunk.tobytes(),
                sample_rate=sample_rate,
                num_channels=1,
                samples_per_channel=len(chunk),
            )
            vad_stream.push_frame(frame)
            await asyncio.sleep(0)

        # append trailing silence to flush
        silence_samples = int(silence_duration * sample_rate)
        vad_stream.push_frame(
            rtc.AudioFrame(
                data=bytes(silence_samples * 2),
                sample_rate=sample_rate,
                num_channels=1,
                samples_per_channel=silence_samples,
            )
        )
        await asyncio.sleep(0)
    finally:
        vad_stream.end_input()
        await proc_task


def main():
    parser = argparse.ArgumentParser(description="Chunk audio via Silero VAD")
    parser.add_argument("input", help="Path to input WAV file")
    parser.add_argument(
        "--out", default="tmp/chunks", help="Output directory for chunks"
    )
    parser.add_argument(
        "--silence",
        type=float,
        default=0.35,
        help="Min silence duration to end speech (s)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise SystemExit(f"Input file not found: {args.input}")

    asyncio.run(chunk_file(args.input, args.out, silence_duration=args.silence))


if __name__ == "__main__":
    main()
