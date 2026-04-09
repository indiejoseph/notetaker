"""
Diagnostic script: runs the same VAD + speaker diarization pipeline on a WAV file
and saves each speech segment as a separate WAV chunk for inspection.

Output files: tmp/chunks/spk{speaker_id}_{start_time}-{end_time}.wav

Usage:
    python dump_vad_chunks.py [path_to_wav] [max_speakers]
"""

import asyncio
import os
import sys
import wave
import numpy as np
import scipy.io.wavfile as wavfile
from scipy.signal import resample_poly
from math import gcd
from livekit import rtc
from livekit.agents import vad
from livekit.plugins import silero
from dotenv import load_dotenv

load_dotenv()


async def dump_chunks(wav_path: str, max_speakers: int = 0):
    from agents.speaker_diarizer import SpeakerDiarizer

    print(f"[DumpChunks] Loading: {wav_path}")
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

    # Resample to 16kHz
    target_rate = 16000
    if sample_rate != target_rate:
        print(f"[DumpChunks] Resampling from {sample_rate}Hz to {target_rate}Hz")
        g = gcd(sample_rate, target_rate)
        audio_data = resample_poly(audio_data, target_rate // g, sample_rate // g).astype(np.int16)
        sample_rate = target_rate

    total_duration = len(audio_data) / sample_rate
    print(f"[DumpChunks] Audio: {total_duration:.1f}s, {sample_rate}Hz, {len(audio_data)} samples")

    # Output dir
    out_dir = os.path.join("tmp", "chunks")
    os.makedirs(out_dir, exist_ok=True)
    # Clean previous chunks
    for f in os.listdir(out_dir):
        if f.endswith(".wav"):
            os.remove(os.path.join(out_dir, f))

    # Init VAD and diarizer (same as pipeline)
    vad_model = silero.VAD.load()
    diarizer = SpeakerDiarizer(max_speakers=max_speakers)

    vad_stream = vad_model.stream()
    chunks_saved = []

    async def process_vad():
        segment_idx = 0
        async for event in vad_stream:
            if event.type == vad.VADEventType.END_OF_SPEECH and event.frames:
                speech_start = max(
                    0.0,
                    event.timestamp - event.silence_duration - event.speech_duration,
                )
                speech_end = event.timestamp - event.silence_duration

                # Combine frames into audio
                chunk_data = bytearray()
                chunk_sr = event.frames[0].sample_rate
                for frame in event.frames:
                    chunk_data.extend(np.frombuffer(frame.data, dtype=np.int16).tobytes())

                # Get speaker ID
                speaker_id = await diarizer.get_speaker_id(event.frames)

                # Format timestamps for filename
                def fmt(s):
                    m, sec = int(s // 60), s % 60
                    return f"{m:02d}m{sec:05.2f}s"

                filename = f"spk{speaker_id}_{fmt(speech_start)}-{fmt(speech_end)}.wav"
                filepath = os.path.join(out_dir, filename)

                with wave.open(filepath, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(chunk_sr)
                    wf.writeframes(chunk_data)

                duration = len(chunk_data) / 2 / chunk_sr
                chunks_saved.append((speech_start, speech_end, speaker_id, duration, filename))
                segment_idx += 1
                print(f"  [{segment_idx}] {filename}  ({duration:.2f}s)")

    vad_task = asyncio.create_task(process_vad())

    # Feed frames to VAD (same 20ms chunking as pipeline)
    samples_per_frame = int(sample_rate * 0.02)
    for i in range(0, len(audio_data), samples_per_frame):
        chunk = audio_data[i : i + samples_per_frame]
        if len(chunk) < samples_per_frame:
            continue
        vad_stream.push_frame(
            rtc.AudioFrame(
                data=chunk.tobytes(),
                sample_rate=sample_rate,
                num_channels=1,
                samples_per_channel=len(chunk),
            )
        )
        await asyncio.sleep(0)

    # Flush trailing speech
    silence_samples = int(0.7 * sample_rate)
    vad_stream.push_frame(
        rtc.AudioFrame(
            data=bytes(silence_samples * 2),
            sample_rate=sample_rate,
            num_channels=1,
            samples_per_channel=silence_samples,
        )
    )
    await asyncio.sleep(0)

    vad_stream.end_input()
    await vad_task

    # Summary
    print(f"\n{'='*60}")
    print(f"Total chunks: {len(chunks_saved)}")
    print(f"Output dir:   {os.path.abspath(out_dir)}")
    total_speech = sum(c[3] for c in chunks_saved)
    print(f"Total speech:  {total_speech:.1f}s / {total_duration:.1f}s ({total_speech/total_duration*100:.0f}%)")
    speakers = set(c[2] for c in chunks_saved)
    print(f"Speakers detected: {sorted(speakers)}")

    # Check for gaps
    print(f"\n{'='*60}")
    print("Timeline (gaps > 1s marked):")
    chunks_saved.sort(key=lambda c: c[0])
    for i, (start, end, spk, dur, fname) in enumerate(chunks_saved):
        gap = ""
        if i > 0:
            gap_s = start - chunks_saved[i - 1][1]
            if gap_s > 1.0:
                gap = f"  *** GAP {gap_s:.1f}s ***"
        print(f"  {start:7.2f}s - {end:7.2f}s  spk{spk}  {dur:.2f}s  {fname}{gap}")


if __name__ == "__main__":
    wav_file = sys.argv[1] if len(sys.argv) > 1 else "tmp/ManuLife_client_supplied_Sample_by_AI_CallCenter.wav"
    max_spk = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    asyncio.run(dump_chunks(wav_file, max_spk))
