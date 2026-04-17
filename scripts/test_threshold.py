#!/usr/bin/env python3
"""
Test different embedding thresholds to find the optimal one.
"""
import glob
import os
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly
from scipy.spatial.distance import cosine
from math import gcd
import torch

from agents.speaker_diarizer import SpeakerDiarizer


def extract_embedding(sd: SpeakerDiarizer, wav_path: str) -> np.ndarray:
    """Extract embedding for a single chunk."""
    sr, data = wavfile.read(wav_path)
    if len(data.shape) > 1:
        data = data.mean(axis=1)

    if np.issubdtype(data.dtype, np.floating):
        max_val = np.max(np.abs(data))
        if max_val > 1.0:
            data = np.clip(data, -32768, 32767).astype(np.int16)
        else:
            data = (data * 32767).astype(np.int16)
    elif data.dtype != np.int16:
        data = data.astype(np.int16)

    if sr != 16000 and data.size > 0:
        g = gcd(sr, 16000)
        up = 16000 // g
        down = sr // g
        data = resample_poly(data, up, down).astype(np.int16)

    audio_float = data.astype(np.float32) / 32768.0
    tensor = torch.from_numpy(audio_float).unsqueeze(0).to(torch.float32)
    with torch.no_grad():
        emb = sd.classifier.encode_batch(tensor).squeeze(0).cpu().numpy()
    emb = emb.flatten().astype(np.float32)
    n = np.linalg.norm(emb) + 1e-8
    return emb / n


def test_threshold(threshold: float):
    """Test diarizer with given threshold."""
    files = sorted(glob.glob("tmp/chunks/*.wav"))
    names = [os.path.basename(f) for f in files]

    sd = SpeakerDiarizer(embedding_threshold=threshold, max_speakers=2)

    embeddings = []
    for f in files:
        emb = extract_embedding(sd, f)
        embeddings.append(emb)

    # Simulate sequential assignment
    results = {}
    diarizer = SpeakerDiarizer(embedding_threshold=threshold, max_speakers=2)

    for name, emb in zip(names, embeddings):
        speaker_id, distance, _ = diarizer._match_speaker(emb, duration=2.0)
        results[name] = speaker_id

    # Compare with expected
    expected = {
        "chunk_0000_0.70-2.50.wav": 1,
        "chunk_0001_2.94-3.74.wav": 2,
        "chunk_0002_4.58-9.34.wav": 1,
        "chunk_0003_9.79-16.48.wav": 1,
        "chunk_0004_17.22-18.69.wav": 2,
        "chunk_0005_20.06-20.74.wav": 1,
        "chunk_0006_21.47-26.85.wav": 2,
        "chunk_0007_27.78-29.28.wav": 2,
        "chunk_0008_29.76-30.14.wav": 1,
        "chunk_0009_30.98-31.74.wav": 1,
    }

    correct = sum(1 for name, speaker_id in results.items() if speaker_id == expected[name])
    total = len(expected)

    return correct, total, results


def main():
    thresholds = [0.25, 0.30, 0.32, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70]

    print("Testing different thresholds:\n")
    print("Threshold | Correct | Result")
    print("-" * 70)

    best_threshold = None
    best_score = 0

    for threshold in thresholds:
        correct, total, results = test_threshold(threshold)
        score = correct / total
        status = "✓ BEST" if score > best_score else ""
        if score > best_score:
            best_threshold = threshold
            best_score = score

        print(f"{threshold:.2f}      | {correct:2d}/{total} ({100*score:.0f}%) | {status}")

    print(f"\n=== Best threshold: {best_threshold} ({best_score*100:.0f}%) ===")


if __name__ == "__main__":
    main()
