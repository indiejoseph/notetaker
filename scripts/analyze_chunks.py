#!/usr/bin/env python3
"""
Analyze chunk embeddings to understand why certain chunks are misclassified.
"""
import argparse
import glob
import os
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly
from scipy.spatial.distance import cosine
from math import gcd

from agents.speaker_diarizer import SpeakerDiarizer


def extract_embedding(sd: SpeakerDiarizer, wav_path: str, target_sr: int = 16000) -> np.ndarray:
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

    if sr != target_sr and data.size > 0:
        g = gcd(sr, target_sr)
        up = target_sr // g
        down = sr // g
        data = resample_poly(data, up, down).astype(np.int16)

    audio_float = data.astype(np.float32) / 32768.0
    if audio_float.size == 0:
        raise RuntimeError(f"No audio in {wav_path}")

    import torch
    tensor = torch.from_numpy(audio_float).unsqueeze(0).to(torch.float32)
    with torch.no_grad():
        emb = sd.classifier.encode_batch(tensor).squeeze(0).cpu().numpy()
    emb = emb.flatten().astype(np.float32)
    n = np.linalg.norm(emb) + 1e-8
    return emb / n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--chunks", default="tmp/chunks", help="Directory with chunk WAV files")
    args = p.parse_args()

    files = sorted(glob.glob(os.path.join(args.chunks, "*.wav")))
    if not files:
        print(f"No WAV files found in {args.chunks}")
        return

    print(f"Extracting embeddings for {len(files)} chunks...")
    sd = SpeakerDiarizer()

    embeddings = []
    names = []
    for f in files:
        try:
            emb = extract_embedding(sd, f)
            embeddings.append(emb)
            names.append(os.path.basename(f))
        except Exception as e:
            print(f"Failed to extract from {f}: {e}")

    # Analyze pairwise distances
    print("\n=== Pairwise Cosine Distances ===")
    print("(distances < 0.3 suggest same speaker, > 0.6 suggest different speakers)\n")

    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            dist = cosine(embeddings[i], embeddings[j])
            print(f"{names[i]} <-> {names[j]}: {dist:.3f}")

    # Test SpeakerDiarizer behavior
    print("\n=== SpeakerDiarizer Sequential Assignment ===\n")
    diarizer = SpeakerDiarizer(max_speakers=0)

    for emb, name in zip(embeddings, names):
        # Simulate get_speaker_id behavior
        speaker_id, distance, distances = diarizer._match_speaker(emb, duration=1.0)
        dist_str = ", ".join([f"s{sid}:{d:.3f}" for sid, d in distances])
        print(f"{name}: speaker {speaker_id} (distance: {distance:.3f}) [{dist_str}]")

    # Ground truth from K-means
    print("\n=== Expected (from K-means) ===")
    expected = {
        "chunk_0000": 1,
        "chunk_0001": 2,
        "chunk_0002": 1,
        "chunk_0003": 1,
        "chunk_0004": 2,
        "chunk_0005": 1,
        "chunk_0006": 2,
        "chunk_0007": 2,
        "chunk_0008": 1,
        "chunk_0009": 1,
    }
    for name in names:
        chunk_name = name.split("_")[0] + "_" + name.split("_")[1]
        exp_speaker = expected.get(chunk_name, "?")
        print(f"{name}: speaker {exp_speaker}")


if __name__ == "__main__":
    main()
