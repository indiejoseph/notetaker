#!/usr/bin/env python3
"""
Run K-means clustering on existing chunk WAV files to evaluate speaker clustering.

Usage:
  python scripts/kmeans_diarize.py --chunks tmp/chunks --max-speakers 2 --out results.csv

The script loads the SpeechBrain ECAPA model via `agents.speaker_diarizer.SpeakerDiarizer`
to extract embeddings for each chunk, then clusters them with K-means (NumPy implementation).
"""
import argparse
import glob
import os
import csv
from math import gcd

import numpy as np
import torch
from scipy.io import wavfile
from scipy.signal import resample_poly

from agents.speaker_diarizer import SpeakerDiarizer


def extract_embedding(
    sd: SpeakerDiarizer, wav_path: str, target_sr: int = 16000
) -> np.ndarray:
    sr, data = wavfile.read(wav_path)
    # mono
    if len(data.shape) > 1:
        data = data.mean(axis=1)

    # normalize to int16
    if np.issubdtype(data.dtype, np.floating):
        max_val = np.max(np.abs(data))
        if max_val > 1.0:
            data = np.clip(data, -32768, 32767).astype(np.int16)
        else:
            data = (data * 32767).astype(np.int16)
    elif data.dtype != np.int16:
        data = data.astype(np.int16)

    # resample if needed
    if sr != target_sr and data.size > 0:
        g = gcd(sr, target_sr)
        up = target_sr // g
        down = sr // g
        data = resample_poly(data, up, down).astype(np.int16)
        sr = target_sr

    audio_float = data.astype(np.float32) / 32768.0
    if audio_float.size == 0:
        raise RuntimeError(f"No audio in {wav_path}")

    # shape (batch, time)
    tensor = torch.from_numpy(audio_float).unsqueeze(0).to(torch.float32)
    with torch.no_grad():
        emb = sd.classifier.encode_batch(tensor).squeeze(0).cpu().numpy()
    emb = emb.flatten().astype(np.float32)
    # normalize
    n = np.linalg.norm(emb) + 1e-8
    return emb / n


def kmeans_numpy(X: np.ndarray, k: int, max_iters: int = 200, seed: int = 42):
    rng = np.random.default_rng(seed)
    n, d = X.shape
    if n == 0:
        return np.array([], dtype=int), np.zeros((0, d))
    if k <= 0:
        raise ValueError("k must be > 0")
    k = min(k, n)
    # initialize centroids by sampling points
    centroids = X[rng.choice(n, size=k, replace=False)].copy()
    labels = np.zeros(n, dtype=int)
    for it in range(max_iters):
        # assign
        dists = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
        new_labels = np.argmin(dists, axis=1)
        # recompute
        moved = False
        for j in range(k):
            members = X[new_labels == j]
            if len(members) == 0:
                # reinit empty centroid
                centroids[j] = X[rng.integers(0, n)]
            else:
                new_cent = members.mean(axis=0)
                if not np.allclose(new_cent, centroids[j], atol=1e-6):
                    moved = True
                centroids[j] = new_cent
        labels = new_labels
        if not moved:
            break
    return labels, centroids


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--chunks", default="tmp/chunks", help="Directory with chunk WAV files"
    )
    p.add_argument("--max-speakers", type=int, default=2, help="K for K-means")
    p.add_argument("--out", default=None, help="CSV output path (optional)")
    args = p.parse_args()

    files = sorted(glob.glob(os.path.join(args.chunks, "*.wav")))
    if not files:
        print(f"No WAV files found in {args.chunks}")
        return

    print(f"Loading speaker model and extracting embeddings for {len(files)} chunks...")
    sd = SpeakerDiarizer(max_speakers=0)

    embeddings = []
    names = []
    for f in files:
        try:
            emb = extract_embedding(sd, f)
            embeddings.append(emb)
            names.append(os.path.basename(f))
            print(f"- extracted {f}")
        except Exception as e:
            print(f"Failed to extract from {f}: {e}")

    X = np.vstack(embeddings)
    labels, centroids = kmeans_numpy(X, args.max_speakers)

    print("K-means results:")
    for name, lbl in zip(names, labels):
        print(f"{name}: speaker_{lbl}")

    if args.out:
        with open(args.out, "w", newline="") as csvfile:
            w = csv.writer(csvfile)
            w.writerow(["chunk", "speaker_label"])
            for name, lbl in zip(names, labels):
                w.writerow([name, int(lbl)])
        print(f"Wrote CSV to {args.out}")


if __name__ == "__main__":
    main()
