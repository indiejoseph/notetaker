#!/usr/bin/env python3
"""
Plot 2D scatter of speaker embeddings for chunk WAV files.

Usage:
  python scripts/plot_embeddings.py --chunks tmp/chunks --k 2 --out tmp/embeddings.png

The script extracts embeddings using `agents.speaker_diarizer.SpeakerDiarizer`,
reduces them to 2D with PCA, runs K-means, and plots a colored scatter.
"""
import argparse
import glob
import os
from math import gcd

import numpy as np
import torch
from scipy.io import wavfile
from scipy.signal import resample_poly
import matplotlib.pyplot as plt

from agents.speaker_diarizer import SpeakerDiarizer


def extract_embedding(
    sd: SpeakerDiarizer, wav_path: str, target_sr: int = 16000
) -> np.ndarray:
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
        sr = target_sr
    audio_float = data.astype(np.float32) / 32768.0
    tensor = torch.from_numpy(audio_float).unsqueeze(0).to(torch.float32)
    with torch.no_grad():
        emb = sd.classifier.encode_batch(tensor).squeeze(0).cpu().numpy()
    emb = emb.flatten().astype(np.float32)
    n = np.linalg.norm(emb) + 1e-8
    return emb / n


def pca_2d(X: np.ndarray):
    # center
    Xc = X - X.mean(axis=0)
    # SVD
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    coords = U[:, :2] * S[:2]
    return coords


def kmeans_numpy(X: np.ndarray, k: int, max_iters: int = 200, seed: int = 42):
    rng = np.random.default_rng(seed)
    n, d = X.shape
    if n == 0:
        return np.array([], dtype=int), np.zeros((0, d))
    k = min(k, n)
    centroids = X[rng.choice(n, size=k, replace=False)].copy()
    labels = np.zeros(n, dtype=int)
    for it in range(max_iters):
        dists = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
        new_labels = np.argmin(dists, axis=1)
        moved = False
        for j in range(k):
            members = X[new_labels == j]
            if len(members) == 0:
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
    p.add_argument("--k", type=int, default=2, help="K for K-means clustering")
    p.add_argument("--out", default="tmp/embeddings.png", help="Output image path")
    p.add_argument("--show", action="store_true", help="Show plot interactively")
    args = p.parse_args()

    files = sorted(glob.glob(os.path.join(args.chunks, "*.wav")))
    if not files:
        print(f"No WAV files found in {args.chunks}")
        return

    sd = SpeakerDiarizer(max_speakers=0)
    embeddings = []
    names = []
    for f in files:
        try:
            emb = extract_embedding(sd, f)
            embeddings.append(emb)
            names.append(os.path.basename(f))
        except Exception as e:
            print(f"Failed to extract embedding from {f}: {e}")

    X = np.vstack(embeddings)
    labels, _ = kmeans_numpy(X, args.k)
    coords = pca_2d(X)

    plt.figure(figsize=(8, 6))
    cmap = plt.get_cmap("tab10")
    for lbl in range(max(1, args.k)):
        idx = labels == lbl
        if idx.sum() == 0:
            continue
        plt.scatter(
            coords[idx, 0],
            coords[idx, 1],
            label=f"speaker_{lbl}",
            s=80,
            alpha=0.8,
            color=cmap(lbl),
        )
    # annotate
    for i, name in enumerate(names):
        plt.text(coords[i, 0], coords[i, 1], name, fontsize=8, alpha=0.8)

    plt.title(f"Embeddings (K={args.k})")
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.savefig(args.out, dpi=150)
    print(f"Saved plot to {args.out}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
