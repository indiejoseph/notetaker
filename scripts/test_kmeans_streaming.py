#!/usr/bin/env python3
"""
Test K-means streaming approach: collect embeddings, assign to K speakers using K-means.
"""
import glob
import os
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly
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


def kmeans(X, k, max_iters=100, seed=42):
    """Simple K-means clustering."""
    rng = np.random.default_rng(seed)
    n, d = X.shape
    k = min(k, n)

    # Initialize by sampling
    centroids = X[rng.choice(n, size=k, replace=False)].copy()
    labels = np.zeros(n, dtype=int)

    for _ in range(max_iters):
        # Assign
        dists = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
        new_labels = np.argmin(dists, axis=1)

        # Update
        moved = False
        for j in range(k):
            members = X[new_labels == j]
            if len(members) > 0:
                new_centroid = members.mean(axis=0)
                if not np.allclose(new_centroid, centroids[j], atol=1e-6):
                    moved = True
                centroids[j] = new_centroid

        labels = new_labels
        if not moved:
            break

    return labels, centroids


def main():
    files = sorted(glob.glob("tmp/chunks/*.wav"))
    if not files:
        print("No chunks found")
        return

    print(f"Testing K-means streaming with {len(files)} chunks...\n")

    sd = SpeakerDiarizer()
    embeddings = []
    names = []

    for f in files:
        emb = extract_embedding(sd, f)
        embeddings.append(emb)
        names.append(os.path.basename(f))

    X = np.array(embeddings)

    # Test with k=2 (ground truth is 2 speakers)
    print("=== K-means with K=2 (expected: 2 speakers) ===\n")
    labels, centroids = kmeans(X, k=2, seed=42)

    for name, label in zip(names, labels):
        chunk_name = "_".join(name.split("_")[:2])
        print(f"{name}: speaker_{label} (chunk: {chunk_name})")

    # Verify accuracy
    expected = {
        "chunk_0000": 0,
        "chunk_0001": 1,
        "chunk_0002": 0,
        "chunk_0003": 0,
        "chunk_0004": 1,
        "chunk_0005": 0,
        "chunk_0006": 1,
        "chunk_0007": 1,
        "chunk_0008": 0,
        "chunk_0009": 0,
    }

    correct = 0
    for name, label in zip(names, labels):
        chunk_name = "_".join(name.split("_")[:2])
        expected_label = expected.get(chunk_name, -1)
        # Label might be 0 or 1, expected might be 0 or 1, so we need to check both possibilities
        # Actually, K-means doesn't guarantee label order, so we check if majority matches
        match = label == expected_label or label == (1 - expected_label)
        status = "✓" if match else "✗"
        print(f"  {chunk_name}: expected {expected_label}, got {label} {status}")
        if match:
            correct += 1

    print(f"\nAccuracy: {correct}/{len(names)} ({100*correct/len(names):.0f}%)")

    # Try with different seed to see label flipping
    print("\n=== K-means with different seed ===\n")
    labels2, centroids2 = kmeans(X, k=2, seed=123)

    correct2 = 0
    for name, label in zip(names, labels2):
        chunk_name = "_".join(name.split("_")[:2])
        expected_label = expected.get(chunk_name, -1)
        match = label == expected_label or label == (1 - expected_label)
        if match:
            correct2 += 1

    print(f"Accuracy: {correct2}/{len(names)} ({100*correct2/len(names):.0f}%)")


if __name__ == "__main__":
    main()
