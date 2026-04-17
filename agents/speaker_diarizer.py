import os
import numpy as np
import torch
from scipy.spatial.distance import cosine
from scipy.signal import resample_poly
from math import gcd


class SpeakerDiarizer:
    """
    Speaker diarization module using SpeechBrain's ECAPA-TDNN model.
    Extracts speaker embeddings and clusters them to identify unique speakers.
    """

    def __init__(
        self,
        model_source: str = "speechbrain/spkrec-ecapa-voxceleb",
        embedding_threshold: float = None,
        max_speakers: int = 0,
        min_chunk_duration: float = None,
    ):
        """
        Initialize the speaker diarizer.

        Args:
            model_source: Path or HuggingFace model identifier
            embedding_threshold: Cosine distance threshold (0-1) for matching speakers.
                - Lower values (0.3-0.5): Stricter matching, creates MORE separate speakers
                - Higher values (0.6-0.8): Lenient matching, groups MORE speakers together
                - Default: 0.5 or from SPEAKER_EMBEDDING_THRESHOLD env var
            max_speakers: Maximum number of speakers to detect (0 = unlimited).
                When the limit is reached, new segments are assigned to the closest
                existing speaker regardless of distance.
        """
        from speechbrain.pretrained import EncoderClassifier

        self.classifier = EncoderClassifier.from_hparams(
            source=model_source,
            savedir="./pretrained_models/spkrec-ecapa-voxceleb",
        )
        # speaker_embeddings stores [speaker_id, centroid_embedding, count, max_duration_seen]
        # centroid = embedding of the longest chunk from this speaker (most representative)
        self.speaker_embeddings: list[list] = []
        self.next_speaker_id = 1
        self.max_speakers = max_speakers
        # Minimum duration (seconds) for representative embedding. Can be
        # overridden via constructor or SPEAKER_MIN_CHUNK_DURATION env var.
        if min_chunk_duration is not None:
            self.min_chunk_duration = float(min_chunk_duration)
        else:
            self.min_chunk_duration = float(
                os.environ.get("SPEAKER_MIN_CHUNK_DURATION", "0.5")
            )

        # Default threshold tuned for ECAPA-TDNN discriminability. Lower is
        # stricter (more separate speakers). Can be overridden via param or
        # SPEAKER_EMBEDDING_THRESHOLD env var.
        if embedding_threshold is not None:
            self.embedding_threshold = float(embedding_threshold)
        else:
            self.embedding_threshold = float(
                os.environ.get("SPEAKER_EMBEDDING_THRESHOLD", "0.32")
            )

        print(
            f"[SpeakerDiarizer] Initialized with embedding_threshold={self.embedding_threshold}, max_speakers={self.max_speakers}"
        )

    async def get_speaker_id(self, audio_frames: list) -> tuple:
        """
        Extract speaker embedding from audio frames and assign speaker ID.

        Args:
            audio_frames: List of rtc.AudioFrame objects

        Returns:
            Tuple of (Speaker ID (int), distance (float))
        """
        try:
            # Convert frames to numpy int16 array (preserve raw bytes)
            if not audio_frames:
                print("[SpeakerDiarizer] No audio frames provided")
                return -1, float("inf"), []

            sample_rate = (
                audio_frames[0].sample_rate
                if hasattr(audio_frames[0], "sample_rate")
                else 16000
            )
            audio_bytes = bytearray()
            for frame in audio_frames:
                # frame.data is raw bytes; extend the buffer directly
                audio_bytes.extend(frame.data)

            if len(audio_bytes) == 0:
                print("[SpeakerDiarizer] Empty audio buffer")
                return -1, float("inf"), []

            # Interpret bytes as int16 PCM
            audio_int16 = np.frombuffer(bytes(audio_bytes), dtype=np.int16)

            # Resample to 16kHz if needed (ECAPA models typically expect 16kHz)
            target_sr = 16000
            if sample_rate != target_sr and audio_int16.size > 0:
                try:
                    g = gcd(sample_rate, target_sr)
                    up = target_sr // g
                    down = sample_rate // g
                    audio_int16 = resample_poly(audio_int16, up, down).astype(np.int16)
                    sample_rate = target_sr
                    print(
                        f"[SpeakerDiarizer] Resampled audio to {target_sr}Hz (length={len(audio_int16)})"
                    )
                except Exception as e:
                    print(f"[SpeakerDiarizer] Resampling failed: {e}")

            # Convert to float32 in [-1, 1]
            audio_float = audio_int16.astype(np.float32) / 32768.0
            if audio_float.size == 0:
                print("[SpeakerDiarizer] No audio after processing")
                return -1, float("inf"), []

            # Create tensor shape (batch, time)
            audio_tensor = torch.from_numpy(audio_float).unsqueeze(0).to(torch.float32)

            # Extract embedding
            with torch.no_grad():
                embedding = (
                    self.classifier.encode_batch(audio_tensor).squeeze(0).cpu().numpy()
                )

            # Flatten embedding
            embedding = embedding.flatten().astype(np.float32)

            # Calculate chunk duration
            duration = self._calculate_duration(audio_frames)

            # Simple energy check to avoid processing non-speech (very quiet) chunks
            try:
                rms = float(np.sqrt(np.mean(np.square(audio_float))))
            except Exception:
                rms = 0.0
            if rms < 1e-4:
                print(
                    f"[SpeakerDiarizer] Chunk too quiet (rms={rms:.6f}), skipping embedding"
                )
                return -1, float("inf"), []

            # Find matching speaker or create new one (also returns distance and per-speaker distances)
            speaker_id, distance, distances = self._match_speaker(embedding, duration)

            print(
                f"[SpeakerDiarizer] Assigned speaker {speaker_id} (duration: {duration:.2f}s, distance: {distance:.3f})"
            )
            return speaker_id, distance, distances

        except Exception as e:
            print(f"[SpeakerDiarizer] Error extracting speaker embedding: {e}")
            import traceback

            traceback.print_exc()
            return -1, float("inf"), []  # Error case

    def _calculate_duration(self, audio_frames: list) -> float:
        """Calculate total duration of audio frames in seconds."""
        if not audio_frames:
            return 0.0
        total_samples = 0
        sample_rate = (
            audio_frames[0].sample_rate
            if hasattr(audio_frames[0], "sample_rate")
            else 16000
        )
        for frame in audio_frames:
            total_samples += (
                frame.samples_per_channel
                if hasattr(frame, "samples_per_channel")
                else len(frame.data) // 2
            )
        return total_samples / sample_rate

    def _match_speaker(self, embedding: np.ndarray, duration: float = 0.0) -> tuple:
        """
        Match embedding to an existing speaker or create a new one.

        Args:
            embedding: Speaker embedding vector (1-D)
            duration: Duration of audio chunk in seconds

        Returns:
            Tuple of (Speaker ID, distance)
        """
        # Ensure embedding is 1-D and normalized
        embedding = embedding.flatten().astype(np.float32)
        norm = np.linalg.norm(embedding) + 1e-8
        if norm != 0.0:
            embedding = embedding / norm

        if not self.speaker_embeddings:
            # First speaker: store centroid (this embedding), count=1, max_duration
            self.speaker_embeddings.append(
                [self.next_speaker_id, embedding.copy(), 1, duration]
            )
            speaker_id = self.next_speaker_id
            self.next_speaker_id += 1
            print(
                f"[SpeakerDiarizer] Created speaker {speaker_id}, embedding shape: {embedding.shape}, duration: {duration:.2f}s"
            )
            return speaker_id, 0.0, []

        # Find closest speaker
        min_distance = float("inf")
        matched_speaker_id = None
        matched_idx = None

        distances = []
        for idx, (
            speaker_id,
            stored_centroid,
            stored_count,
            stored_max_duration,
        ) in enumerate(self.speaker_embeddings):
            try:
                distance = float(cosine(embedding, stored_centroid.flatten()))
                distances.append((speaker_id, float(distance)))
                if distance < min_distance:
                    min_distance = distance
                    matched_speaker_id = speaker_id
                    matched_idx = idx
            except Exception as e:
                print(f"[SpeakerDiarizer] Error computing cosine distance: {e}")
                continue

        # If close enough to existing speaker, or max speakers reached, assign to closest
        at_limit = (
            self.max_speakers > 0 and len(self.speaker_embeddings) >= self.max_speakers
        )
        if min_distance <= self.embedding_threshold or at_limit:
            # Update centroid: use the longest chunk embedding as the representative
            if matched_idx is not None:
                spk_id, stored_centroid, stored_count, stored_max_duration = (
                    self.speaker_embeddings[matched_idx]
                )
                new_count = stored_count + 1
                # Only update centroid if this chunk is longer (more representative)
                if duration > stored_max_duration:
                    new_centroid = embedding.copy()
                    new_max_duration = duration
                    print(
                        f"[SpeakerDiarizer] Updated centroid for speaker {spk_id} (count={new_count}, new longest chunk: {duration:.2f}s)"
                    )
                else:
                    new_centroid = stored_centroid
                    new_max_duration = stored_max_duration
                    print(
                        f"[SpeakerDiarizer] Matched existing speaker {spk_id} (count={new_count}, distance: {min_distance:.3f})"
                    )

                self.speaker_embeddings[matched_idx] = [
                    spk_id,
                    new_centroid,
                    new_count,
                    new_max_duration,
                ]

            if at_limit and min_distance > self.embedding_threshold:
                print(
                    f"[SpeakerDiarizer] Max speakers ({self.max_speakers}) reached, "
                    f"forcing match to speaker {matched_speaker_id} (distance: {min_distance:.3f})"
                )
            if distances:
                return matched_speaker_id, float(min_distance), distances
            else:
                return matched_speaker_id, float(min_distance), []

        # If the new chunk is too short to create a reliable new speaker, assign
        # to the closest speaker instead of creating a new one to avoid
        # proliferation of short-noise clusters.
        if duration < self.min_chunk_duration and self.speaker_embeddings:
            # Still update the centroid if this chunk is longer
            if matched_idx is not None:
                spk_id, stored_centroid, stored_count, stored_max_duration = (
                    self.speaker_embeddings[matched_idx]
                )
                new_count = stored_count + 1
                if duration > stored_max_duration:
                    new_centroid = embedding.copy()
                    new_max_duration = duration
                else:
                    new_centroid = stored_centroid
                    new_max_duration = stored_max_duration

                self.speaker_embeddings[matched_idx] = [
                    spk_id,
                    new_centroid,
                    new_count,
                    new_max_duration,
                ]

            print(
                f"[SpeakerDiarizer] Short chunk ({duration:.2f}s) - assigning to closest speaker {matched_speaker_id} (distance {min_distance:.3f})"
            )
            return matched_speaker_id, float(min_distance), distances

        # Create a new speaker
        self.speaker_embeddings.append(
            [self.next_speaker_id, embedding.copy(), 1, duration]
        )
        speaker_id = self.next_speaker_id
        self.next_speaker_id += 1
        print(
            f"[SpeakerDiarizer] Created new speaker {speaker_id} (closest distance was {min_distance:.3f}, duration: {duration:.2f}s)"
        )
        return speaker_id, float(min_distance), distances

    def reset(self):
        """Reset speaker tracking for a new session."""
        self.speaker_embeddings.clear()
        self.next_speaker_id = 1
