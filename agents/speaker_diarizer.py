import os
import numpy as np
import torch
from scipy.spatial.distance import cosine


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
        self.speaker_embeddings: list[tuple[int, np.ndarray]] = (
            []
        )  # [(speaker_id, embedding)]
        self.next_speaker_id = 1
        self.max_speakers = max_speakers

        # Get threshold from parameter, environment, or default
        if embedding_threshold is not None:
            self.embedding_threshold = embedding_threshold
        else:
            self.embedding_threshold = float(
                os.environ.get("SPEAKER_EMBEDDING_THRESHOLD", "0.5")
            )

        print(
            f"[SpeakerDiarizer] Initialized with embedding_threshold={self.embedding_threshold}, max_speakers={self.max_speakers}"
        )

    async def get_speaker_id(self, audio_frames: list) -> int:
        """
        Extract speaker embedding from audio frames and assign speaker ID.

        Args:
            audio_frames: List of rtc.AudioFrame objects

        Returns:
            Speaker ID (int)
        """
        try:
            # Convert frames to numpy array
            audio_data = bytearray()
            sample_rate = audio_frames[0].sample_rate if audio_frames else 16000

            for frame in audio_frames:
                audio_data.extend(np.frombuffer(frame.data, dtype=np.int16))

            # Convert to torch tensor
            audio_tensor = torch.from_numpy(
                np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
            )
            audio_tensor = audio_tensor.unsqueeze(0)  # Add batch dimension

            # Normalize audio
            audio_max = torch.max(torch.abs(audio_tensor))
            if audio_max > 0:
                audio_tensor = audio_tensor / audio_max
            else:
                audio_tensor = audio_tensor

            # Extract embedding
            with torch.no_grad():
                embedding = (
                    self.classifier.encode_batch(audio_tensor).squeeze(0).cpu().numpy()
                )

            # Flatten and normalize embedding
            embedding = embedding.flatten().astype(np.float32)
            embedding = embedding / (np.linalg.norm(embedding) + 1e-8)

            # Find matching speaker or create new one
            speaker_id = self._match_speaker(embedding)

            print(f"[SpeakerDiarizer] Assigned speaker {speaker_id}")
            return speaker_id

        except Exception as e:
            print(f"[SpeakerDiarizer] Error extracting speaker embedding: {e}")
            import traceback

            traceback.print_exc()
            return -1  # Error case

    def _match_speaker(self, embedding: np.ndarray) -> int:
        """
        Match embedding to an existing speaker or create a new one.

        Args:
            embedding: Speaker embedding vector (1-D)

        Returns:
            Speaker ID
        """
        # Ensure embedding is 1-D
        embedding = embedding.flatten().astype(np.float32)

        if not self.speaker_embeddings:
            # First speaker
            self.speaker_embeddings.append((self.next_speaker_id, embedding.copy()))
            speaker_id = self.next_speaker_id
            self.next_speaker_id += 1
            print(
                f"[SpeakerDiarizer] Created speaker {speaker_id}, embedding shape: {embedding.shape}"
            )
            return speaker_id

        # Find closest speaker
        min_distance = float("inf")
        matched_speaker_id = None

        for speaker_id, stored_embedding in self.speaker_embeddings:
            try:
                distance = cosine(embedding, stored_embedding.flatten())
                if distance < min_distance:
                    min_distance = distance
                    matched_speaker_id = speaker_id
            except Exception as e:
                print(f"[SpeakerDiarizer] Error computing cosine distance: {e}")
                continue

        # If close enough to existing speaker, or max speakers reached, assign to closest
        at_limit = self.max_speakers > 0 and len(self.speaker_embeddings) >= self.max_speakers
        if min_distance <= self.embedding_threshold or at_limit:
            # Update stored embedding with average (optional smoothing)
            for i, (sid, stored_emb) in enumerate(self.speaker_embeddings):
                if sid == matched_speaker_id:
                    updated_emb = (stored_emb + embedding) / 2.0
                    updated_emb = updated_emb / (np.linalg.norm(updated_emb) + 1e-8)
                    self.speaker_embeddings[i] = (sid, updated_emb)
                    break
            if at_limit and min_distance > self.embedding_threshold:
                print(
                    f"[SpeakerDiarizer] Max speakers ({self.max_speakers}) reached, "
                    f"forcing match to speaker {matched_speaker_id} (distance: {min_distance:.3f})"
                )
            else:
                print(
                    f"[SpeakerDiarizer] Matched existing speaker {matched_speaker_id} (distance: {min_distance:.3f})"
                )
            return matched_speaker_id

        # New speaker
        self.speaker_embeddings.append((self.next_speaker_id, embedding.copy()))
        speaker_id = self.next_speaker_id
        self.next_speaker_id += 1
        print(
            f"[SpeakerDiarizer] Created new speaker {speaker_id} (closest distance was {min_distance:.3f})"
        )
        return speaker_id

    def reset(self):
        """Reset speaker tracking for a new session."""
        self.speaker_embeddings.clear()
        self.next_speaker_id = 1
