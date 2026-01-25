"""Voice Activity Detection using WebRTC VAD."""

import numpy as np
import webrtcvad


class WebRTCVAD:
    """
    Wrapper for WebRTC VAD with streaming speech boundary detection.

    Detects speech start/end events based on consecutive speech/silence frames.
    """

    def __init__(self, sample_rate=16000, aggressiveness=2):
        """
        Initialize VAD.

        Args:
            sample_rate: Audio sample rate (must be 8000, 16000, 32000, or 48000)
            aggressiveness: VAD aggressiveness (0-3, higher = more aggressive filtering)
        """
        self.sample_rate = sample_rate
        self.vad = webrtcvad.Vad(aggressiveness)

        # WebRTC VAD requires 10, 20, or 30ms frames
        # At 16kHz: 10ms=160, 20ms=320, 30ms=480 samples
        self.frame_duration_ms = 30
        self.frame_size = int(sample_rate * self.frame_duration_ms / 1000)

        # State for speech boundary detection
        self.buffer = np.array([], dtype=np.float32)
        self.speech_active = False
        self.speech_frames = 0
        self.silence_frames = 0

        # Thresholds for speech start/end detection
        self.speech_start_frames = 3  # ~90ms of speech to trigger start
        self.silence_end_frames = 10  # ~300ms of silence to trigger end

    def start(self):
        """Initialize VAD for a new session."""
        self.buffer = np.array([], dtype=np.float32)
        self.speech_active = False
        self.speech_frames = 0
        self.silence_frames = 0

    def process_chunk(self, audio_chunk):
        """
        Process an audio chunk through VAD.

        Args:
            audio_chunk: numpy array of float32 audio samples

        Returns:
            List of events: {'start': True} or {'end': True}
        """
        # Flatten if needed (sounddevice gives (frames, channels))
        if audio_chunk.ndim > 1:
            audio_chunk = audio_chunk.flatten()

        # Add to buffer
        self.buffer = np.concatenate([self.buffer, audio_chunk])

        events = []

        # Process in frame_size pieces
        while len(self.buffer) >= self.frame_size:
            frame = self.buffer[:self.frame_size]
            self.buffer = self.buffer[self.frame_size:]

            # Convert to 16-bit PCM for webrtcvad
            frame_int16 = (frame * 32767).astype(np.int16).tobytes()

            try:
                is_speech = self.vad.is_speech(frame_int16, self.sample_rate)
            except Exception:
                # Skip invalid frames
                continue

            if is_speech:
                self.speech_frames += 1
                self.silence_frames = 0

                # Trigger speech start after threshold
                if not self.speech_active and self.speech_frames >= self.speech_start_frames:
                    self.speech_active = True
                    events.append({'start': True})

            else:
                self.silence_frames += 1
                self.speech_frames = 0

                # Trigger speech end after silence threshold
                if self.speech_active and self.silence_frames >= self.silence_end_frames:
                    self.speech_active = False
                    events.append({'end': True})

        return events

    def stop(self):
        """Clean up VAD session."""
        self.buffer = np.array([], dtype=np.float32)
        self.speech_active = False


# Alias for compatibility
SileroVAD = WebRTCVAD
