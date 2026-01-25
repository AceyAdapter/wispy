"""Streaming transcription with VAD-triggered segments."""

import tempfile
import os
import threading
from queue import Queue, Empty
from dataclasses import dataclass
from typing import List, Optional, Callable

import numpy as np
from scipy.io import wavfile
import mlx_whisper

from vad import SileroVAD

SAMPLE_RATE = 16000


@dataclass
class TranscriptionSegment:
    """A transcribed speech segment."""
    text: str
    start_sample: int
    end_sample: int


class StreamingTranscriber:
    """
    Streaming transcription using VAD to detect speech boundaries.

    Audio is buffered and when VAD detects end of speech, the segment
    is queued for background transcription. Final text is accumulated
    and returned when stop() is called.
    """

    def __init__(self, model_repo: str, on_status: Optional[Callable[[str], None]] = None,
                 on_segment: Optional[Callable[[str], None]] = None):
        """
        Initialize streaming transcriber.

        Args:
            model_repo: HuggingFace repo for MLX Whisper model
            on_status: Optional callback for status updates
            on_segment: Optional callback when a segment is transcribed (for real-time pasting)
        """
        self.model_repo = model_repo
        self.on_status = on_status or (lambda s: None)
        self.on_segment = on_segment or (lambda s: None)

        self.vad = SileroVAD(sample_rate=SAMPLE_RATE)
        self.segments: List[TranscriptionSegment] = []

        # Audio buffering
        self.current_audio: List[np.ndarray] = []
        self.current_audio_samples = 0
        self.speech_active = False
        self.total_samples = 0

        # Max segment duration (transcribe even without pause)
        self.max_segment_seconds = 3
        self.max_segment_samples = int(SAMPLE_RATE * self.max_segment_seconds)

        # Overlap for context (prepend to each chunk)
        self.overlap_seconds = 1
        self.overlap_samples = int(SAMPLE_RATE * self.overlap_seconds)
        self.overlap_buffer: Optional[np.ndarray] = None
        self.last_transcription = ""

        # Background transcription
        self.transcription_queue: Queue = Queue()
        self.transcription_thread: Optional[threading.Thread] = None
        self.running = False
        self.lock = threading.Lock()

    def start(self):
        """Start the streaming transcriber."""
        self.vad.start()
        self.segments = []
        self.current_audio = []
        self.current_audio_samples = 0
        self.speech_active = False
        self.total_samples = 0
        self.running = True
        self.overlap_buffer = None
        self.last_transcription = ""

        # Start background transcription thread
        self.transcription_thread = threading.Thread(
            target=self._transcription_worker,
            daemon=True
        )
        self.transcription_thread.start()

    def process_audio(self, audio_chunk: np.ndarray):
        """
        Process incoming audio chunk.

        Args:
            audio_chunk: numpy array from sounddevice callback
        """
        if not self.running:
            return

        # Always buffer audio during speech
        chunk_samples = len(audio_chunk.flatten())
        if self.speech_active:
            self.current_audio.append(audio_chunk.copy())
            self.current_audio_samples += chunk_samples

            # Check if we've hit max segment duration
            if self.current_audio_samples >= self.max_segment_samples:
                self._queue_current_segment()

        # Process through VAD
        events = self.vad.process_chunk(audio_chunk)

        for event in events:
            if 'start' in event:
                # Speech started
                self.speech_active = True
                self.current_audio = [audio_chunk.copy()]
                self.current_audio_samples = chunk_samples
                self.on_status("Speech detected...")

            elif 'end' in event:
                # Speech ended - queue segment for transcription
                self._queue_current_segment()
                self.speech_active = False

        self.total_samples += chunk_samples

    def _queue_current_segment(self):
        """Queue current audio buffer for transcription."""
        if not self.current_audio:
            return

        audio_data = np.concatenate(self.current_audio, axis=0)

        # Only transcribe if we have enough audio (at least 0.3 seconds)
        if len(audio_data) > SAMPLE_RATE * 0.3:
            # Prepend overlap buffer for context
            if self.overlap_buffer is not None:
                audio_with_context = np.concatenate([self.overlap_buffer, audio_data], axis=0)
            else:
                audio_with_context = audio_data

            # Save last 1 second as overlap for next chunk
            if len(audio_data) > self.overlap_samples:
                self.overlap_buffer = audio_data[-self.overlap_samples:]
            else:
                self.overlap_buffer = audio_data.copy()

            self.transcription_queue.put((audio_with_context, self.total_samples, self.last_transcription))
            self.on_status("Transcribing...")

        self.current_audio = []
        self.current_audio_samples = 0

    def _strip_overlap_words(self, text: str, previous: str) -> str:
        """Remove words from start of text that overlap with end of previous."""
        if not previous or not text:
            return text

        prev_words = previous.lower().split()
        new_words = text.split()

        if not prev_words or not new_words:
            return text

        # Look for overlap: find where new_words starts repeating prev_words ending
        # Check last N words of previous against first N words of new
        max_overlap = min(len(prev_words), len(new_words), 10)  # Check up to 10 words

        best_overlap = 0
        for overlap_len in range(1, max_overlap + 1):
            prev_end = prev_words[-overlap_len:]
            new_start = [w.lower() for w in new_words[:overlap_len]]
            if prev_end == new_start:
                best_overlap = overlap_len

        if best_overlap > 0:
            return " ".join(new_words[best_overlap:])
        return text

    def _transcription_worker(self):
        """Background thread for transcribing queued segments."""
        while self.running or not self.transcription_queue.empty():
            try:
                audio_data, start_sample, prev_transcription = self.transcription_queue.get(timeout=0.1)
            except Empty:
                continue

            # Save to temp file for mlx_whisper
            audio_int16 = (audio_data.flatten() * 32767).astype(np.int16)
            fd, temp_path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)

            try:
                wavfile.write(temp_path, SAMPLE_RATE, audio_int16)
                result = mlx_whisper.transcribe(temp_path, path_or_hf_repo=self.model_repo)
                text = result["text"].strip()

                # Strip overlapping words from previous chunk
                text = self._strip_overlap_words(text, prev_transcription)

                if text:
                    segment = TranscriptionSegment(
                        text=text,
                        start_sample=start_sample,
                        end_sample=start_sample + len(audio_data)
                    )
                    with self.lock:
                        self.segments.append(segment)
                        self.last_transcription = text
                    # Paste segment immediately
                    self.on_segment(text)

            except Exception as e:
                print(f"Transcription error: {e}")
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)

            self.transcription_queue.task_done()

    def stop(self) -> str:
        """
        Stop streaming and return accumulated transcription.

        Returns:
            Complete transcribed text from all segments
        """
        # Handle any remaining audio in buffer
        if self.speech_active and self.current_audio:
            audio_data = np.concatenate(self.current_audio, axis=0)
            if len(audio_data) > SAMPLE_RATE * 0.3:
                # Add overlap context for final segment
                if self.overlap_buffer is not None:
                    audio_data = np.concatenate([self.overlap_buffer, audio_data], axis=0)
                self.transcription_queue.put((audio_data, self.total_samples, self.last_transcription))

        self.speech_active = False
        self.current_audio = []

        # Wait for transcription queue to drain
        self.transcription_queue.join()

        # Stop the worker thread
        self.running = False
        if self.transcription_thread:
            self.transcription_thread.join(timeout=1.0)

        # Clean up VAD
        self.vad.stop()

        # Combine all segments
        with self.lock:
            # Sort by start time and join
            self.segments.sort(key=lambda s: s.start_sample)
            text = " ".join(seg.text for seg in self.segments)

        return text.strip()
