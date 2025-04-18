"""Voice activity detection."""
import io
from typing import Optional


class SileroVad:
    """Voice activity detection with silero VAD."""

    def __init__(self, threshold: float, trigger_level: int) -> None:
        from pysilero_vad import SileroVoiceActivityDetector

        self.detector = SileroVoiceActivityDetector()
        self.threshold = threshold
        self.trigger_level = trigger_level
        self._activation = 0
        
        # Buffer for audio chunks that are not the right size
        self.audio_buffer = io.BytesIO()
        # Silero VAD typically expects chunks of 512 samples (1024 bytes for 16-bit audio)
        self.chunk_size = 1024  # 512 samples * 2 bytes per sample for 16-bit audio

    def __call__(self, audio_bytes: Optional[bytes]) -> bool:
        if audio_bytes is None:
            # Reset
            self._activation = 0
            self.detector.reset()
            self.audio_buffer = io.BytesIO()  # Clear buffer
            return False

        # Add new audio to the buffer
        self.audio_buffer.write(audio_bytes)
        buffer_value = self.audio_buffer.getvalue()
        
        # Process complete chunks from the buffer
        speech_detected = False
        while len(buffer_value) >= self.chunk_size:
            # Extract a properly sized chunk
            chunk = buffer_value[:self.chunk_size]
            buffer_value = buffer_value[self.chunk_size:]
            
            # Process the chunk
            try:
                if self.detector(chunk) >= self.threshold:
                    # Speech detected
                    self._activation += 1
                    if self._activation >= self.trigger_level:
                        self._activation = 0
                        speech_detected = True
                else:
                    # Silence detected
                    self._activation = max(0, self._activation - 1)
            except Exception as e:
                # If an error occurs, log it and continue
                import logging
                logging.getLogger().error(f"VAD error processing chunk: {e}")
                self._activation = max(0, self._activation - 1)
        
        # Store remaining incomplete chunk back in the buffer
        self.audio_buffer = io.BytesIO()
        self.audio_buffer.write(buffer_value)
        
        return speech_detected