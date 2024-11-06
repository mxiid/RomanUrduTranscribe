from pydub import AudioSegment
import math
from pathlib import Path
import gc
import wave
import contextlib

class AudioSplitter:
    def __init__(self, max_size_mb=25, overlap_seconds=10):
        self.max_size_bytes = (max_size_mb * 1024 * 1024) - (1 * 1024 * 1024)
        self.overlap_ms = overlap_seconds * 1000
        self.chunk_duration_ms = 10 * 60 * 1000  # 10 minutes chunks

    def get_audio_length(self, file_path):
        """Get audio length without loading the entire file"""
        try:
            audio = AudioSegment.from_file(file_path)
            duration = len(audio)
            del audio
            gc.collect()
            return duration
        except Exception as e:
            raise Exception(f"Error getting audio length: {str(e)}")

    def get_chunks_info(self, total_duration_ms):
        """Generate chunk information based on total duration"""
        start_ms = 0
        
        while start_ms < total_duration_ms:
            end_ms = min(start_ms + self.chunk_duration_ms, total_duration_ms)
            if end_ms - start_ms < 1000:  # Skip chunks shorter than 1 second
                break
            yield (start_ms, end_ms)
            start_ms = end_ms - self.overlap_ms
            
            # Safety check
            if start_ms >= total_duration_ms:
                break

    def load_chunk(self, file_path, start_ms, end_ms):
        """Load a specific chunk of audio"""
        try:
            audio = AudioSegment.from_file(file_path)
            chunk = audio[start_ms:end_ms]
            del audio
            gc.collect()
            return {
                'audio': chunk,
                'start_time': start_ms,
                'end_time': end_ms
            }
        except Exception as e:
            raise Exception(f"Error loading chunk: {str(e)}")