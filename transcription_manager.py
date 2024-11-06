from typing import List, Dict
import openai
from datetime import timedelta
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

class TranscriptionManager:
    def __init__(self):
        self.context_window = ""
        self.max_context_tokens = 1000

    def format_timestamp(self, milliseconds):
        """Convert milliseconds to HH:MM:SS format"""
        seconds = milliseconds / 1000
        return str(timedelta(seconds=round(seconds)))

    def transcribe_chunk(self, chunk_data: Dict, previous_context: str = "") -> Dict:
        """Transcribe a single chunk with context and timestamps"""
        try:
            # Create temporary file for the chunk
            temp_path = f"temp_chunk_{chunk_data['start_time']}.wav"
            chunk_data['audio'].export(temp_path, format="wav")
            
            # Prepare prompt with context
            prompt = (f"Previous context: {previous_context}\n"
                     f"Continue transcription in Roman Urdu, maintaining English words. "
                     f"Ensure smooth connection with previous context if applicable.")
            
            with open(temp_path, "rb") as file:
                transcription = openai.audio.transcriptions.create(
                    model="whisper-1",
                    file=file,
                    response_format="verbose_json",
                    language="ur",
                    prompt=prompt
                )
            
            # Adjust timestamps to account for chunk position
            base_time = chunk_data['start_time']
            for segment in transcription.segments:
                segment.start = base_time + (segment.start * 1000)  # Convert to ms
                segment.end = base_time + (segment.end * 1000)      # Convert to ms
            
            return {
                'transcription': transcription,
                'start_time': chunk_data['start_time'],
                'end_time': chunk_data['end_time']
            }
            
        finally:
            # Cleanup temporary file
            Path(temp_path).unlink(missing_ok=True)

    def refine_chunk(self, chunk_result: Dict, previous_context: str = "") -> Dict:
        """Refine transcription with GPT-4, maintaining timestamps"""
        try:
            # Prepare timestamped text for GPT-4
            segments = chunk_result['transcription'].segments
            formatted_text = ""
            for segment in segments:
                start_time = self.format_timestamp(segment.start)
                end_time = self.format_timestamp(segment.end)
                formatted_text += f"[{start_time} - {end_time}] {segment.text}\n"

            messages = [
                {"role": "system", "content": 
                 "You are processing part of a longer transcription. Your task is to:"
                 "\n1. Maintain Roman Urdu with English code-switching"
                 "\n2. Ensure smooth connection with previous context"
                 "\n3. Mark any uncertain connections with [...]"
                 "\n4. Preserve timestamps exactly as provided [HH:MM:SS - HH:MM:SS]"
                 "\n5. Keep each timestamped line separate"},
                {"role": "user", "content": 
                 f"Previous context: {previous_context}\n\n"
                 f"Current chunk transcription:\n{formatted_text}\n\n"
                 f"Please refine while maintaining timestamps and continuity:"}
            ]
            
            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0
            )
            
            return {
                'text': response.choices[0].message.content,
                'start_time': chunk_result['start_time'],
                'end_time': chunk_result['end_time']
            }
            
        except Exception as e:
            raise Exception(f"Refinement error: {str(e)}")