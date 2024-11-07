from typing import List, Dict
import openai
from datetime import timedelta
from pathlib import Path
from dotenv import load_dotenv
import os
import streamlit as st

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

    def transcribe_chunk(self, chunk_data: Dict) -> Dict:
        """Just transcribe without context"""
        try:
            temp_path = f"temp_chunk_{chunk_data['start_time']}.wav"
            chunk_data['audio'].export(temp_path, format="wav")

            with open(temp_path, "rb") as file:
                transcription = openai.audio.transcriptions.create(
                    model="whisper-1",
                    file=file,
                    response_format="verbose_json",
                    language="hi",
                    temperature=0.2,
                    prompt=chunk_data.get(
                        "prompt",
                        (
                            "Yeh ek business meeting hai jisme circular debt, power sector, "
                            "distribution companies, aur generation sites ke baare mein "
                            "baat ho rahi hai. IESCO aur FESCO jaise utilities ke "
                            "problems discuss ho rahe hain. Payment aur debt ke issues "
                            "par focus hai. Har sentence ko Roman Urdu mein transcribe karein."
                        ),
                    ),
                )

            # Format the transcription with timestamps
            formatted_text = ""
            for segment in transcription.segments:
                start_time = self.format_timestamp(segment.start * 1000 + chunk_data['start_time'])
                end_time = self.format_timestamp(segment.end * 1000 + chunk_data['start_time'])
                formatted_text += f"[{start_time} - {end_time}] {segment.text}\n"

            return {
                'text': formatted_text,
                'start_time': chunk_data['start_time'],
                'end_time': chunk_data['end_time']
            }

        finally:
            Path(temp_path).unlink(missing_ok=True)

    def refine_chunk(self, chunk_result: Dict, previous_context: str = "") -> Dict:
        """Handle context and refinement in GPT"""
        try:
            messages = [
                {"role": "system", "content": 
                 "You are a transcription refiner for Roman Urdu business audio. Context:\n"
                 "- This is about power sector, circular debt, and utilities\n"
                 "- Contains terms like IESCO, FESCO, distribution companies\n"
                 "- Discusses payments, generation, and consumer issues\n"
                 "Rules:\n"
                 "1. Keep the exact timestamp format: [HH:MM:SS - HH:MM:SS]\n"
                 "2. Ensure complete, grammatical sentences\n"
                 "3. Use Roman Urdu with English technical terms\n"
                 "4. Maintain conversation flow with previous context"},
                {"role": "user", "content": 
                 f"Previous context: {previous_context}\n\n"
                 f"Refine this transcription:\n{chunk_result['text']}"}
            ]

            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0,
                timeout=180
            )

            refined_text = response.choices[0].message.content
            refined_text = '\n'.join([
                line for line in refined_text.split('\n')
                if (line.strip() and '[' in line)
            ])

            return {
                'text': refined_text,
                'start_time': chunk_result['start_time'],
                'end_time': chunk_result['end_time']
            }

        except Exception as e:
            print(f"GPT-4 refinement error: {str(e)}")
            return chunk_result
