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
    def transcribe_chunk(self, chunk_data: Dict) -> Dict:
        """Transcribe the entire audio file"""
        temp_path = "temp_transcription.wav"
        try:
            chunk_data['audio'].export(temp_path, format="wav")
            
            with open(temp_path, "rb") as file:
                transcription = openai.audio.transcriptions.create(
                    model="whisper-1",
                    file=file,
                    response_format="verbose_json",
                    language="ur",
                    temperature=0,
                    prompt="Transcribe this audio file in Urdu while retaining English terms."
                )
            
            formatted_text = ""
            for segment in transcription.segments:
                start_time = str(timedelta(seconds=round(segment.start)))
                end_time = str(timedelta(seconds=round(segment.end)))
                formatted_text += f"[{start_time} - {end_time}] {segment.text}\n"
            
            return {'text': formatted_text}
            
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def refine_chunk(self, chunk_result: Dict) -> Dict:
        """Refine with GPT-4"""
        try:
            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": 
                     "You are a Roman Urdu transcription refiner for business audio.\n"
                     "Rules:\n"
                     "1. Keep the exact timestamp format: [HH:MM:SS - HH:MM:SS]\n"
                     "2. Ensure complete, grammatical sentences\n"
                     "3. Use Roman Urdu with English technical terms\n"
                     "4. Maintain natural conversation flow"},
                    {"role": "user", "content": f"Refine this transcription:\n{chunk_result['text']}"}
                ],
                temperature=0
            )
            
            return {'text': response.choices[0].message.content}
            
        except Exception as e:
            print(f"GPT-4 refinement error: {str(e)}")
            return chunk_result
