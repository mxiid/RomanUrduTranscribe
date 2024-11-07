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
        """Just transcribe the chunk"""
        temp_path = f"temp_chunk_{chunk_data['start_time']}.wav"
        try:
            chunk_data['audio'].export(temp_path, format="wav")
            
            with open(temp_path, "rb") as file:
                transcription = openai.audio.transcriptions.create(
                    model="whisper-1",
                    file=file,
                    response_format="verbose_json",
                    language="hi",
                    temperature=0.2,
                    prompt=("Yeh ek business meeting hai jisme circular debt, power sector, "
                        "distribution companies, aur generation sites ke baare mein "
                        "baat ho rahi hai. IESCO aur FESCO jaise utilities ke "
                        "problems discuss ho rahe hain. Payment aur debt ke issues "
                        "par focus hai."
                    )
                )
            
            formatted_text = ""
            for segment in transcription.segments:
                start_time = str(timedelta(seconds=round(segment.start + chunk_data['start_time']/1000)))
                end_time = str(timedelta(seconds=round(segment.end + chunk_data['start_time']/1000)))
                formatted_text += f"[{start_time} - {end_time}] {segment.text}\n"
            
            return {'text': formatted_text}
            
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def refine_chunk(self, chunk_result: Dict, previous_context: str = "") -> Dict:
        """Refine with GPT-4"""
        try:
            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a Roman Urdu transcription refiner..."},
                    {"role": "user", "content": f"Previous: {previous_context}\n\nRefine: {chunk_result['text']}"}
                ],
                temperature=0
            )
            
            return {'text': response.choices[0].message.content}
            
        except Exception as e:
            print(f"GPT-4 refinement error: {str(e)}")
            return chunk_result
