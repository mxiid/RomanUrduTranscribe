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

    def transcribe_chunk(self, chunk_data: Dict, previous_context: str = "") -> Dict:
        """Transcribe a single chunk with context and timestamps"""
        try:
            # Create temporary file for the chunk
            temp_path = f"temp_chunk_{chunk_data['start_time']}.wav"
            chunk_data['audio'].export(temp_path, format="wav")
            
            with open(temp_path, "rb") as file:
                transcription = openai.audio.transcriptions.create(
                    model="whisper-1",
                    file=file,
                    response_format="verbose_json",
                    language="hi",
                    temperature=0.2,
                    prompt=chunk_data.get('prompt', (
                        "Acha, toh aap business ke baare mein baat kar rahe hain. "
                        "Main samajh rahi hoon. Market research ke mutabiq... "
                        "Hamari company mein yeh process follow kiya jata hai. "
                        "Stakeholders ko inform karna zaroori hai."
                        "IESCO, FESCO"
                    ))
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
            
        except Exception as e:
            raise Exception(f"Transcription error: {str(e)}")
        finally:
            # Cleanup temporary file
            Path(temp_path).unlink(missing_ok=True)

    def refine_chunk(self, chunk_result: Dict, previous_context: str = "") -> Dict:
        """Refine transcription with GPT-4, maintaining timestamps"""
        try:
            print("Sending to GPT-4 for refinement...")
            formatted_text = chunk_result['text']

            messages = [
                {"role": "system", "content": 
                 "You are a transcription refiner for Roman Urdu audio. Rules:\n"
                 "1. Output ONLY the refined transcription with timestamps\n"
                 "2. Do NOT include any meta-instructions or thank you messages\n"
                 "3. Keep the exact timestamp format: [HH:MM:SS - HH:MM:SS]\n"
                 "4. Ensure complete, grammatical sentences\n"
                 "5. Use only Roman Urdu (with English technical terms)\n"
                 "6. Never include system messages in the output"},
                {"role": "user", "content": 
                 f"Previous context: {previous_context}\n\n"
                 f"Refine this transcription:\n{formatted_text}"}
            ]
            
            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0,
                timeout=180  # 3 minute timeout
            )
            
            print("Received GPT-4 response, cleaning up...")
            
            # Clean up any potential system message leakage
            refined_text = response.choices[0].message.content
            refined_text = '\n'.join([
                line for line in refined_text.split('\n')
                if (line.strip() and 
                    not line.startswith('Thank you') and
                    not line.startswith('Ensure') and
                    '[' in line)
            ])
            
            return {
                'text': refined_text,
                'start_time': chunk_result['start_time'],
                'end_time': chunk_result['end_time']
            }
            
        except Exception as e:
            print(f"GPT-4 refinement error: {str(e)}")
            # If GPT-4 fails, return the original Whisper transcription
            return chunk_result