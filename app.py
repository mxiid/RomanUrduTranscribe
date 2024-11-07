import streamlit as st
from audio_splitter import AudioSplitter
from transcription_manager import TranscriptionManager
import tempfile
import os
import gc
import math
from pydub import AudioSegment
import openai

def process_audio_in_chunks(file_path):
    splitter = AudioSplitter(max_size_mb=12, overlap_seconds=1)
    manager = TranscriptionManager()
    
    try:
        # Get total duration and calculate chunks
        total_duration = splitter.get_audio_length(file_path)
        chunk_count = math.ceil(total_duration / (5 * 60 * 1000)) + 1  # Add buffer of 1
        
        st.info(f"Step 1: Splitting and transcribing {chunk_count} chunks...")
        progress_bar = st.progress(0)
        
        # Create directory for chunk transcriptions if it doesn't exist
        os.makedirs("chunk_transcriptions", exist_ok=True)
        
        # Store transcriptions
        transcriptions = []
        
        for i, (start_ms, end_ms) in enumerate(splitter.get_chunks_info(total_duration)):
            # Ensure progress never exceeds 1.0
            progress = min(0.99, (i + 1) / chunk_count)
            progress_bar.progress(progress)
            
            with st.spinner(f'Processing chunk {i + 1} of {chunk_count}...'):
                chunk = splitter.load_chunk(file_path, start_ms, end_ms)
                result = manager.transcribe_chunk(chunk)
                transcriptions.append(result)
                
                # Save individual chunk transcription
                chunk_file = f"chunk_transcriptions/chunk_{i+1}_of_{chunk_count}.txt"
                with open(chunk_file, 'w') as f:
                    f.write(f"Chunk {i+1} of {chunk_count}\n")
                    f.write(f"Time range: {start_ms/1000:.2f}s to {end_ms/1000:.2f}s\n")
                    f.write("-" * 50 + "\n")
                    f.write(result['text'])
                
                # Show path to saved file
                st.info(f"Saved chunk transcription to: {chunk_file}")
                
                del chunk
                gc.collect()
        
        # Save complete raw transcription
        temp_path = "raw_transcriptions.txt"
        with open(temp_path, 'w') as f:
            for t in transcriptions:
                f.write(t['text'] + "\n---CHUNK_BREAK---\n")
        
        return temp_path
        
    except Exception as e:
        st.error(f"Error in processing: {str(e)}")
        return None

def refine_transcriptions(file_path):
    manager = TranscriptionManager()
    
    try:
        st.info("Step 2: Refining transcriptions with GPT-4...")
        
        # Read raw transcriptions
        with open(file_path, 'r') as f:
            content = f.read()
        
        chunks = content.split("---CHUNK_BREAK---")
        chunks = [c.strip() for c in chunks if c.strip()]
        
        # Process chunks with context
        refined_text = ""
        previous_context = ""
        
        for i, chunk in enumerate(chunks):
            with st.spinner(f'Refining chunk {i + 1} of {len(chunks)}...'):
                refined_chunk = manager.refine_chunk({'text': chunk}, previous_context)
                refined_text += refined_chunk['text'] + "\n"
                previous_context = extract_context(refined_chunk['text'])
        
        return refined_text
        
    except Exception as e:
        st.error(f"Error in refinement: {str(e)}")
        return None

def extract_context(text: str) -> str:
    """Extract meaningful context from the last few lines"""
    lines = [l for l in text.split('\n') if l.strip()]
    if not lines:
        return ""
        
    # Take last 2 meaningful lines without timestamps
    context_lines = []
    for line in lines[-2:]:
        text_part = line[line.find(']')+1:].strip()
        if text_part and not text_part.endswith('...'):
            context_lines.append(text_part)
    
    return ' '.join(context_lines)

def main():
    st.title("Long Audio Transcription - Roman Urdu")
    
    uploaded_file = st.file_uploader("Upload audio file", type=["mp3"])
    
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        try:
            # Step 1: Process chunks
            transcriptions_path = process_audio_in_chunks(tmp_file_path)
            
            if transcriptions_path:
                # Step 2: Refine with GPT
                final_text = refine_transcriptions(transcriptions_path)
                
                if final_text:
                    st.success("Processing completed!")
                    
                    # Display results
                    st.markdown("### Final Transcription")
                    st.text(final_text)
                    
                    # Download button
                    st.download_button(
                        "Download Transcription",
                        final_text,
                        "transcription.txt",
                        "text/plain"
                    )
        
        finally:
            # Cleanup
            os.unlink(tmp_file_path)
            if transcriptions_path and os.path.exists(transcriptions_path):
                os.unlink(transcriptions_path)
            gc.collect()

if __name__ == "__main__":
    main()
