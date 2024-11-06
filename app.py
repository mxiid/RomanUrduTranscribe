import streamlit as st
from audio_splitter import AudioSplitter
from transcription_manager import TranscriptionManager
import tempfile
import os
import gc
import math

def process_long_audio(file_path):
    # Initialize components with adjusted settings
    splitter = AudioSplitter(max_size_mb=24, overlap_seconds=5)
    manager = TranscriptionManager()
    
    try:
        # Get total duration
        total_duration = splitter.get_audio_length(file_path)
        
        # Calculate chunks based on minutes
        minutes = total_duration / (60 * 1000)  # Convert ms to minutes
        chunk_count = math.ceil(minutes / 5)  # Split into 5-minute chunks
        
        st.info(f"Audio will be processed in {chunk_count} chunks")
        
        # Process chunks with progress bar
        progress_bar = st.progress(0)
        transcription_results = []
        previous_context = ""
        
        processed_chunks = 0
        for start_ms, end_ms in chunks_generator:
            # Update progress
            progress = min(0.99, processed_chunks / chunk_count)
            progress_bar.progress(progress)
            
            # Process chunk
            with st.spinner(f'Processing chunk {processed_chunks + 1} of {chunk_count}...'):
                try:
                    # Load and process single chunk
                    chunk = splitter.load_chunk(file_path, start_ms, end_ms)
                    
                    # Transcribe chunk
                    chunk_result = manager.transcribe_chunk(chunk, previous_context)
                    
                    # Refine transcription
                    refined_result = manager.refine_chunk(chunk_result, previous_context)
                    
                    # Store result
                    transcription_results.append(refined_result)
                    
                    # Update context for next chunk
                    previous_context = '\n'.join([
                        line.split('] ')[-1] 
                        for line in refined_result['text'].split('\n')
                        if line.strip()
                    ])[-500:]  # Keep last 500 characters
                    
                    # Force memory cleanup
                    del chunk
                    gc.collect()
                    
                except Exception as e:
                    st.error(f"Error processing chunk {processed_chunks + 1}: {str(e)}")
                    continue
            
            processed_chunks += 1
            
            # Safety check - stop if we've processed more chunks than expected
            if processed_chunks > chunk_count + 1:  # Allow for 1 extra chunk due to rounding
                st.warning("Processed more chunks than expected. Please check the audio splitting logic.")
                break
        
        # Combine results and handle overlaps
        final_transcription = combine_transcriptions(transcription_results)
        
        return final_transcription
        
    except Exception as e:
        st.error(f"Error processing audio: {str(e)}")
        return None

def combine_transcriptions(transcription_results):
    """Combine transcription chunks while handling overlaps"""
    combined_lines = []
    overlap_threshold = 30000  # 30 seconds in milliseconds
    
    for i, result in enumerate(transcription_results):
        lines = result['text'].split('\n')
        
        for line in lines:
            if not line.strip():
                continue
                
            # Parse timestamp and text
            try:
                timestamp_part = line[line.find('[')+1:line.find(']')]
                start_time = parse_timestamp(timestamp_part.split(' - ')[0])
                
                # Check for overlap with previous lines
                if combined_lines:
                    prev_timestamp = combined_lines[-1][line.find('[')+1:line.find(']')]
                    prev_end_time = parse_timestamp(prev_timestamp.split(' - ')[1])
                    
                    # Skip if too close to previous line
                    if abs(start_time - prev_end_time) < overlap_threshold:
                        continue
                
                combined_lines.append(line)
            except:
                continue
    
    return '\n'.join(combined_lines)

def parse_timestamp(timestamp_str):
    """Convert HH:MM:SS timestamp to milliseconds"""
    h, m, s = map(int, timestamp_str.split(':'))
    return (h * 3600 + m * 60 + s) * 1000

def main():
    st.title("Long Audio Transcription - Roman Urdu")
    
    # Add memory warning
    st.warning("For optimal performance, please ensure your audio file is compressed and in MP3 format.")
    
    # Add file size limit
    max_file_size = 500 * 1024 * 1024  # 500MB limit
    
    uploaded_file = st.file_uploader("Upload audio file", type=["mp3"])  # Limit to MP3 only for now
    
    if uploaded_file:
        # Check file size
        file_size = len(uploaded_file.getvalue())
        if file_size > max_file_size:
            st.error(f"File too large! Maximum size is 500MB. Your file is {file_size / (1024 * 1024):.1f}MB")
            return
            
        # Show file info
        st.info(f"File size: {file_size / (1024 * 1024):.1f}MB")
        
        # Create a temporary file with the uploaded content
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        try:
            # Process the audio file
            final_transcription = process_long_audio(tmp_file_path)
            
            if final_transcription:
                st.success("Transcription completed!")
                
                # Create tabs for viewing options
                view_tab, raw_tab = st.tabs(["Formatted View", "Raw Text"])
                
                with view_tab:
                    for line in final_transcription.split('\n'):
                        if line.strip():
                            st.text(line)
                
                with raw_tab:
                    st.text(final_transcription)
                
                # Download button
                st.download_button(
                    label="Download Transcription",
                    data=final_transcription,
                    file_name="full_transcription.txt",
                    mime="text/plain"
                )
            
        finally:
            # Clean up
            os.unlink(tmp_file_path)
            gc.collect()

if __name__ == "__main__":
    main()