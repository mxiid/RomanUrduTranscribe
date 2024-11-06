import streamlit as st
from audio_splitter import AudioSplitter
from transcription_manager import TranscriptionManager
import tempfile
import os
import gc
import math
from pydub import AudioSegment
import openai

def process_long_audio(file_path):
    # Initialize components with adjusted settings
    splitter = AudioSplitter(max_size_mb=24, overlap_seconds=10)
    manager = TranscriptionManager()
    
    try:
        # Get total duration
        total_duration = splitter.get_audio_length(file_path)
        
        # Calculate chunks based on minutes
        minutes = total_duration / (60 * 1000)  # Convert ms to minutes
        chunk_count = math.ceil(minutes / 10)  # Changed to 10-minute chunks
        
        # Create chunks generator
        chunks_generator = splitter.get_chunks_info(total_duration)
        
        st.info(f"Audio will be processed in {chunk_count} chunks")
        
        # Process chunks with progress bar
        progress_bar = st.progress(0)
        whisper_results = []
        gpt_results = []
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
                    
                    # Get Whisper transcription
                    whisper_result = manager.transcribe_chunk(chunk, previous_context)
                    whisper_results.append(whisper_result)
                    
                    # Get GPT-4 refined version
                    gpt_result = manager.refine_chunk(whisper_result, previous_context)
                    gpt_results.append(gpt_result)
                    
                    # Update context for next chunk
                    previous_context = '\n'.join([
                        line.split('] ')[-1] 
                        for line in gpt_result['text'].split('\n')
                        if line.strip()
                    ])[-500:]
                    
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
        
        # Combine results
        whisper_transcription = combine_transcriptions(whisper_results)
        gpt_transcription = combine_transcriptions(gpt_results)
        
        if whisper_transcription and gpt_transcription:
            st.success("Transcription completed!")
            
            # Create tabs for different viewing options
            whisper_tab, gpt_tab, raw_tab = st.tabs(["Whisper Output", "GPT Refined", "Raw Text"])
            
            with whisper_tab:
                st.markdown("### Original Whisper Output")
                for line in whisper_transcription.split('\n'):
                    if line.strip():
                        st.text(line)
            
            with gpt_tab:
                st.markdown("### GPT-4 Refined Version")
                for line in gpt_transcription.split('\n'):
                    if line.strip():
                        st.text(line)
            
            with raw_tab:
                st.text(gpt_transcription)
            
            # Download buttons for both versions
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    label="Download Whisper Version",
                    data=whisper_transcription,
                    file_name="whisper_transcription.txt",
                    mime="text/plain"
                )
            with col2:
                st.download_button(
                    label="Download GPT Version",
                    data=gpt_transcription,
                    file_name="gpt_transcription.txt",
                    mime="text/plain"
                )
        
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

def process_audio_oneshot(file_path):
    """Process entire audio file in one shot using Whisper API"""
    try:
        manager = TranscriptionManager()
        
        with st.spinner('Processing entire file with Whisper API...'):
            # Convert to WAV with correct parameters for Whisper
            temp_path = "temp_whisper.wav"
            audio = AudioSegment.from_file(file_path)
            audio = audio.set_channels(1).set_frame_rate(16000)
            audio.export(
                temp_path,
                format="wav",
                parameters=[
                    "-ac", "1",     # mono
                    "-ar", "16000"  # 16kHz
                ]
            )
            
            file_size_mb = os.path.getsize(temp_path) / (1024 * 1024)
            st.info(f"Processed file size: {file_size_mb:.2f}MB")
            
            try:
                with open(temp_path, "rb") as file:
                    transcription = openai.audio.transcriptions.create(
                        model="whisper-1",
                        file=file,
                        response_format="verbose_json",
                        language="ur"
                    )
                
                # Format the transcription with timestamps
                formatted_text = ""
                for segment in transcription.segments:
                    start_time = manager.format_timestamp(segment.start * 1000)
                    end_time = manager.format_timestamp(segment.end * 1000)
                    formatted_text += f"[{start_time} - {end_time}] {segment.text}\n"
                
                result = {'text': formatted_text}
                
                if result:
                    st.success("Transcription completed!")
                    
                    # Create tabs for different viewing options
                    formatted_tab, raw_tab = st.tabs(["Formatted View", "Raw Text"])
                    
                    with formatted_tab:
                        st.markdown("### Whisper Transcription")
                        for line in result['text'].split('\n'):
                            if line.strip():
                                st.text(line)
                    
                    with raw_tab:
                        st.text(result['text'])
                    
                    # Download button
                    st.download_button(
                        label="Download Transcription",
                        data=result['text'],
                        file_name="whisper_transcription.txt",
                        mime="text/plain"
                    )
            finally:
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                
    except Exception as e:
        st.error(f"Error in one-shot processing: {str(e)}")
        return None

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
        file_size_mb = file_size / (1024 * 1024)
        
        if file_size > max_file_size:
            st.error(f"File too large! Maximum size is 500MB. Your file is {file_size_mb:.1f}MB")
            return
            
        # Show file info
        st.info(f"File size: {file_size_mb:.1f}MB")
        
        # Create a temporary file with the uploaded content
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        try:
            # Let user choose processing method if file is small enough
            if file_size_mb < 25:
                processing_method = st.radio(
                    "Choose processing method:",
                    ["One-shot processing (Whisper only)", "Chunk processing (Whisper + GPT)"],
                    help="One-shot processing is faster but only uses Whisper. Chunk processing adds GPT refinement but takes longer."
                )
                
                if processing_method == "One-shot processing (Whisper only)":
                    process_audio_oneshot(tmp_file_path)
                else:
                    process_long_audio(tmp_file_path)
            else:
                st.info("File size requires chunk processing (Whisper + GPT)")
                process_long_audio(tmp_file_path)
            
        finally:
            # Clean up
            os.unlink(tmp_file_path)
            gc.collect()

if __name__ == "__main__":
    main()
