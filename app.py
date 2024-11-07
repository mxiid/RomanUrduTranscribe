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
    # Initialize with smaller chunks
    splitter = AudioSplitter(max_size_mb=12, overlap_seconds=1)
    manager = TranscriptionManager()

    try:
        total_duration = splitter.get_audio_length(file_path)
        minutes = total_duration / (60 * 1000)
        chunk_count = math.ceil(minutes / 5)

        chunks_generator = splitter.get_chunks_info(total_duration)

        st.info(f"Audio will be processed in {chunk_count} chunks of 5 minutes each")

        progress_bar = st.progress(0)
        whisper_results = []
        previous_context = ""

        processed_chunks = 0
        for start_ms, end_ms in chunks_generator:
            progress = min(0.99, processed_chunks / chunk_count)
            progress_bar.progress(progress)

            with st.spinner(f'Processing chunk {processed_chunks + 1} of {chunk_count}...'):
                try:
                    chunk = splitter.load_chunk(file_path, start_ms, end_ms)

                    # Simple prompt for Whisper - just focus on transcription
                    chunk["prompt"] = (
                        "Yeh ek business meeting hai jisme circular debt, power sector, "
                        "distribution companies, aur generation sites ke baare mein "
                        "baat ho rahi hai. IESCO aur FESCO jaise utilities ke "
                        "problems discuss ho rahe hain. Payment aur debt ke issues "
                        "par focus hai. Har sentence ko Roman Urdu mein transcribe karein."
                    )

                    # Get Whisper transcription
                    whisper_result = manager.transcribe_chunk(chunk)  # No context passed

                    # Check for recursion in the result
                    if check_for_recursion(whisper_result['text']):
                        st.warning(f"Detected potential recursion in chunk {processed_chunks + 1}. Retrying...")
                        chunk['prompt'] = "Har line alag honi chahiye, repetition nahi honi chahiye."
                        whisper_result = manager.transcribe_chunk(chunk)  # Retry without context

                    # Now pass context to GPT for refinement
                    refined_result = manager.refine_chunk(whisper_result, previous_context)
                    whisper_results.append(refined_result)

                    # Update context for next chunk from refined result
                    previous_context = extract_context(refined_result['text'])

                    del chunk
                    gc.collect()

                except Exception as e:
                    st.error(f"Error processing chunk {processed_chunks + 1}: {str(e)}")
                    continue

            processed_chunks += 1

        # Combine all results
        final_text = ""
        for result in whisper_results:
            final_text += result['text'] + "\n"

        st.success("Transcription completed!")

        # Create tabs for different viewing options
        formatted_tab, raw_tab = st.tabs(["Formatted View", "Raw Text"])

        with formatted_tab:
            st.markdown("### Whisper Transcription")
            for line in final_text.split('\n'):
                if line.strip():
                    st.text(line)

        with raw_tab:
            st.text(final_text)

        # Download button
        st.download_button(
            label="Download Transcription",
            data=final_text,
            file_name="whisper_transcription.txt",
            mime="text/plain"
        )

    except Exception as e:
        st.error(f"Error in processing: {str(e)}")
        return None

def check_for_recursion(text: str) -> bool:
    """Check if the transcription contains recursive patterns"""
    lines = text.split('\n')
    if len(lines) < 3:
        return False
        
    # Check for repeated phrases
    seen_phrases = set()
    repeated_count = 0
    
    for line in lines:
        # Get the actual text without timestamp
        text_part = line[line.find(']')+1:].strip()
        if text_part in seen_phrases:
            repeated_count += 1
            if repeated_count > 2:  # Allow max 2 repetitions
                return True
        seen_phrases.add(text_part)
    
    return False

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

def process_audio_oneshot(file_path):
    try:
        manager = TranscriptionManager()
        
        # Debug: Check audio file and trim to start from 2:12
        try:
            audio = AudioSegment.from_file(file_path)
            
            # Trim to start from 2:12 (132 seconds)
            start_time = 132 * 1000  # Convert to milliseconds
            trimmed_audio = audio[start_time:]
            
            st.info(f"""
                Audio file debug info:
                - Original Duration: {len(audio)/1000:.2f} seconds
                - Trimmed Duration: {len(trimmed_audio)/1000:.2f} seconds
                - Channels: {audio.channels}
                - Sample Rate: {audio.frame_rate} Hz
            """)
            
            # Save trimmed version
            debug_path = "debug_whisper_input.mp3"
            trimmed_audio.export(debug_path, format="mp3")
            
            # Add download button for the debug file
            with open(debug_path, 'rb') as debug_file:
                st.download_button(
                    label="Download Trimmed Audio File",
                    data=debug_file,
                    file_name="debug_whisper_input.mp3",
                    mime="audio/mp3"
                )
            
        except Exception as e:
            st.error(f"Error processing audio file: {str(e)}")
            return None
            
        with st.spinner('Processing with Whisper API...'):
            try:
                with open(debug_path, "rb") as file:
                    transcription = openai.audio.transcriptions.create(
                        model="whisper-1",
                        file=file,
                        response_format="verbose_json",
                        language="hi",
                        temperature=0.2,
                        prompt=(
                            "Acha, toh aap business ke baare mein baat kar rahe hain. "
                            "Main samajh rahi hoon. Market research ke mutabiq... "
                            "Hamari company mein yeh process follow kiya jata hai. "
                            "Stakeholders ko inform karna zaroori hai."
                            "IESCO, FESCO"
                        )
                    )
                
                # Add debug info for Whisper response
                st.info(f"""
                    Whisper Response Debug:
                    - Number of segments: {len(transcription.segments)}
                    - Duration: {transcription.duration}
                """)
                
                # Format the transcription with timestamps
                formatted_text = ""
                for segment in transcription.segments:
                    # Convert segment times to milliseconds and add offset
                    segment_start = int((segment.start * 1000) + start_time)
                    segment_end = int((segment.end * 1000) + start_time)
                    
                    # Format timestamps
                    start_time_str = manager.format_timestamp(segment_start)
                    end_time_str = manager.format_timestamp(segment_end)
                    
                    formatted_text += f"[{start_time_str} - {end_time_str}] {segment.text}\n"
                
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
                    
            except Exception as e:
                st.error(f"Transcription error: {str(e)}")
            finally:
                # Clean up debug file
                if os.path.exists(debug_path):
                    os.remove(debug_path)
                
    except Exception as e:
        st.error(f"Error in processing: {str(e)}")
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
