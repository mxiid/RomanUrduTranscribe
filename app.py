import streamlit as st
from transcription_manager import TranscriptionManager
import tempfile
import os
from pydub import AudioSegment
import gc

def process_audio_oneshot(file_path):
    manager = TranscriptionManager()
    
    try:
        st.info("Processing audio file...")
        
        # Load and prepare audio
        audio = AudioSegment.from_file(file_path)
        audio = audio.set_channels(1).set_frame_rate(16000)
        
        # Export as temporary WAV file
        temp_wav = "temp_audio.wav"
        audio.export(temp_wav, format="wav")
        
        # Transcribe entire file
        with st.spinner('Transcribing...'):
            whisper_result = manager.transcribe_chunk({
                'audio': audio,
                'start_time': 0,
                'end_time': len(audio)
            })
            
            # Save raw transcription
            with open("raw_transcription.txt", "w") as f:
                f.write(whisper_result['text'])
            st.info("Saved raw transcription to: raw_transcription.txt")
            
            # Refine with GPT-4
            st.info("Refining with GPT-4...")
            refined_result = manager.refine_chunk(whisper_result)
            
            return whisper_result['text'], refined_result['text']
            
    except Exception as e:
        st.error(f"Error in processing: {str(e)}")
        return None, None
    finally:
        # Cleanup
        if os.path.exists(temp_wav):
            os.remove(temp_wav)

def main():
    st.title("Long Audio Transcription - Roman Urdu")
    
    uploaded_file = st.file_uploader("Upload audio file", type=["mp3"])
    
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        try:
            whisper_text, refined_text = process_audio_oneshot(tmp_file_path)
            
            if whisper_text and refined_text:
                st.success("Processing completed!")
                
                # Create three tabs
                whisper_tab, gpt_tab, diff_tab = st.tabs(["Whisper Transcription", "GPT-4 Refined", "Differences"])
                
                with whisper_tab:
                    st.markdown("### Raw Whisper Transcription")
                    st.text(whisper_text)
                    # Download button for Whisper
                    st.download_button(
                        "Download Whisper Transcription",
                        whisper_text,
                        "whisper_transcription.txt",
                        "text/plain"
                    )
                
                with gpt_tab:
                    st.markdown("### GPT-4 Refined Transcription")
                    st.text(refined_text)
                    # Download button for GPT
                    st.download_button(
                        "Download GPT Refined Transcription",
                        refined_text,
                        "gpt_refined_transcription.txt",
                        "text/plain"
                    )
                
                with diff_tab:
                    st.markdown("### Differences between Transcriptions")
                    # Show differences in a more readable format
                    whisper_lines = whisper_text.split('\n')
                    gpt_lines = refined_text.split('\n')
                    
                    for i, (w_line, g_line) in enumerate(zip(whisper_lines, gpt_lines)):
                        if w_line != g_line:
                            st.markdown(f"**Line {i+1}:**")
                            st.text("Whisper: " + w_line)
                            st.text("GPT-4:  " + g_line)
                            st.markdown("---")
        
        finally:
            # Cleanup
            os.unlink(tmp_file_path)
            gc.collect()

if __name__ == "__main__":
    main()
