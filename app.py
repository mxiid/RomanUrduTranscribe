import streamlit as st
from transcription_manager import TranscriptionManager
import tempfile
import os
from pydub import AudioSegment

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
            result = manager.transcribe_chunk({
                'audio': audio,
                'start_time': 0,
                'end_time': len(audio)
            })
            
            # Save raw transcription
            with open("raw_transcription.txt", "w") as f:
                f.write(result['text'])
            st.info("Saved raw transcription to: raw_transcription.txt")
            
            # Refine with GPT-4
            st.info("Refining with GPT-4...")
            refined_result = manager.refine_chunk(result)
            
            return refined_result['text']
            
    except Exception as e:
        st.error(f"Error in processing: {str(e)}")
        return None
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
            final_text = process_audio_oneshot(tmp_file_path)
            
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
            gc.collect()

if __name__ == "__main__":
    main()
