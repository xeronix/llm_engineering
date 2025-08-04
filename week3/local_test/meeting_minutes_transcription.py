import torch
import librosa
import numpy as np
import os
import subprocess
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers import pipeline

def extract_audio_from_mp4(mp4_file_path, output_audio_path=None):
    """Extract audio from MP4 video file using ffmpeg"""
    print(f"Extracting audio from MP4: {mp4_file_path}")
    
    try:
        # If no output path specified, create one
        if output_audio_path is None:
            base_name = os.path.splitext(mp4_file_path)[0]
            output_audio_path = f"{base_name}_extracted_audio.wav"
        
        # Use ffmpeg to extract audio
        command = [
            'ffmpeg',
            '-i', mp4_file_path,
            '-vn',  # Disable video
            '-acodec', 'pcm_s16le',  # Audio codec
            '-ar', '16000',  # Sample rate
            '-ac', '1',  # Mono
            '-y',  # Overwrite output file
            output_audio_path
        ]
        
        # Run ffmpeg command (suppress output)
        result = subprocess.run(command, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"Audio extracted to: {output_audio_path}")
            return output_audio_path
        else:
            print(f"❌ ffmpeg error: {result.stderr}")
            return None
        
    except FileNotFoundError:
        print("❌ ffmpeg not found. Please install ffmpeg:")
        print("  On Mac: brew install ffmpeg")
        print("  On Ubuntu: sudo apt install ffmpeg")
        return None
    except Exception as e:
        print(f"❌ Error extracting audio from MP4: {e}")
        return None

def chunk_audio(audio, chunk_length_seconds=30, sampling_rate=16000):
    """Split audio into chunks of specified length"""
    chunk_length_samples = chunk_length_seconds * sampling_rate
    chunks = []
    
    for i in range(0, len(audio), chunk_length_samples):
        chunk = audio[i:i + chunk_length_samples]
        if len(chunk) > 0:  # Only add non-empty chunks
            chunks.append(chunk)
    
    return chunks

def transcribe_long_audio(file_path, model, processor, device):
    """Transcribe long audio file by processing it in chunks"""
    print(f"Loading audio file: {file_path}")
    
    # Load the full audio file
    audio, sr = librosa.load(file_path, sr=16000)
    audio_duration = len(audio) / 16000
    print(f"Audio duration: {audio_duration:.2f} seconds ({audio_duration/60:.2f} minutes)")
    
    # Split into chunks
    chunks = chunk_audio(audio, chunk_length_seconds=29)  # Use almost full 30-second limit
    print(f"Split into {len(chunks)} chunks")
    
    # Prepare forced decoder IDs
    forced_decoder_ids = processor.get_decoder_prompt_ids(language="en", task="transcribe")
    
    full_transcription = []
    
    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i+1}/{len(chunks)}...")
        
        # Process the chunk
        inputs = processor(chunk, sampling_rate=16000, return_tensors="pt")
        input_features = inputs.input_features.to(device)
        
        # Generate transcription for this chunk
        with torch.no_grad():
            predicted_ids = model.generate(
                input_features,
                forced_decoder_ids=forced_decoder_ids,
                max_new_tokens=400
            )
        
        # Decode the transcription
        chunk_transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        full_transcription.append(chunk_transcription)
        
        # Optional: print progress
        print(f"Chunk {i+1} transcribed: {chunk_transcription[:100]}...")
    
    return " ".join(full_transcription)

def generate_meeting_minutes(transcription):
    """Generate meeting minutes from transcription using a local Hugging Face model"""
    print("\n🤖 Generating meeting minutes using local model...")
    
    try:
        print("Loading text generation model...")
        
        # Use a summarization model
        summarizer = pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
            device=0 if torch.cuda.is_available() else -1
        )
        
        # Split transcription into chunks
        max_chunk_length = 1000
        transcription_chunks = [transcription[i:i+max_chunk_length] 
                              for i in range(0, len(transcription), max_chunk_length)]
        
        summaries = []
        for i, chunk in enumerate(transcription_chunks):
            print(f"Summarizing chunk {i+1}/{len(transcription_chunks)}...")
            
            summary = summarizer(chunk, max_length=200, min_length=50, do_sample=False)
            summaries.append(summary[0]['summary_text'])
        
        combined_summary = " ".join(summaries)
        
        meeting_minutes = f"""
MEETING MINUTES
===============

📝 EXECUTIVE SUMMARY:
{combined_summary}

📋 KEY DISCUSSION POINTS:
- Main topics covered in the meeting
- Important decisions and conclusions reached
- Significant information shared during the discussion

📌 NOTES:
This summary was generated from the meeting transcription using AI analysis.
For detailed information, please refer to the full transcription.

🕒 MEETING DURATION: Approximately {len(transcription.split())/150:.1f} minutes
(Based on average speaking pace of 150 words per minute)
        """
        
        return meeting_minutes
        
    except Exception as e:
        print(f"❌ Error generating meeting minutes: {e}")
        
        word_count = len(transcription.split())
        estimated_duration = word_count / 150
        
        fallback_minutes = f"""
MEETING MINUTES
===============

📝 TRANSCRIPTION SUMMARY:
The meeting was approximately {estimated_duration:.1f} minutes long and covered various topics.

📋 FULL TRANSCRIPTION:
{transcription}

📌 NOTES:
- Word count: {word_count} words
- Estimated duration: {estimated_duration:.1f} minutes
- This is the raw transcription. For a detailed analysis, please review the content manually.
        """
        
        return fallback_minutes

# Main execution
if __name__ == "__main__":
    model_id = "openai/whisper-large-v3"
    print(f"Loading Whisper model: {model_id}")
    print("Note: This is a large model (~3GB) and may take time to download on first use...")
    
    processor = WhisperProcessor.from_pretrained(model_id)
    model = WhisperForConditionalGeneration.from_pretrained(model_id)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")
    
    input_file_path = "/Users/vipul.mehta/Downloads/demo_meeting.mp4"
    
    if not os.path.exists(input_file_path):
        print(f"❌ Error: Input file not found: {input_file_path}")
        exit(1)
    
    file_extension = os.path.splitext(input_file_path)[1].lower()
    
    if file_extension == '.mp4':
        audio_file_path = extract_audio_from_mp4(input_file_path)
        if audio_file_path is None:
            print("❌ Failed to extract audio from MP4. Exiting.")
            exit(1)
    elif file_extension in ['.mp3', '.wav', '.m4a']:
        audio_file_path = input_file_path
        print(f"Using audio file directly: {audio_file_path}")
    else:
        print(f"❌ Unsupported file format: {file_extension}")
        print("Supported formats: .mp4, .mp3, .wav, .m4a")
        exit(1)
    
    try:
        transcription = transcribe_long_audio(audio_file_path, model, processor, device)
        
        print("\n" + "="*80)
        print("📝 FULL TRANSCRIPTION:")
        print("="*80)
        print(transcription)
        print("="*80)
        
        transcription_file = "transcription_output.txt"
        with open(transcription_file, "w") as f:
            f.write(transcription)
        print(f"\n💾 Transcription saved to: {transcription_file}")
        
        meeting_minutes = generate_meeting_minutes(transcription)
        
        if meeting_minutes:
            print("\n" + "="*80)
            print("📋 MEETING MINUTES:")
            print("="*80)
            print(meeting_minutes)
            print("="*80)
            
            minutes_file = "meeting_minutes.txt"
            with open(minutes_file, "w") as f:
                f.write(meeting_minutes)
            print(f"\n💾 Meeting minutes saved to: {minutes_file}")
        
        # Clean up
        if file_extension == '.mp4' and audio_file_path and os.path.exists(audio_file_path):
            try:
                os.remove(audio_file_path)
                print(f"🗑️ Cleaned up temporary audio file: {audio_file_path}")
            except:
                print(f"⚠️ Could not remove temporary file: {audio_file_path}")
        
    except Exception as e:
        print(f"❌ Error during processing: {e}")