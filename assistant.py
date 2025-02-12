import os
import sounddevice as sd
import numpy as np
from pydub import AudioSegment
from openai import OpenAI
import requests
import io

# API Configurations
ASSEMBLYAI_API_KEY = os.getenv('ASSEMBLYAI_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
ELEVENLABS_API_KEY = os.getenv('ELEVENLABS_API_KEY')
ELEVENLABS_VOICE_ID = "21m00Tcm4TlvDq8ikWAM"  # Default voice ID

# Initialize clients
openai_client = OpenAI(api_key=OPENAI_API_KEY)

def record_audio(duration=5, sample_rate=16000):
    """Record audio from microphone"""
    print("Recording...")
    audio = sd.rec(int(duration * sample_rate), 
                  samplerate=sample_rate, 
                  channels=1, 
                  dtype=np.int16)
    sd.wait()
    print("Recording complete")
    return audio

def save_to_mp3(audio, filename="input.mp3"):
    """Save numpy array to MP3"""
    audio = np.int16(audio * 32767).flatten()
    audio_segment = AudioSegment(
        audio.tobytes(),
        frame_rate=16000,
        sample_width=2,
        channels=1
    )
    audio_segment.export(filename, format="mp3")
    return filename

def speech_to_text(audio_file):
    """Convert speech to text using AssemblyAI"""
    headers = {
        "authorization": ASSEMBLYAI_API_KEY,
        "content-type": "application/json"
    }
    
    # Upload audio
    upload_url = "https://api.assemblyai.com/v2/upload"
    with open(audio_file, "rb") as f:
        response = requests.post(upload_url, headers=headers, data=f)
    audio_url = response.json()["upload_url"]

    # Transcribe
    transcript_endpoint = "https://api.assemblyai.com/v2/transcript"
    json = {"audio_url": audio_url}
    response = requests.post(transcript_endpoint, json=json, headers=headers)
    transcript_id = response.json()["id"]

    # Poll for completion
    polling_endpoint = f"https://api.assemblyai.com/v2/transcript/{transcript_id}"
    while True:
        response = requests.get(polling_endpoint, headers=headers)
        status = response.json()["status"]
        if status == "completed":
            return response.json()["text"]
        elif status == "error":
            raise Exception("Transcription failed")

def generate_response(prompt):
    """Generate LLM response using OpenAI"""
    response = openai_client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Keep responses concise."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150
    )
    return response.choices[0].message.content

def text_to_speech(text):
    """Convert text to speech using ElevenLabs"""
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}"
    
    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json"
    }
    
    data = {
        "text": text,
        "model_id": "eleven_monolingual_v1",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.5
        }
    }
    
    response = requests.post(url, json=data, headers=headers)
    
    if response.status_code == 200:
        # Play audio directly
        audio = AudioSegment.from_file(io.BytesIO(response.content), format="mp3")
        playback = sd.play(np.array(audio.get_array_of_samples()), 
                          samplerate=audio.frame_rate)
        sd.wait()
    else:
        print("TTS Error:", response.text)

def main_loop():
    """Main interaction loop"""
    while True:
        try:
            # Step 1: Record voice
            audio = record_audio()
            
            # Step 2: Save and transcribe
            audio_file = save_to_mp3(audio)
            text = speech_to_text(audio_file)
            print(f"You said: {text}")
            
            # Step 3/4: Generate response
            response = generate_response(text)
            print(f"Assistant: {response}")
            
            # Step 5: Convert and play response
            text_to_speech(response)
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    # Verify dependencies
    try:
        import sounddevice
    except ImportError:
        print("Please install required packages:")
        print("pip install sounddevice numpy pydub requests openai python-dotenv")
    
    # Start assistant
    main_loop()