import os
import sys
import numpy as np
import sounddevice as sd
import wave
import subprocess
import time
import requests
import json
from tortoise.api import TextToSpeech
from tortoise.utils.audio import load_audio, load_voice, load_voices
import re
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

# Configurations
WHISPER_SERVER_URL = "http://127.0.0.1:8080/inference"  # URL of your Whisper server
LLAMA_SERVER_URL = "http://127.0.0.1:8081/completion"  # URL of your Llama server
AUDIO_FILE = "recorded.wav"
OUTPUT_AUDIO = "response.mp3"
VOICE_SAMPLES_DIR = "voice_samples"  # Directory to store your voice samples
CONDITIONING_LATENTS = None  # Will store your voice characteristics

# clone the voice 
def setup_voice_clone():
    """
    Set up the voice cloning model using your voice samples.
    Requires multiple short audio clips of your voice for better quality.
    """
    global CONDITIONING_LATENTS
    
    if not os.path.exists(VOICE_SAMPLES_DIR):
        os.makedirs(VOICE_SAMPLES_DIR)
        print(f"Created voice samples directory at {VOICE_SAMPLES_DIR}")
        print("Please add 3-5 short WAV recordings of your voice (2-5 seconds each)")
        print("Each file should be clear speech with minimal background noise")
        return False
    
    try:
        # Initialize Tortoise TTS
        tts = TextToSpeech()
        
        # Load your voice samples
        voice_samples = []
        for file in os.listdir(VOICE_SAMPLES_DIR):
            if file.endswith('.wav'):
                sample_path = os.path.join(VOICE_SAMPLES_DIR, file)
                audio = load_audio(sample_path, 22050)
                voice_samples.append(audio)
        
        if not voice_samples:
            print("No voice samples found. Please add WAV files to the voice_samples directory.")
            return False
        
        # Generate voice conditioning latents
        CONDITIONING_LATENTS = tts.get_conditioning_latents(voice_samples)
        print("Voice clone model successfully created!")
        return True
        
    except Exception as e:
        print(f"Error setting up voice clone: {e}")
        return False

def split_into_sentences(text):
    """Split text into sentences and further into chunks if sentences are too long."""
    # First, split into sentences
    sentences = sent_tokenize(text)
    chunks = []
    
    for sentence in sentences:
        # If sentence is too long, split it by punctuation
        if len(sentence) > 150:
            parts = re.split('[,;:]', sentence)
            # Filter out empty parts and strip whitespace
            parts = [p.strip() for p in parts if p.strip()]
            chunks.extend(parts)
        else:
            chunks.append(sentence)
    
    return chunks

# Function to Record Audio
def record_audio(
    sample_rate=16000,
    silence_threshold_db=-40,
    silence_duration=3.0,
    max_duration=60.0,
    chunk_duration=0.5
):
    """
    Records audio until silence is detected or max duration is reached.
    
    Args:
        sample_rate (int): Audio sample rate.
        silence_threshold_db (float): Silence threshold in decibels.
        silence_duration (float): Seconds of silence to trigger stop.
        max_duration (float): Maximum recording time in seconds.
        chunk_duration (float): Duration of each audio chunk to analyze.
    """
    print("Recording... (Press Ctrl+C to stop early)")
    sys.stdout.flush()

    chunk_size = int(chunk_duration * sample_rate)
    recorded_chunks = []
    last_sound_time = time.time()
    start_time = time.time()

    with sd.InputStream(samplerate=sample_rate, channels=1, dtype=np.int16) as stream:
        while True:
            current_time = time.time()
            elapsed = current_time - start_time

            # Stop if max duration reached
            if elapsed >= max_duration:
                print("Maximum recording duration reached.")
                break

            # Read audio chunk
            data, _ = stream.read(chunk_size)
            if data.size == 0:
                continue

            # Store audio data
            recorded_chunks.append(data.flatten())

            # Calculate RMS and convert to dBFS
            samples = data.astype(np.float32) / 32768.0  # Normalize to [-1, 1]
            rms = np.sqrt(np.mean(samples ** 2))
            db = 20 * np.log10(rms) if rms > 0 else -np.inf

            # Silence detection logic
            if db < silence_threshold_db:
                if (current_time - last_sound_time) >= silence_duration:
                    print(f"Silence detected for {silence_duration} seconds.")
                    break
            else:
                last_sound_time = current_time

    # Combine and save audio
    audio = np.concatenate(recorded_chunks, axis=0)
    with wave.open(AUDIO_FILE, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio.tobytes())
    print(f"Recording saved to {AUDIO_FILE}")
    sys.stdout.flush()

def speech_to_text():
    with open(AUDIO_FILE, "rb") as audio_file:
        files = {"file": (AUDIO_FILE, audio_file, "audio/wav")}  # Correct file field name
        data = {"temperature": 0.0, "temperature_inc": 0.2, "response_format": "json"}  # Other parameters

        try:
            response = requests.post(WHISPER_SERVER_URL, files=files, data=data, timeout=300) # Include data parameters
            response.raise_for_status()  # Check for HTTP errors
            result = response.json()
            if result and "text" in result:
                transcription = result["text"].strip()
                return transcription
            else:
                print("Unexpected response from Whisper server:", result)
                return ""
        except requests.exceptions.RequestException as e:
            print(f"Error communicating with Whisper server: {e}")
            return ""
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON response from Whisper server: {e}")
            return ""


# Function to Generate AI Response
def generate_response(prompt):
    headers = {"Content-Type": "application/json"}
    data = {"prompt": prompt, "n_predict": 200}  # Adjust n_predict as needed

    try:
        response = requests.post(LLAMA_SERVER_URL, headers=headers, json=data, timeout=300)
        response.raise_for_status()  # Raise an exception for bad status codes
        result = response.json()
        if result and "content" in result:
            response_text = result["content"].strip()
            return response_text
        else:
            print("Unexpected response from LLaMa server:", result)
            return "Sorry, I couldn't generate a response."
    except requests.exceptions.RequestException as e:
        print(f"Error communicating with LLaMa server: {e}")
        return "Sorry, I couldn't generate a response."
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON response from LLaMa server: {e}")
        return "Sorry, I couldn't generate a response."


def text_to_speech(text):
    """Convert text to speech using your cloned voice, handling long responses."""
    if CONDITIONING_LATENTS is None:
        print("Voice clone not set up. Using default TTS...")
        # Fallback to edge-tts
        output_audio = "response.mp3"
        cmd = [
            "edge-tts",
            "--text", text,
            "--voice", "en-US-AvaNeural",
            "--write-media", output_audio
        ]
        subprocess.run(cmd, check=True)
        subprocess.run(["afplay", output_audio], check=True)
        return

    try:
        # Split the text into manageable chunks
        chunks = split_into_sentences(text)
        tts = TextToSpeech()
        
        # Process each chunk separately
        for i, chunk in enumerate(chunks):
            print(f"Processing chunk {i+1}/{len(chunks)}: {chunk}")
            
            # Generate speech for the chunk
            output_file = f"chunk_{i}.wav"
            gen_audio = tts.tts_with_preset(
                chunk,
                voice_samples=None,
                conditioning_latents=CONDITIONING_LATENTS,
                preset="fast"
            )
            
            # Save the chunk
            tts.save_audio(gen_audio, output_file)
            
            # Play the chunk
            subprocess.run(["afplay", output_file], check=True)
            
            # Clean up the temporary file
            os.remove(output_file)
            
    except Exception as e:
        print(f"Error in voice cloning TTS: {e}")
        # Fallback to edge-tts if voice cloning fails
        output_audio = "response.mp3"
        cmd = [
            "edge-tts",
            "--text", text,
            "--voice", "en-US-AvaNeural",
            "--write-media", output_audio
        ]
        subprocess.run(cmd, check=True)
        subprocess.run(["afplay", output_audio], check=True)

# Main Assistant Loop
def main_loop():
    """Main assistant loop with voice cloning."""
    # First, set up voice cloning
    if not setup_voice_clone():
        print("Voice cloning setup incomplete. Using default voice.")

    while True:
        try:
            record_audio()
            text = speech_to_text()
            if not text:
                print("No speech detected, please try again.")
                continue

            print(f"You said: {text}")
            sys.stdout.flush()

            print("Generating AI response...")
            sys.stdout.flush()
            response = generate_response(text)
            print(f"Assistant: {response}")
            sys.stdout.flush()

            text_to_speech(response)

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main_loop()
