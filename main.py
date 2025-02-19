import os
import sys
import numpy as np
import sounddevice as sd
import wave
import subprocess
import time
import requests
import json

# Configurations
WHISPER_SERVER_URL = "http://127.0.0.1:8080/inference"  # URL of your Whisper server
LLAMA_SERVER_URL = "http://127.0.0.1:8081/completion"  # URL of your Llama server
AUDIO_FILE = "recorded.wav"
OUTPUT_AUDIO = "response.mp3"

# Function to Record Audio
def record_audio(duration=5, sample_rate=16000):
    print("Recording...")
    sys.stdout.flush()
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype=np.int16)
    sd.wait()
    print("Recording complete.")
    sys.stdout.flush()
    with wave.open(AUDIO_FILE, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio.tobytes())


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


# Function to Convert Text to Speech and Play It
def text_to_speech(text):
    # Use --write-media instead of --out since edge-tts expects that argument.
    cmd = [
        "edge-tts",
        "--text", text,
        "--voice", "en-US-AvaNeural",
        "--write-media", OUTPUT_AUDIO
    ]
    print("Executing TTS command:")
    print(" ".join(cmd))
    sys.stdout.flush()
    try:
        subprocess.run(cmd, check=True)
        subprocess.run(["afplay", OUTPUT_AUDIO], check=True)
    except Exception as e:
        print("Error in TTS:")
        print(e)

# Main Assistant Loop
def main_loop():
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
