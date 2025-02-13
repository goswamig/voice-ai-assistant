import os
import numpy as np
import sounddevice as sd
import wave
import subprocess
from pydub import AudioSegment
import torch
from TTS.api import TTS

# Configurations
WHISPER_PATH = "./whisper.cpp/build/bin/whisper-cli"
WHISPER_MODEL = "./whisper.cpp/models/ggml-large-v3.bin"  # More accurate model
LLAMA_PATH = "./llama.cpp/build/bin/llama-cli"
LLAMA_MODEL = "./llama.cpp/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"  # Best mix of speed & quality
AUDIO_FILE = "recorded.wav"

# Function to Record Audio
def record_audio(duration=5, sample_rate=16000):
    print("Recording...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype=np.int16)
    sd.wait()
    print("Recording complete.")

    with wave.open(AUDIO_FILE, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio.tobytes())

# Function to Convert Speech to Text
def speech_to_text():
    cmd = f"{WHISPER_PATH} -m {WHISPER_MODEL} -f {AUDIO_FILE} -l en --output-txt --threads 10"
    subprocess.run(cmd, shell=True)
    with open(AUDIO_FILE + ".txt", "r") as f:
        return f.read().strip()

# Function to Generate AI Response
def generate_response(prompt):
    cmd = f'echo "{prompt}" | {LLAMA_PATH} -m {LLAMA_MODEL} --temp 0.7 --n-predict 200 --threads 10'
    output = subprocess.check_output(cmd, shell=True).decode()
    return output.split("### Assistant:")[-1].strip()

# Faster Text to Speech
def text_to_speech(text):
    os.system(f'edge-tts --text "{text}" --voice en-US-Wavenet-D --out response.mp3')
    os.system("afplay response.mp3")  # macOS built-in player

# Main Assistant Loop
def main_loop():
    while True:
        try:
            record_audio()
            text = speech_to_text()
            print(f"You said: {text}")

            response = generate_response(text)
            print(f"Assistant: {response}")

            text_to_speech(response)

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main_loop()
