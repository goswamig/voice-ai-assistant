import os
import sys
import numpy as np
import sounddevice as sd
import wave
import subprocess

# Configurations
WHISPER_PATH = "./whisper.cpp/build/bin/whisper-cli"
WHISPER_MODEL = "./whisper.cpp/models/ggml-large-v3.bin"  # More accurate model
LLAMA_PATH = "./llama.cpp/build/bin/llama-cli"
LLAMA_MODEL = "./llama.cpp/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"  # Best mix of speed & quality
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

# Function to Convert Speech to Text
def speech_to_text():
    cmd = f"{WHISPER_PATH} -m {WHISPER_MODEL} -f {AUDIO_FILE} -l en --output-txt --threads 10"
    print("Executing Whisper command:")
    print(cmd)
    sys.stdout.flush()
    subprocess.run(cmd, shell=True)
    
    txt_file = AUDIO_FILE + ".txt"
    if os.path.exists(txt_file):
        with open(txt_file, "r") as f:
            transcription = f.read().strip()
            return transcription
    return ""

# Function to Generate AI Response
def generate_response(prompt):
    # Use -no-cnv flag to disable conversation mode
    cmd = f'{LLAMA_PATH} -m {LLAMA_MODEL} -p "{prompt}" --temp 0.7 --n-predict 200 --threads 10 -no-cnv'
    print("Executing LLAMA command:")
    print(cmd)
    sys.stdout.flush()
    try:
        output = subprocess.check_output(cmd, shell=True, timeout=300).decode()
        print("LLAMA output:")
        print(output)
        sys.stdout.flush()
        # Remove performance logs and any debug text (if present)
        if "llama_perf_" in output:
            output = output.split("llama_perf_")[0].strip()
        if "[end of text]" in output:
            output = output.split("[end of text]")[0].strip()
        return output
    except subprocess.CalledProcessError as e:
        error_msg = e.output.decode()
        print("Error generating response:")
        print(error_msg)
        sys.stdout.flush()
        return "Sorry, I couldn't generate a response."
    except subprocess.TimeoutExpired:
        print("LLAMA command timed out.")
        sys.stdout.flush()
        return "Sorry, the response generation timed out."

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
