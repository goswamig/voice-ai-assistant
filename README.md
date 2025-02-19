# Voice AI Assistant Setup Guide

This guide will help you set up and run the Voice AI Assistant, which uses `whisper.cpp` for speech-to-text and `llama.cpp` for generating AI responses.

This project sets up a voice assistant using `whisper.cpp` for speech-to-text (STT) and `llama.cpp` for generating responses. It also uses `edge-tts` for text-to-speech (TTS). You can run this locally on Mac-3 pro

## Prerequisites

Before you begin, ensure you have the following installed on your system:

*   **Python 3.10:**  It is recommended to use Python 3.10 for compatibility.
*   **Git:**  Required for cloning repositories.
*   **CMake:** Required to build `llama.cpp`.
*   **Build tools:**  Ensure you have the necessary build tools for compiling C++ code (like `build-essential` on Debian/Ubuntu, or Xcode command-line tools on macOS).
*   **FFmpeg and PortAudio:** Required for `sounddevice`. Install using your system's package manager (e.g., `brew install ffmpeg portaudio` on macOS or `sudo apt-get install ffmpeg libportaudio2 libportaudiocpp0` on Debian/Ubuntu).

## Setup Instructions

Follow these steps to set up the Voice AI Assistant:

### 1. Create a Python Virtual Environment

It's recommended to create a virtual environment to manage project dependencies.

```sh
python3.10 -m venv venv
source venv/bin/activate  # On Linux/macOS
```

### 2. Install Dependencies

```sh
pip install -r requirements.txt
brew install ffmpeg portaudio
```

### 3. Download & Build whisper.cpp (for Speech-to-Text)
Clone the repository, build it, and download a model:
```sh
git clone https://github.com/ggerganov/whisper.cpp.git
cd whisper.cpp
WHISPER_METAL=1 make  # Use Metal acceleration on macOS (optional)
bash ./models/download-ggml-model.sh large-v3  # Download a high-quality model
```

### 4. Download & Build llama.cpp (for Language Model)
Clone the repository, build it, and download a model:
```sh
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp

# Clean up any previous builds
git pull
rm -rf build

# Build with Metal support (if on macOS)
mkdir build && cd build
cmake -DLLAMA_METAL=on ..
cmake --build . --parallel
```
Download the model for llama.cpp
```sh
cd ../models
wget https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf
```

### 5. Start the Servers
Open separate terminal windows and start each server:

Start `whisper.cpp` server:
```sh
./whisper.cpp/build/bin/whisper-server -m ./whisper.cpp/models/ggml-large-v3.bin
```

Start `llama.cpp` server:
```sh
./llama.cpp/build/bin/llama-server -m ./llama.cpp/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf -c 2048 --port 8081
```

### 6. Verify the Servers
Check if the servers are working by running these commands in your terminal:

For `whisper.cpp`:
```sh
curl -X POST "http://127.0.0.1:8080/inference" -F "file=@recorded.wav"
```

for `llama.cpp`:
```sh
curl -X POST "http://127.0.0.1:8081/completion" -H "Content-Type: application/json" -d '{"prompt": "Hello, how are you?", "n_predict": 200}'
```

### 7. Run the Assistant
With both servers running, execute the main script:
```sh
python main.py
```

### Debugging & Optimization Tips 
#### 1. Out of Memory Errors 
* Consider using a smaller Llama model (e.g., `mistral-7b.Q2_K.gguf`).
* Lower the context size (change `-c 2048` to `-c 1024`) in the Llama server command.

#### 2. Improving Performance on Powerful Machines
* Offload computations to the GPU by adding the `-ngl 35` flag when starting the Llama server:
 ```sh
 ./llama.cpp/build/bin/llama-server -m ./llama.cpp/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf -c 2048 --port 8081 -ngl 35
 ```
* Build whisper.cpp with OpenBLAS for better performance:
```sh
WHISPER_OPENBLAS=1 make
```

#### 3. Enhancing Speed for llama.cpp
* Use a quantized GGUF model (e.g., `Q4_K_M` offers a good balance between speed and quality).
* Reduce the `n_predict` parameter in `main.py` to limit the response length.
