# Voice AI Assistant
Pre-requisite: Build llama.cpp and whisper.cpp and download the models for them before running below commands. This doc covers that portion later.
1) Run whisper server ./whisper.cpp/build/bin/whisper-server -m ./whisper.cpp/models/ggml-large-v3.bin in a terminal.
2) Run llama server ./llama.cpp/build/bin/llama-server -m ./llama.cpp/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf -c 2048 --port 8081 
2) Run python main.py in another terminal 

## TODO 
1) Current recoding is for 5 seconds make it as if user stops saying stop recording 
2) Make it work in ur own voice

## Get API Keys:

- [AssemblyAI](https://www.assemblyai.com/) (Free tier available)
- [OpenAI](https://platform.openai.com/docs/overview)
- [ElevenLabs](https://elevenlabs.io/) (Free tier available)

## Set Environment Variables:

```sh
export ASSEMBLYAI_API_KEY='your_key'
export OPENAI_API_KEY='your_key'
export ELEVENLABS_API_KEY='your_key'
```

### Install Required libraries 
```sh
pip install -r requirements.txt
```



# Install dependencies  
brew install ffmpeg portaudio
pip install numpy sounddevice pydub torch torchaudio coqui-tts


# Download & Setup Whisper.cpp (for STT):
git clone https://github.com/ggerganov/whisper.cpp.git
cd whisper.cpp
WHISPER_METAL=1 make   # Enable Metal acceleration
bash ./models/download-ggml-model.sh large-v3  # Use larger model for better transcription


# Download & Setup llama.cpp (for LLM):
    # 1. Clone the repository
    git clone https://github.com/ggerganov/llama.cpp.git
    cd llama.cpp

    # 2. Update the repo (if already cloned)
    git pull

    # 3. Remove old builds if they exist
    rm -rf build

    # 4. Create and move into the build directory
    mkdir build && cd build

    # 5. Run CMake with Metal support enabled
    cmake -DLLAMA_METAL=on ..

    # 6. Build the project using all available CPU cores
    cmake --build . --parallel

# download Faster TTS 
pip install edge-tts  # Alternative to Coqui TTS


