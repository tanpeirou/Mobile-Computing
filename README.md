1. Download Ubuntu 26.04

2. Setup:
sudo apt update && sudo apt upgrade -y
sudo apt-get install portaudio19-dev python3-pyaudio # For microphone input
sudo apt install python3.14-venv
python3 -m venv .env
source .env/bin/activate
pip install useful-moonshine-onnx transformers torch numpy silero-vad
sudo apt update && sudo apt install python3.14-dev -y
pip install pyaudio numpy silero-vad

4. Download refiner.py
5. Download live_pipeline.py
6. Run in environment
   python3 live_pipeline.py
