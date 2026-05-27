import threading
import queue
import time
import numpy as np
import pyaudio
import moonshine_onnx as moonshine
from refiner import RealLocalLLM

# Thread-safe queues for async cross-talk
audio_chunk_queue = queue.Queue()
ui_display_queue = queue.Queue()

# Constants for standard 16kHz audio required by Moonshine
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK_SIZE = 512  # Number of frames read per step

# Initialize models globally
print("[Initialization] Loading real local LLM (Qwen 2.5 4-bit) into memory...")
local_llm_engine = RealLocalLLM(model_path="qwen2.5-1.5b-instruct-q4_k_m.gguf")

# ========================================================
# 1. LIVE MICROPHONE PRODUCER THREAD
# ========================================================
def mic_stream_producer():
    """Continuously records raw PCM audio from the mic to a queue."""
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                    input=True, frames_per_buffer=CHUNK_SIZE)
    
    print("\n>>> MICROPHONE ACTIVE: Start speaking now... (Press Ctrl+C to stop)")
    
    while True:
        try:
            # Read raw bytes from hardware sound card
            raw_data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
            # Convert raw bytes to standard float32 normalized vectors (-1.0 to 1.0)
            audio_data = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0
            audio_chunk_queue.put(audio_data)
        except Exception as e:
            print(f"Hardware Recording Error: {e}")
            break

# ========================================================
# 2. CORE DUAL-PATH PROCESSING PIPELINE
# ========================================================
def evaluate_cascade_gate(raw_text):
    """Linguistic gate checking for empty data or severe truncations."""
    words = raw_text.split()
    if len(words) == 0:
        return False, 1.0
        
    # Heuristic Signal Proxy: Flag highly suspicious lengths or known error fragments
    is_fragment = len(words) <= 2 or "black" in raw_text.lower()
    confidence = 0.50 if is_fragment else 0.95
    
    if confidence < 0.88:
        return True, confidence
    return False, confidence

def process_live_utterance(audio_frames):
    """Runs ASR immediately and hands off to the LLM if the gate fails."""
    start_time = time.time()
    
    # Concatenate the accumulated frames into a single continuous array
    full_audio = np.concatenate(audio_frames)
    
    # 1. FAST PATH (Run real Moonshine inference directly on memory vector)
    # moonshine.transcribe accepts file paths or raw numpy arrays directly
    segments = moonshine.transcribe(full_audio, "moonshine/tiny")
    raw_text = segments[0] if segments else ""
    asr_latency = time.time() - start_time
    
    if not raw_text.strip():
        return # Skip empty silences
        
    # Send speculative output to live display instantly
    ui_display_queue.put({
        "type": "SPECULATIVE",
        "text": raw_text,
        "latency": asr_latency
    })
    
    # 2. EVALUATE CASCADE GATE
    should_refine, score = evaluate_cascade_gate(raw_text)
    
    if should_refine:
        # Pass to the slow path background thread to prevent live audio dropouts
        threading.Thread(
            target=slow_path_refinement, 
            args=(raw_text, start_time)
        ).start()
    else:
        ui_display_queue.put({"type": "FINAL_BYPASS", "text": raw_text})

def slow_path_refinement(raw_text, initial_start_time):
    """Background context optimization worker."""
    prompt = f"Fix speech-to-text grammar and homophone errors: {raw_text}"
    refined_text = local_llm_engine.generate(prompt)
    total_latency = time.time() - initial_start_time
    
    ui_display_queue.put({
        "type": "REFINED",
        "text": refined_text,
        "latency": total_latency
    })

# ========================================================
# 3. VAD UTTERANCE PACKAGER & CHUNKER
# ========================================================
def audio_processing_orchestrator():
    """Assembles stream frames into discrete semantic sentences using a time window."""
    accumulated_frames = []
    silence_threshold_bytes = 15  # Tweak this value to adjust voice sensitivity
    silent_chunks_count = 0
    is_speaking = False
    
    while True:
        # Pull 30ms frame from microphone input queue
        frame = audio_chunk_queue.get()
        accumulated_frames.append(frame)
        
        # Super-lightweight energy detector to act as basic VAD segmentation
        energy = np.linalg.norm(frame)
        if energy > 0.3:  # Voice activity active
            is_speaking = True
            silent_chunks_count = 0
        else:
            if is_speaking:
                silent_chunks_count += 1
                
        # If the user pauses speaking for roughly ~600ms, process the accumulated block
        if is_speaking and silent_chunks_count > 20:
            # Dispatch accumulated memory frames for transcription
            threading.Thread(target=process_live_utterance, args=(list(accumulated_frames),)).start()
            # Clear window buffer for the next incoming sentence
            accumulated_frames.clear()
            is_speaking = False
            silent_chunks_count = 0

# ========================================================
# 4. LIVE INTERFACE CONSUMER
# ========================================================
def run_live_display():
    """Simulates rolling live display captions updating dynamically."""
    print("--- Live Interface Captions Initialized ---")
    current_line_id = 0
    
    while True:
        message = ui_display_queue.get()
        
        if message["type"] == "SPECULATIVE":
            print(f"\r[LIVE CAPTION (Fast Path)]: {message['text']} ({message['latency']:.2f}s)", end="")
        elif message["type"] == "REFINED":
            # Overwrite the speculative line smoothly once the LLM finishes context correction
            print(f"\r[LIVE CAPTION (LLM Refined)]: {message['text']} (Total: {message['latency']:.2f}s)\n")
        elif message["type"] == "FINAL_BYPASS":
            print(" -> [Finalized (Bypass)]\n")
            
        ui_display_queue.task_done()

# ========================================================
# MAIN COORDINATION FRAME
# ========================================================
if __name__ == "__main__":
    # Thread A: Hardware Microphone Audio Ingestion
    mic_thread = threading.Thread(target=mic_stream_producer, daemon=True)
    mic_thread.start()
    
    # Thread B: Stream Orchestrator & Feature Segmentation
    orchestrator_thread = threading.Thread(target=audio_processing_orchestrator, daemon=True)
    orchestrator_thread.start()
    
    # Thread C: Asynchronous UI State Flow Manager
    display_thread = threading.Thread(target=run_live_display, daemon=True)
    display_thread.start()
    
    # Keep the main process thread alive indefinitely
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[System Shutdown] Live pipeline terminated successfully.")

