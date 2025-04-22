import argparse
import asyncio
import json
import queue  # Added
import uuid  # Added
import functools # Added
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from pydantic import BaseModel, Field
from scipy.signal import resample

from celery import Celery

import os
import time
import types
from pathlib import Path

import numpy as np
import soundfile as sf
import tritonclient
import tritonclient.grpc.aio as grpcclient_aio # Renamed original import
import tritonclient.grpc as grpcclient_sync # Added sync client import
from tritonclient.utils import np_to_triton_dtype, InferenceServerException # Added InferenceServerException

class UserData:
    def __init__(self):
        self._completed_requests = queue.Queue()
        self._first_chunk_time = None
        self._start_time = None

    def record_start_time(self):
        self._start_time = time.time()

    def get_first_chunk_latency(self):
        if self._first_chunk_time and self._start_time:
            return self._first_chunk_time - self._start_time
        return None
class TTSRequest(BaseModel):
    reference_audio_path: str # Or maybe base64 encoded audio data
    reference_text: str
    target_text: str
    stream_id: str # To identify the output

app = FastAPI()

celery_app = Celery('tasks', broker='redis://redis:6379/0', backend="redis://redis:6379/0")
celery_app.conf.task_queues = {
    'voice_queue': {
        'exchange': 'voice_queue',
        'routing_key': 'voice_queue',
    }
}


def callback(user_data, result, error):
    if user_data._first_chunk_time is None and not error:
        user_data._first_chunk_time = time.time() # Record time of first successful chunk
    if error:
        user_data._completed_requests.put(error)
    else:
        user_data._completed_requests.put(result)
# --- End Added UserData and callback ---


def run_sync_streaming_inference(
    sync_triton_client: tritonclient.grpc.InferenceServerClient,
    model_name: str,
    inputs: list,
    outputs: list,
    request_id: str,
    user_data: UserData,
    chunk_overlap_duration: float,
    save_sample_rate: int,
):
    """Helper function to run the blocking sync streaming call."""
    start_time_total = time.time()
    user_data.record_start_time() # Record start time for first chunk latency calculation

    # Establish stream
    sync_triton_client.start_stream(callback=functools.partial(callback, user_data))

    # Send requestprotocol_client
    sync_triton_client.async_stream_infer(
        model_name,
        inputs,
        request_id=request_id,
        outputs=outputs,
        enable_empty_final_response=True,
    )

    # Process results
    audios = []
    audio_bytes = b""
    while True:
        try:
            result = user_data._completed_requests.get() # Add timeout
            if isinstance(result, InferenceServerException):
                print(f"Received InferenceServerException: {result}")
                sync_triton_client.stop_stream()
                return None, None, None # Indicate error
            # Get response metadata
            response = result.get_response()
            final = response.parameters["triton_final_response"].bool_param
            if final is True:
                break

            audio_chunk = result.as_numpy("waveform").reshape(-1)
            if audio_chunk.size > 0: # Only append non-empty chunks
                wav = audio_chunk.clone().detach().cpu().numpy()
                wav = wav[None, : int(wav.shape[0])]
                wav = np.clip(wav, -1, 1)
                wav = (wav * 32767).astype(np.int16)
                audio_float = wav.astype(np.float32) / 32767.0
                ratio = 8000 / 16000
                output_samples = int(len(audio_float) * ratio)
                
                # Resample the audio
                resampled_audio = resample(audio_float, output_samples)
                audio_bytes = (np.clip(resampled_audio, -1, 1) * 32767).astype(np.int16).tobytes()



                audios.append(audio_bytes)
            else:
                print("Warning: received empty audio chunk.")

        except queue.Empty:
            print(f"Timeout waiting for response for request id {request_id}")
            sync_triton_client.stop_stream()
            return None, None, None # Indicate error

    sync_triton_client.stop_stream()
    end_time_total = time.time()
    total_request_latency = end_time_total - start_time_total
    first_chunk_latency = user_data.get_first_chunk_latency()

    # Reconstruct audio using cross-fade (from client_grpc_streaming.py)
    actual_duration = 0
    if audios:
        cross_fade_samples = int(chunk_overlap_duration * save_sample_rate)
        fade_out = np.linspace(1, 0, cross_fade_samples)
        fade_in = np.linspace(0, 1, cross_fade_samples)
        reconstructed_audio = None

        # Simplified reconstruction based on client_grpc_streaming.py
        if not audios:
            print("Warning: No audio chunks received.")
            reconstructed_audio = np.array([], dtype=np.float32) # Empty array
        elif len(audios) == 1:
            reconstructed_audio = audios[0]
        else:
            reconstructed_audio = audios[0][:-cross_fade_samples] # Start with first chunk minus overlap
            for i in range(1, len(audios)):
                 # Cross-fade section
                 cross_faded_overlap = (audios[i][:cross_fade_samples] * fade_in +
                                        audios[i - 1][-cross_fade_samples:] * fade_out)
                 # Middle section of the current chunk
                 middle_part = audios[i][cross_fade_samples:-cross_fade_samples]
                 # Concatenate
                 reconstructed_audio = np.concatenate([reconstructed_audio, cross_faded_overlap, middle_part])
            # Add the last part of the final chunk
            reconstructed_audio = np.concatenate([reconstructed_audio, audios[-1][-cross_fade_samples:]])

        if reconstructed_audio is not None and reconstructed_audio.size > 0:
            actual_duration = len(reconstructed_audio) / save_sample_rate
            # Save reconstructed audio
            os.makedirs(os.path.dirname("audio_save_path"), exist_ok=True)
            sf.write("audio_save_path", reconstructed_audio, save_sample_rate, "PCM_16")
        else:
            print("Warning: No audio chunks received or reconstructed.")
            actual_duration = 0 # Set duration to 0 if no audio

    else:
         print("Warning: No audio chunks received.")
         actual_duration = 0

    return total_request_latency, first_chunk_latency, actual_duration


def prepare_request_input_output(
    protocol_client,
    waveform,
    reference_text,
    target_text,
    sample_rate=16000,
    padding_duration=None
):
    assert len(waveform.shape) == 1, "waveform should be 1D"
    lengths = np.array([[len(waveform)]], dtype=np.int32)

    if padding_duration:
        duration = len(waveform) / sample_rate

        if reference_text:
           estimated_target_duration = duration / len(reference_text) * len(target_text)
        else:
           estimated_target_duration = duration

        required_total_samples = padding_duration * sample_rate * (
            (int(estimated_target_duration + duration) // padding_duration) + 1
        )
        samples = np.zeros((1, required_total_samples), dtype=np.float32)
        samples[0, : len(waveform)] = waveform
    else:
        samples = waveform.reshape(1, -1).astype(np.float32)
    
    # Common input creation logic
    inputs = [
        protocol_client.InferInput("reference_wav", samples.shape, np_to_triton_dtype(samples.dtype)),
        protocol_client.InferInput(
            "reference_wav_len", lengths.shape, np_to_triton_dtype(lengths.dtype)
        ),
        protocol_client.InferInput("reference_text", [1, 1], "BYTES"),
        protocol_client.InferInput("target_text", [1, 1], "BYTES"),
    ]
    inputs[0].set_data_from_numpy(samples)
    inputs[1].set_data_from_numpy(lengths)

    input_data_numpy = np.array([reference_text], dtype=object)
    input_data_numpy = input_data_numpy.reshape((1, 1))
    inputs[2].set_data_from_numpy(input_data_numpy)

    input_data_numpy = np.array([target_text], dtype=object)
    input_data_numpy = input_data_numpy.reshape((1, 1))
    inputs[3].set_data_from_numpy(input_data_numpy)

    outputs = [protocol_client.InferRequestedOutput("waveform")]

    return inputs, outputs





def load_audio(wav_path, target_sample_rate=16000):
    assert target_sample_rate == 16000, "hard coding in server"
    if isinstance(wav_path, dict):
        waveform = wav_path["array"]
        sample_rate = wav_path["sampling_rate"]
    else:
        waveform, sample_rate = sf.read(wav_path)
    if sample_rate != target_sample_rate:

        num_samples = int(len(waveform) * (target_sample_rate / sample_rate))
        waveform = resample(waveform, num_samples)
    with open("defualt_speaker.json", "r") as f:
        f.write(json.dumps({"speaker": "waveform"}))
    return waveform, target_sample_rate



async def process_streaming_request(
    stream_id: str,
    target_text: str,
    audio: str,
    reference_text: str,
    audio_save_dir: str = "./",
    save_sample_rate: int = 16000,
    chunk_overlap_duration: float = 0.1,
    padding_duration: int = 10):

    """Processes the TTS request and returns the result."""
    # websocket until `process_streaming_request` completes.
    protocol_client = grpcclient_sync
    server_url = "localhost:8001" # Hardcoded for now
    model_name = "spark_tts"

    total_duration = 0.0
    latency_data = []
    sync_triton_client = None # Initialize client variable

    try: # Wrap in try...finally to ensure client closing
        print(f"{stream_id}: Initializing sync client for streaming...")
        sync_triton_client = grpcclient_sync.InferenceServerClient(url=server_url, verbose=True) # Create client here

        print(f"{stream_id}: Starting streaming processing for {target_text} items.")
        try:
            waveform, sample_rate = load_audio(audio, target_sample_rate=16000)
            inputs, outputs = prepare_request_input_output(
                protocol_client,
                waveform,
                reference_text,
                target_text,
                sample_rate,
                padding_duration=padding_duration
            )
            request_id = str(uuid.uuid4())
            user_data = UserData()


            total_request_latency, first_chunk_latency, actual_duration = await asyncio.to_thread(
                run_sync_streaming_inference,
                sync_triton_client,
                model_name,
                inputs,
                outputs,
                request_id,
                user_data,
                chunk_overlap_duration,
                save_sample_rate
            )

            if total_request_latency is not None:
                print(f"{stream_id}: First Chunk Latency: {first_chunk_latency:.4f}s, Total Latency: {total_request_latency:.4f}s, Duration: {actual_duration:.4f}s")
                latency_data.append((total_request_latency, first_chunk_latency, actual_duration))
                total_duration += actual_duration
            else:
                    print(f"{stream_id}: failed.")


        except Exception as e:
            print(f"{stream_id}: Error during streaming inference: {e}")
            return None, None, None


    finally: # Ensure client is closed
        if sync_triton_client:
            try:
                print(f"{target_text}: Closing sync client...")
                sync_triton_client.close()
            except Exception as e:
                print(f"{stream_id}: Error closing sync client: {e}")


    print(f"{stream_id}: Finished streaming processing. Total duration synthesized: {total_duration:.4f}s")


@celery_app.task(name="celery_app.process_streaming_request")
def spark_processing(    
    stream_id: str,
    target_text: str,
    audio: str,
    reference_text: str,
    audio_save_dir: str = "./",
    save_sample_rate: int = 16000,
    chunk_overlap_duration: float = 0.1,
    padding_duration: int = 10):

    try: 
        asyncio.run(process_streaming_request(
            stream_id=stream_id,
            target_text=target_text,
            audio=audio,
            reference_text=reference_text,
            audio_save_dir=audio_save_dir,
            save_sample_rate=save_sample_rate,
            chunk_overlap_duration=chunk_overlap_duration,
            padding_duration=padding_duration
        ))
    except Exception as e:
        print(f"Error in celery task: {e}")
        return None, None, None
    

    