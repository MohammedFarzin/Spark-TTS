
import argparse
import asyncio
import json
import queue  # Added
import uuid  # Added
import functools # Added
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from pydantic import BaseModel, Field

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


class TTSRequest(BaseModel):
    reference_audio_path: str # Or maybe base64 encoded audio data
    reference_text: str
    target_text: str
    stream_id: str # To identify the output

app = FastAPI()

celery_app = Celery('tasks', broker='redis://localhost:6379/0', backend="redis://localhost:6379/0")
celery_app.conf.task_queues = {
    'voice_queue': {
        'exchange': 'voice_queue',
        'routing_key': 'voice_queue',
    }
}


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
        from scipy.signal import resample

        num_samples = int(len(waveform) * (target_sample_rate / sample_rate))
        waveform = resample(waveform, num_samples)
    return waveform, target_sample_rate



@celery_app.task(name="celery_app.process_streaming_request")
def process_streaming_request(
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

        print(f"{stream_id}: Starting streaming processing for {text} items.")
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

            audio_save_path = os.path.join(audio_save_dir, f"{item['target_audio_path']}.wav")

            total_request_latency, first_chunk_latency, actual_duration = await asyncio.to_thread(
                run_sync_streaming_inference,
                sync_triton_client,
                model_name,
                inputs,
                outputs,
                request_id,
                user_data,
                chunk_overlap_duration,
                save_sample_rate,
                audio_save_path
            )

            if total_request_latency is not None:
                print(f"{name}: Item {i} - First Chunk Latency: {first_chunk_latency:.4f}s, Total Latency: {total_request_latency:.4f}s, Duration: {actual_duration:.4f}s")
                latency_data.append((total_request_latency, first_chunk_latency, actual_duration))
                total_duration += actual_duration
            else:
                    print(f"{name}: Item {i} failed.")


        except FileNotFoundError:
            print(f"Error: Audio file not found for item {i}: {item['audio_filepath']}")
        except Exception as e:
            print(f"Error processing item {i} ({item['target_audio_path']}): {e}")
            import traceback
            traceback.print_exc()


    finally: # Ensure client is closed
        if sync_triton_client:
            try:
                print(f"{name}: Closing sync client...")
                sync_triton_client.close()
            except Exception as e:
                print(f"{name}: Error closing sync client: {e}")


    print(f"{name}: Finished streaming processing. Total duration synthesized: {total_duration:.4f}s")



    

    


@app.websocket("/ws_tts")
async def handle_tts_websocket(websocket: WebSocket):
    """Handles incoming TTS requests via WebSocket."""
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            try:
                request_data = json.loads(data)
                request_model = TTSRequest(**request_data)

                # Inform client processing started
                await websocket.send_json({"status": "processing", "target_audio_id": request_model.stream_id})

                # Process the request
                # NOTE: This blocks processing further messages on this specific


                result = celery_app.send_task(
                    'tasks.process_streaming_request',
                    args=[request_model.stream_id, request_model.target_text, request_model.reference_audio_path, request_model.reference_text, ],
                    queue='voice_queue'
                )
                if result:
                    await websocket.send_json({"status": "success", "target_audio_id": request_model.target_audio_id, "data": result})
                else:
                    await websocket.send_json({"status": "error", "target_audio_id": request_model.target_audio_id, "message": "Processing failed"})

            except json.JSONDecodeError:
                await websocket.send_json({"status": "error", "message": "Invalid JSON received."})
            except Exception as e: # Catch validation errors etc.
                await websocket.send_json({"status": "error", "message": f"Error processing request: {str(e)}"})

    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"WebSocket Error: {e}")
        # Attempt to close gracefully
        await websocket.close(code=1011) # Internal Error



# --- Main execution block to run the server ---
if __name__ == "__main__":
    print("Starting FastAPI server...")
    # Run the FastAPI app using Uvicorn
    # host="0.0.0.0" makes it accessible on your network
    # reload=True is useful for development (auto-restarts on code changes)
    uvicorn.run("triton_server_code:app", host="0.0.0.0", port=8000, reload=True) # Change port if needed