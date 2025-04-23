
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
from Logger import Log
import traceback

logger = Log("fastapi.log")
logger = logger.initialize_logger_handler()

class TTSRequest(BaseModel):
    reference_audio_path: str # Or maybe base64 encoded audio data
    reference_text: str
    target_text: str
    stream_id: str # To identify the output

app = FastAPI()

celery_app = Celery('celery_app', broker='redis://redis:6379/0', backend="redis://redis:6379/0") 



@app.websocket("/ws_tts")
async def handle_tts_websocket(websocket: WebSocket):
    """Handles incoming TTS requests via WebSocket."""
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            try:
                request_data = json.loads(data)
                logger.info(f"Received request: {request_data}")
                request_model = TTSRequest(**request_data)


                # Inform client processing started
                # await websocket.send_json({"status": "processing", "target_audio_id": request_model.stream_id})

                # Process the request
                # NOTE: This blocks processing further messages on this specific


                result = celery_app.send_task(
                    'tasks.process_streaming_request',
                    args=[request_model.stream_id, request_model.target_text, request_model.reference_audio_path, request_model.reference_text, ],
                    queue='voice_queue'
                )
                if result:
                    logger.info(f"Processing result: {result}")
                    await websocket.send_json({"status": "success", "target_audio_id": request_model.stream_id, "data": result})
                else:
                    logger.error(f"Processing failed for request: {request_model}")
                    await websocket.send_json({"status": "error", "target_audio_id": request_model.stream_id, "message": "Processing failed"})

            except json.JSONDecodeError:
                logger.error("Invalid JSON received.")
                await websocket.send_json({"status": "error", "message": "Invalid JSON received."})
            except Exception as e: # Catch validation errors etc.
                logger.error(f"Error processing request: {traceback.format_exc()}")
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
    uvicorn.run(app, host="0.0.0.0", port=8000) # Change port if needed