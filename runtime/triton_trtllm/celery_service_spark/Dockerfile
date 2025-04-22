FROM nvcr.io/nvidia/tritonserver:25.02-trtllm-python-py3
RUN apt-get update && apt-get install -y cmake
RUN git clone https://github.com/pytorch/audio.git && cd audio && git checkout c670ad8 && PATH=/usr/local/cuda/bin:$PATH python3 setup.py develop
RUN pip install einx==0.3.0 omegaconf==2.3.0 soundfile==0.12.1 soxr==0.5.0.post1 gradio tritonclient librosa fastapi
WORKDIR /workspace

COPY requirements.txt .

COPY . .
RUN pip install -r requirements.txt


CMD ["celery", "-A", "spark:celery_app", "worker", "--autoscale=3,12", "--loglevel=info", "-Q", "voice_queue"]



