FROM python:3.10

RUN apt-get update && apt-get install -y cmake
RUN pip install einx==0.3.0 omegaconf==2.3.0 soundfile==0.12.1 soxr==0.5.0.post1 gradio tritonclient librosa fastapi
WORKDIR /workspace

COPY requirements.txt .
COPY . .
RUN pip install -r requirements.txt


CMD ["python", "-m", "triton_server_code"]



