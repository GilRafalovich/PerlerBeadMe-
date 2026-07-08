# Use a PyTorch CUDA base image (Vertex AI compatible)
FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies required by OpenCV and TripoSR
RUN apt-get update && apt-get install -y git git-lfs libgl1-mesa-glx libglib2.0-0

WORKDIR /app

# Clone the official TripoSR repository to access the tsr module
RUN git clone https://github.com/VAST-AI-Research/TripoSR.git
WORKDIR /app/TripoSR

# Install model dependencies and FastAPI for the web server
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir fastapi uvicorn python-multipart

# Copy the custom Vertex AI prediction script into the TripoSR directory
COPY serve.py /app/TripoSR/serve.py

# Download the model weights during the Docker build process
# This ensures the Vertex AI endpoint can scale from zero without downloading 4GB every time.
RUN python -c "from tsr.system import TSR; TSR.from_pretrained('stabilityai/TripoSR', config_name='config.yaml', weight_name='model.ckpt')"

# Vertex AI Custom Endpoints default to listening on port 8080
EXPOSE 8080

CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8080"]
