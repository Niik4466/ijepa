# Use the official PyTorch runtime base image with CUDA support
FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime

# Set working directory inside the container
WORKDIR /workspace

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install minimal additional dependencies required by I-JEPA and GPU monitoring
RUN pip install --no-cache-dir \
    pyyaml \
    submitit \
    nvidia-ml-py

# Copy repository code to the working directory
COPY . /workspace

# Set default entrypoint to run the training script
ENTRYPOINT ["python", "main.py"]
