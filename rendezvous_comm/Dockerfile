FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

WORKDIR /workspace

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy full repo (VMAS + rendezvous_comm)
COPY . /workspace/

# Install VMAS in dev mode + rendezvous_comm deps
RUN pip install --no-cache-dir -e . && \
    pip install --no-cache-dir \
    benchmarl \
    tensordict \
    torchrl \
    pyyaml \
    pandas \
    scipy \
    && pip cache purge

# Default: run training script with config path passed as CMD arg
ENV RESULTS_DIR=/workspace/results
ENTRYPOINT ["python", "rendezvous_comm/train.py"]
