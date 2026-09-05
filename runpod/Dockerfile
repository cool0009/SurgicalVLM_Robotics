# RunPod training image for SurgicalVLM_Robotics (COMP702)
# The venv is baked directly into the container (no host venv setup needed).
# Stack: CUDA 12.4-targeted, Python 3.11, torch >= 2.x (from requirements.txt).
#
# Build (then push to a registry so RunPod can pull it):
#   docker build -f runpod/Dockerfile -t <registry>/surgicalvlm:7b .
#
# Pod conventions:
#   - The repo is baked at /workspace (WORKDIR).
#   - A Network Volume is mounted at $RUNPOD_VOLUME (default /runpod-volume).
#   - All heavy state (data, HF cache, checkpoints, logs) lives on the volume
#     so a pod can be terminated and restarted without losing work.
#
# The requirements bump torch/transformers to the pinned ">=" versions (CUDA
# wheels from PyPI) over the torch preinstalled in the base template.

FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TOKENIZERS_PARALLELISM=false
ENV RUNPOD_VOLUME=/runpod-volume

WORKDIR /workspace

# Only the repo is copied; data/, checkpoints/, logs/, .git etc. are excluded
# via .dockerignore (they live on the volume or come from data.tar).
COPY . /workspace

# Install runtime deps. torch/transformers wheels satisfy the ">=" pins in
# requirements.txt, replacing the base image's baked-in versions.
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Import smoke-check: verify the key stacks import in the built image.
RUN python -c "import torch; print('torch', torch.__version__, 'cuda_avail=', torch.cuda.is_available())" \
    && python -c "import transformers; print('transformers', transformers.__version__)" \
    && python -c "import peft; print('peft', peft.__version__)" \
    && python -c "import accelerate; print('accelerate', accelerate.__version__)"

# Keep-alive: run the base image's /start.sh (starts sshd from PUBLIC_KEY and
# keeps the container running). Training is launched via runpod/train.sh.
CMD ["/start.sh"]