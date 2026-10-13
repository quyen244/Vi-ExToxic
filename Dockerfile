# Uses Unsloth's CUDA-ready image to keep Torch/CUDA/Unsloth versions aligned.
# Build with: docker build -t vi-extoxic-sft .
FROM unsloth/unsloth:latest

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /workspace/vi-extoxic
COPY training/requirements.txt /tmp/training-requirements.txt
RUN python -m pip install --no-cache-dir -r /tmp/training-requirements.txt

COPY training /workspace/vi-extoxic/training

ENTRYPOINT ["python", "-m", "training.train_sft"]
CMD ["--help"]
