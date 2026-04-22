# Sử dụng NVIDIA CUDA image làm nền tảng
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

# Thiết lập biến môi trường
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Cài đặt Python và các công cụ hệ thống
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Cài đặt uv để quản lý package nhanh
RUN pip install uv

# Thiết lập thư mục làm việc
WORKDIR /app

# Cài đặt Torch và Unsloth (Khớp phiên bản CUDA 12.1)
# Lưu ý: Cài đặt xformers và triton tương thích
RUN uv pip install --system \
    "torch==2.4.0" \
    "triton==3.0.0" \
    "xformers<0.0.28" \
    --index-url https://download.pytorch.org/whl/cu121

RUN uv pip install --system \
    "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" \
    "trl<0.10.0" peft accelerate bitsandbytes fastapi uvicorn pydantic

# Cài đặt các thư viện bổ trợ cho Qwen (optional nhưng khuyên dùng)
RUN uv pip install --system --no-build-isolation causal_conv1d==1.4.0

# Copy mã nguồn vào container
COPY . .

# Mở cổng cho FastAPI
EXPOSE 8000

# Lệnh chạy service
CMD ["python3", "api_service.py"]