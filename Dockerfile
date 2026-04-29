# Sử dụng bản Runtime thay vì Devel để giảm dung lượng (nếu không cần compile lại kernel)
# Nhưng với Unsloth, dùng bản devel vẫn an toàn nhất để khớp với các custom kernels.
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # Unsloth inference cực nhanh nhờ các kernel này
    UNSLOTH_DISABLE_FAST_KERNELS=0 

# Gộp các lệnh RUN để giảm số lượng layer, xóa cache ngay sau khi cài
RUN apt-get update && apt-get install -y \
    python3.10 python3-pip git curl \
    && rm -rf /var/lib/apt/lists/*

# Cài đặt uv - "vua tốc độ" hiện nay để thay thế pip
RUN pip install --no-cache-dir uv

WORKDIR /app

# ── Cài đặt Core Stack (Gộp lại để tối ưu) ──
# Cài bản Unsloth từ PyPI để tránh lỗi Git RPC như bạn vừa gặp
RUN uv pip install --system --no-cache \
    "torch==2.4.0" \
    "triton==3.0.0" \
    "xformers==0.0.27.post2" \
    --index-url https://download.pytorch.org/whl/cu121 && \
    uv pip install --system --no-cache \
    "transformers==4.56.2" \
    "bitsandbytes" \
    "sentencepiece" \
    "protobuf" \
    "accelerate" \
    "peft" \
    "trl==0.22.2" \
    "unsloth" 

# ── API Service ──
RUN uv pip install --system --no-cache \
    fastapi uvicorn pydantic python-multipart httpx

COPY . .

# # ── Health-check ─────────────────────────────────────────────
# HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
#     CMD curl -f http://localhost:8000/health || exit 1
 
EXPOSE 8000
 

# Chỉ 1 worker để tránh OOM vì mô hình LLM chiếm dụng VRAM cố định
CMD ["uvicorn", "app.inference_service.api_service:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]