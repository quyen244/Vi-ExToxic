### 1. Set up môi trường trên Kaggle

Kaggle cung cấp 2 GPU T4 (tổng 30GB VRAM), khá ổn để train mô hình 3B bằng kỹ thuật PEFT (LoRA).

**Các bước chuẩn bị:**
1.  **Bật Internet:** Trong menu bên phải (Settings), chọn "Internet on".
2.  **Chọn Accelerator:** Chọn "GPU T4 x2".
3.  **Cài đặt thư viện:** Chạy lệnh sau trong cell đầu tiên:

```python
!pip install -U trl transformers accelerate peft bitsandbytes vllm
```

4.  **Hugging Face Login:** Để tải mô hình Qwen và lưu kết quả.
```python
from huggingface_hub import login
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
token = user_secrets.get_secret("hf_token") # Lưu token trong phần Add-ons -> Secrets
login(token=token)
```

---

### 2. Giới thiệu thư viện TRL (Transformer Reinforcement Learning)

**TRL** là một thư viện "ngách" nhưng cực kỳ mạnh mẽ của Hugging Face, chuyên dùng để tối ưu hóa các mô hình ngôn ngữ lớn (LLMs) thông qua các phương pháp học tăng cường (RL) và hậu đào tạo (post-training).

**Các Key Concepts:**
*   **Model Classes:** TRL hỗ trợ các mô hình có sẵn từ `transformers` và tích hợp tốt với `peft` (để dùng LoRA).
*   **Trainers (Trái tim của TRL):**
    *   `SFTTrainer`: Fine-tuning có giám sát thông thường.
    *   `RewardTrainer`: Huấn luyện mô hình chấm điểm (Reward Model).
    *   `PPOTrainer/DPOTrainer`: Tối ưu hóa dựa trên phản hồi của con người (RLHF).
    *   **`GKDTrainer`**: Đây là thành phần mới, chuyên dùng cho **Generalized Knowledge Distillation**.
*   **Data Collators:** Tự động xử lý padding và định dạng completion cho các bài toán sinh câu (generation).

---

### 3. Phương pháp On-policy Knowledge Distillation (GKD)

#### A. Cơ chế GKD (Generalized Knowledge Distillation)
Trong Distillation truyền thống (Off-policy), mô hình Học sinh học trên dữ liệu cố định có sẵn. Điều này dẫn đến **Exposure Bias**: Khi Học sinh tự tạo văn bản, nó mắc lỗi nhỏ, các lỗi này tích lũy dần khiến kết quả cuối cùng bị tệ đi.

**GKD giải quyết bằng cách:**
1.  Cho mô hình **Học sinh (Student)** tự sinh câu trả lời (On-policy).
2.  Mô hình **Giáo viên (Teacher)** sẽ soi vào câu trả lời đó và cung cấp xác suất (logits).
3.  Học sinh học cách khớp phân phối logits của mình với Giáo viên trên chính những gì nó vừa tạo ra.

#### B. Cách thực hiện cụ thể

Vì mô hình **Qwen 2.5-72B** quá lớn để chạy trên Kaggle, ta sẽ giả định bạn host nó ở một server riêng qua **vLLM** và gọi API (hoặc dùng phiên bản 7B làm Teacher để demo trên Kaggle).

##### Bước 1: Thiết lập Teacher API (vLLM)
Nếu bạn có server riêng chạy vLLM:
```bash
# Lệnh chạy trên server (vLLM)
python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-72B-Instruct --tensor-parallel-size 4
```
Trong TRL, `GKDTrainer` có thể nhận một `teacher_model` cục bộ hoặc một endpoint. Tuy nhiên, bản `trl` hiện tại ưu tiên `teacher_model` là một object model. Nếu dùng API, bạn cần một lớp Wrapper.

##### Bước 2: Triển khai GKD với TRL (Code mẫu)

Dưới đây là cách cấu hình `GKDTrainer` với Qwen 2.5-3B (Student) và sử dụng hàm mất mát Jensen-Shannon.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import GKDConfig, GKDTrainer, ModelConfig
from peft import LoraConfig

# 1. Load Tokenizer
model_id = "Qwen/Qwen2.5-3B-Instruct"
teacher_id = "Qwen/Qwen2.5-72B-Instruct" # Giả định dùng API hoặc model nhỏ hơn trên local

# 2. Cấu hình LoRA cho Student (Để chạy được trên Kaggle)
lora_config = LoraConfig(
    r=8,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    task_type="CAUSAL_LM",
)

# 3. Thiết lập GKD Config
gkd_config = GKDConfig(
    output_dir="./qwen-gkd-distilled",
    learning_rate=5e-5,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    max_steps=1000,
    # GKD Specific params
    lmbda=1.0,           # 1.0 = Hoàn toàn On-policy
    beta=0.1,            # Trọng số cho KD loss
    loss_type="jsd",     # Jensen-Shannon Divergence giúp ổn định gradient
    server_url="http://your-vllm-endpoint:8000/v1", # Nếu dùng vLLM API
    model_kwargs={"torch_dtype": torch.bfloat16},
)

# 4. Load Student Model
student_model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    device_map="auto", 
    torch_dtype=torch.bfloat16
)

# 5. Khởi tạo GKDTrainer
# Lưu ý: Nếu Teacher là API, TRL sẽ gửi yêu cầu sinh logits qua mạng
trainer = GKDTrainer(
    model=student_model,
    args=gkd_config,
    train_dataset=train_dataset, # Dataset chứa các câu prompt
    peft_config=lora_config,
    tokenizer=AutoTokenizer.from_pretrained(model_id)
)

# 6. Training
trainer.train()
```

#### C. Tại sao dùng Jensen-Shannon Divergence (JSD)?
Trong bài báo *Agarwal et al.*, JSD được khuyến khích hơn KL-Divergence thông thường vì:
*   **Tính đối xứng:** JSD không bị bùng nổ gradient khi xác suất của Student hoặc Teacher tiệm cận 0 ở một số token.
*   **Ổn định:** Nó giới hạn giá trị loss trong khoảng [0, 1], giúp quá trình hội tụ mượt mà hơn khi huấn luyện On-policy (vốn dĩ rất nhạy cảm với nhiễu).

### Lưu ý quan trọng cho Kaggle:
1.  **Memory:** Nếu 3B vẫn gây tràn bộ nhớ (OOM) khi chạy cùng lúc với các tiến trình khác, hãy dùng `load_in_4bit=True` trong `model_kwargs`.
2.  **Dataset:** Dữ liệu cho GKD chỉ cần cột `prompt` (hoặc `input`). Student sẽ tự tạo `output`, sau đó Teacher gán nhãn. Một dataset tốt cho việc này là `Open-Orca` hoặc `ShareGPT`.
3.  **Teacher Model:** Nếu không có API cho bản 72B, hãy thử nghiệm với Teacher là **Qwen2.5-7B-Instruct** (có thể vừa khít VRAM nếu dùng 4-bit quantization trên GPU thứ 2 của Kaggle).
