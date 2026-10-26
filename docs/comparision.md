Để thực hiện các thí nghiệm này một cách chuẩn khoa học (Scientific Method), bạn cần tuân thủ quy trình: **Giả thuyết -> Thiết lập biến số -> Thực nghiệm -> Kiểm định thống kê**.

Dưới đây là 3 pipeline chi tiết:

---

### 1. Pipeline kiểm chứng Giả thuyết $H_0$: Input (Text + Emotion) > Text đơn thuần

Đây là dạng **Ablation Study** (Nghiên cứu bóc tách) để xem feature "Emotion" có đóng góp thực sự vào hiệu năng hay không.

*   **Dữ liệu (Data):** Dataset cần có 3 cột chính: `Text`, `Emotion_Label` (từ người gán hoặc tool), và `Target_Label` (nhãn chuẩn).
*   **Thiết lập mô hình:**
    *   **Nhóm Đối chứng (Control Group):** Model $M_{base}$ chỉ nhận input là `Text`.
    *   **Nhóm Thử nghiệm (Experimental Group):** Model $M_{emotion}$ nhận input là `Text` + `Emotion` (ví dụ: nối thêm string emotion vào text hoặc dùng Multi-modal fusion).
*   **Quy trình:**
    1.  **K-fold Cross-validation:** Chia dữ liệu thành 5 hoặc 10 phần để đảm bảo kết quả không do ngẫu nhiên.
    2.  **Training:** Huấn luyện cả 2 nhóm với cùng Hyperparameters (Learning rate, Batch size, Epochs).
    3.  **Metrics:** Đo Accuracy, F1-Score (Macro), và Precision/Recall.
    4.  **Kiểm định thống kê (Statistical Test):** Sử dụng **Paired T-test** hoặc **Wilcoxon Signed-Rank Test** trên kết quả của các fold. 
        *   Nếu $p-value < 0.05$, ta bác bỏ $H_0$ (tức là Emotion thực sự có giúp ích).

---

### 2. Pipeline kiểm chứng $H_0$: Knowledge Distillation (KD) tốt hơn Fine-tuning thông thường

Mục tiêu là chứng minh việc "học từ Model lớn" (Teacher) tốt hơn là "tự học từ nhãn" (Hard labels).

*   **Thiết lập mô hình:**
    *   **Teacher ($T$):** Một model lớn đã đạt accuracy cao (ví dụ: GPT-4 hoặc Qwen-72B).
    *   **Student ($S$):** Một model nhỏ (ví dụ: PhoBERT hoặc Qwen-1.5B).
*   **Quy trình:**
    1.  **Baseline ($S_{base}$):** Fine-tune Student trực tiếp trên tập dữ liệu bằng nhãn gốc (Cross-Entropy Loss).
    2.  **Distillation ($S_{distill}$):** 
        *   Cho Teacher chạy qua tập dữ liệu để lấy **Soft Labels** (xác suất đầu ra/Logits).
        *   Huấn luyện Student bằng hàm Loss kết hợp: $Loss = \alpha L_{KD}(Soft\_Labels) + (1-\alpha) L_{CE}(Hard\_Labels)$.
    3.  **So sánh:** Đánh giá $S_{base}$ và $S_{distill}$ trên cùng một tập Test ẩn.
    4.  **Phân tích đường cong học tập (Learning Curve):** Kiểm tra xem KD có giúp model nhỏ hội tụ nhanh hơn hoặc đạt "trần" hiệu năng cao hơn không.

---

### 3. Pipeline so sánh: Truyền thống vs. BERT Encoders vs. LLMs

Đây là bài toán **Benchmarking** để tìm ra sự đánh đổi giữa **Hiệu năng (Performance)** và **Chi phí (Resource)**.

#### Danh sách đối tượng so sánh:
1.  **Traditional:** TF-IDF + SVM hoặc Logistic Regression.
2.  **Encoders (SOTA cục bộ):** PhoBERT hoặc viSBERT (Fine-tuned).
3.  **LLMs (Few-shot/Zero-shot):** Qwen-3B-Instruct (chạy qua Ollama) với kỹ thuật Prompting.
4.  **LLMs (Fine-tuned):** Qwen-3B sử dụng LoRA/QLoRA.

#### Các bước thực hiện:
1.  **Chuẩn bị Dataset chuẩn:** Sử dụng một tập dữ liệu tiếng Việt (như UIT-VSFC hoặc dữ liệu của bạn) đã được làm sạch.
2.  **Thực thi:**
    *   **SVM:** Baseline nhanh nhất, không cần GPU.
    *   **PhoBERT:** Cần GPU, fine-tune toàn bộ trọng số.
    *   **Qwen-3B (Ollama):** Không training, chỉ dùng Prompt (ví dụ: "Phân tích văn bản sau theo định dạng JSON..."). 
    *   **Qwen-3B (SFT):** Fine-tune với tập dữ liệu nhỏ bằng kỹ thuật LoRA.
3.  **Tiêu chí đánh giá (Multi-metric):**
    *   **Chất lượng:** Macro F1-Score (quan trọng nhất cho tập dữ liệu lệch nhãn).
    *   **Tốc độ (Latency):** Thời gian xử lý 1 sample (ms).
    *   **Tài nguyên (Compute):** Lượng VRAM tiêu thụ.
    *   **Độ tin cậy:** Khả năng output đúng định dạng JSON yêu cầu (LLMs hay bị lỗi format).

#### Bảng so sánh kết quả (Mẫu):

| Model | F1-Score | Latency | VRAM | Khả năng giải thích (Reasoning) |
| :--- | :--- | :--- | :--- | :--- |
| **SVM** | 0.72 | 1ms | 0GB | Không |
| **PhoBERT** | 0.88 | 20ms | 2GB | Thấp |
| **Qwen-3B (Zero-shot)** | 0.81 | 200ms | 4GB | Rất cao |
| **Qwen-3B (LoRA)** | **0.91** | 200ms | 4GB | Rất cao |

### Tổng kết quy trình khoa học:
Để kết quả được chấp nhận trong các báo cáo khoa học, bạn cần:
1.  **Fix seed:** Đảm bảo tính lặp lại (Reproducibility).
2.  **Error Analysis:** Lấy ra 20-50 sample mà các model đoán sai để phân tích tại sao (do slang, do ngữ cảnh hay do thiếu kiến thức nền).
3.  **Ablation Study:** Chứng minh rõ mỗi thành phần thêm vào (như Emotion) đóng góp bao nhiêu % vào tổng điểm.

---

## Vi-ExToxic evaluation results

All runs use the same 320-row held-out dataset and deterministic decoding.
Few-shot examples come from the supplied example and the training split, never the holdout.

### No-shot results

| Run | Prompt language | Examples | Accuracy | Macro-F1 | Unparsed |
|---|---|---:|---:|---:|---:|
| No-CoT (Vietnamese prompt) | Vietnamese | 320 | 78.12% | 37.07% | 0 |
| CoT-distilled (Vietnamese prompt) | Vietnamese | 320 | 71.25% | 40.43% | 0 |
| CoT (Vietnamese prompt) | Vietnamese | 320 | 74.38% | 34.06% | 0 |
| CoT (English prompt) | English | 320 | 71.56% | 34.58% | 0 |

### CoT prompt-language comparison (no-shot)

English CoT Macro-F1 is 0.52 percentage points higher than Vietnamese CoT.

### Few-shot impact

| Run | No-shot Accuracy | Few-shot Accuracy | No-shot Macro-F1 | Few-shot Macro-F1 | Macro-F1 change |
|---|---:|---:|---:|---:|---:|
| No-CoT (Vietnamese prompt) | 78.12% | 62.81% | 37.07% | 32.51% | -4.56 pp |
| CoT (Vietnamese prompt) | 74.38% | 61.88% | 34.06% | 31.46% | -2.60 pp |
| CoT-distilled (Vietnamese prompt) | 71.25% | 67.50% | 40.43% | 34.44% | -5.99 pp |

### Per-class F1: no-shot

| Label | Support | No-CoT | CoT-distilled | CoT-VI | CoT-EN |
|---|---:|---:|---:|---:|---:|
| Constructive/Clean | 231 | 88.31% | 83.37% | 85.27% | 83.83% |
| Implicit Toxicity | 29 | 28.57% | 27.69% | 20.41% | 20.59% |
| Explicit Hostility | 53 | 68.47% | 62.50% | 64.62% | 68.47% |
| Identity-Based Hate | 6 | 0.00% | 28.57% | 0.00% | 0.00% |
| Ambiguous/Noise | 1 | 0.00% | 0.00% | 0.00% | 0.00% |

### Per-class F1: few-shot impact

| Label | Support | No-CoT (0 → 3) | CoT-VI (0 → 3) | CoT-distilled (0 → 3) |
|---|---:|---:|---:|---:|
| Constructive/Clean | 231 | 88.31% → 79.62% | 85.27% → 77.51% | 83.37% → 81.00% |
| Implicit Toxicity | 29 | 28.57% → 9.52% | 20.41% → 17.86% | 27.69% → 4.55% |
| Explicit Hostility | 53 | 68.47% → 62.00% | 64.62% → 55.05% | 62.50% → 66.67% |
| Identity-Based Hate | 6 | 0.00% → 11.43% | 0.00% → 6.90% | 28.57% → 20.00% |
| Ambiguous/Noise | 1 | 0.00% → 0.00% | 0.00% → 0.00% | 0.00% → 0.00% |

Total held-out support: 320 examples.
Classes with fewer than 10 examples: Identity-Based Hate (n=6), Ambiguous/Noise (n=1).
Interpret per-class scores with their support, especially for rare labels.
