
<p align="center">
  <a href="https://www.uit.edu.vn/" title="University of Information Technology" style="border: none;">
    <img src="https://i.imgur.com/WmMnSRt.png" alt="University of Information Technology (UIT)">
  </a>
</p>

<h1 align="center"><b>SE363 - Vi-ExToxic: Explainable Hate Speech Detection</b></h1>

# **SE363 Personal Project: Vi-ExToxic (Explainable Toxic Detection for VN Social Media)**

> This project focuses on building an **Explainable Hate Speech Detection System** tailored for Vietnamese social media. By leveraging Small Language Models (SLMs) like Qwen2.5-3B / Llama-3.2-3B and combining textual data with **Emotion labels**, the system not only classifies toxic content but also generates a transparent "Thought Process" (Reasoning Path) explaining *why* a specific sentence is toxic, effectively decoding sarcasm and implicit toxicity.
> 
> **Technical Highlights:** Implementation of **Reasoning Scaffolding** and **Knowledge Distillation** to transfer reasoning capabilities from Large LLMs (Teacher) to SLMs (Student). Optimized for local execution using **Unsloth** (QLoRA Fine-tuning), **HuggingFace**, and deployed with a **Streamlit** interactive UI. 

<p align="center">
  <img src="thumbnail.png" width="800" alt="Vi-ExToxic Project Thumbnail">
  <br>
  <i>(Please replace thumbnail.png with your actual project screenshot)</i>
</p>

---

## **Team Information**

| No. | Student ID | Full Name | Role | Github | Email |
| --- | --- | --- | --- | --- | --- |
| 1 | 23521329 | Nguyen Van Quyen | Developer | [quyen244](https://github.com/quyen244) | 23521329@gm.uit.edu.vn |

*(Note: Update your team information if needed)*

---

## **Table of Contents**

* [Overview](#overview)
* [System Architecture](#system-architecture)
* [Tech Stack](#tech-stack)
* [Key Features](#key-features) 
* [Repository Structure](#repository-structure)
* [Installation & Usage](#installation--usage)
* [Example Queries](#example-queries)
* [DEMO](#demo)
* [Contributing](#contributing)
* [License](#license)

---

## **Overview**

**Vi-ExToxic** is an advanced content moderation system designed to bridge the gap between binary classification and human-like reasoning. Users (or systems) input social media comments along with their contextual emotions, and the AI automatically:

1. **Analyzes** the semantics and emotional synergy using a "step-by-step" reasoning approach.
2. **Identifies** hidden nuances such as sarcasm, passive-aggressiveness, and contextual insults.
3. **Categorizes** the text into one of 5 deep-context labels (Constructive/Clean, Implicit Toxicity, Explicit Hostility, Identity-Based Hate, Ambiguous/Noise).
4. **Explains** the final decision through a transparent logic trace.

### **Problem Statement**
Traditional Hate Speech models (like BERT-based classifiers) act as black boxes, outputting mere binary labels (0 or 1). They often struggle with the complex, context-heavy nature of Vietnamese slang, teencode, and emotional sarcasm. Moderators cannot trust an AI that bans users without providing a valid, culturally aware explanation.

### **Solution**
By utilizing **Knowledge Distillation**, we generate high-quality Chain-of-Thought (CoT) datasets from powerful LLMs (Gemini/GPT-4o). We then enforce a **Reasoning Scaffolding** technique (`### Phân tích` -> `### Tuy nhiên` -> `### Do đó` -> `### Kết luận`) to fine-tune a 3-Billion parameter model (Qwen2.5-3B). This results in a lightweight, explainable, and highly accurate model that can run locally on consumer-grade hardware.

---

## **System Architecture**

The project architecture is divided into two main pipelines:

### 1. Data Engineering & Distillation Pipeline
* **Stage 1 (Synthesis):** Gemini/GPT-4o acts as the Teacher, generating reasoning traces from Raw Text + Emotion Labels.
* **Stage 2 (Cross-Verification):** Secondary LLMs evaluate the logic to prevent hallucinations.
* **Stage 3 (Cultural Check):** Rule-based filtering for the latest Vietnamese Gen-Z slangs.
* **Stage 4 (Distillation):** Fine-tuning the SLM (Qwen-3B) using `Unsloth` on the curated Gold Dataset.

### 2. Inference & UI Pipeline
* **Input Layer:** Accepts text and emotion via a Streamlit interface.
* **Local Inference:** Runs the quantized 4-bit model locally to generate the Reasoning Trace.
* **Output:** Displays the Thought Process and the Final Label.

---

## **Tech Stack**

* **Core Frameworks:** PyTorch, HuggingFace Transformers
* **Fine-Tuning:** Unsloth (LoRA/QLoRA for fast & memory-efficient training)
* **Teacher LLMs (Data Generation):** OpenAI API, Google Gemini API
* **Student Model:** Qwen2.5-3B-Instruct / Llama-3.2-3B
* **Frontend UI:** Streamlit
* **Data Processing:** Pandas, Regex for text normalization

---

## **Key Features**

* 🧠 **Reasoning Scaffolding:** Forces the AI to strictly follow a logical path before jumping to conclusions, reducing bias and errors.
* 🎭 **Emotion-Aware Analysis:** Uses underlying emotions (e.g., Sadness, Anger, Surprise) as critical context to detect Sarcasm.
* 📊 **5-Tier Classification:** Goes beyond "Toxic/Non-Toxic" to classify severity: Clean, Implicit Toxicity, Explicit Hostility, Identity-Based Hate, and Ambiguous.
* ⚡ **Local & Lightweight:** Runs completely offline on standard GPUs (RTX 3060/4060) using 4-bit Quantization without sacrificing accuracy.
* 🔍 **X-AI UI (Explainable AI):** A user-friendly dashboard showing the real-time "Thought Path" of the AI.

---

## **Repository Structure**

```text
hate_speech_slm_project/
├── data/
│   ├── raw/                # Original Excel/CSV datasets with Emotion labels
│   ├── processed/          # Cleaned text (normalized teencode, slang)
│   └── gold_reasoning/     # Distilled dataset with Reasoning Scaffolding (.jsonl)
├── notebooks/              
│   ├── 01_data_exploration.ipynb     # EDA and Emotion distribution
│   └── 02_teacher_distillation.ipynb # API scripts for Teacher model
├── src/                    
│   ├── data_engine.py      # Automated reasoning generation scripts
│   ├── trainer.py          # Unsloth fine-tuning configuration
│   └── utils.py            # Vietnamese text processing utilities
├── models/                 # Saved LoRA adapters and quantized GGUF weights
├── app/                    
│   └── main.py             # Streamlit web application
├── requirements.txt        
└── README.md               
````

-----

## **Installation & Usage**

### 1\. Prerequisites

  * Python 3.10+
  * NVIDIA GPU with CUDA support (Minimum 8GB VRAM for training, 4GB for Inference).

### 2\. Setup Environment

```bash
# Clone the repository
git clone [https://github.com/quyen244/hate_speech_slm_project.git](https://github.com/quyen244/hate_speech_slm_project.git)
cd hate_speech_slm_project

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

### 3\. Running the Streamlit App

```bash
streamlit run app/main.py
```

-----

## **Example Queries**

**Input:**

  * **Text:** "Đúng là thiên tài, làm 10 bài sai cả 11." (A true genius, did 10 exercises and got 11 wrong.)
  * **Emotion:** Surprise

**AI Output (Reasoning Scaffolding):**

> **\#\#\# Phân tích:** Câu này sử dụng từ ngữ mang nghĩa tích cực là 'thiên tài'. <br>
> **\#\#\# Tuy nhiên:** Vế sau 'làm 10 bài sai cả 11' tạo ra sự mâu thuẫn trực tiếp và vô lý. Nhãn cảm xúc 'Surprise' ở đây nhấn mạnh sự châm biếm về năng lực yếu kém. <br>
> **\#\#\# Do đó:** Người nói đang sử dụng hình thức khen ngợi giả tạo để mỉa mai và hạ nhục đối tượng. <br>
> **\#\#\# Kết luận:** Implicit Toxicity (Độc hại ẩn ý - Mỉa mai).

-----

## **DEMO**

*(Include screenshots or a GIF recording of your Streamlit application here showing the reasoning process being generated step-by-step on the screen).*

\<p align="center"\>
\<img src="demo\_screenshot.png" width="700" alt="UI Demo"\>
\</p\>

-----

## **Contributing**

Contributions, issues, and feature requests are welcome\!
Feel free to check the [issues page](https://www.google.com/search?q=https://github.com/quyen244/hate_speech_slm_project/issues) if you want to contribute.

1.  Fork the Project
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the Branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request

-----

## **License**

Distributed under the MIT License. See `LICENSE` for more information.

