import time
import random
import torch
import json
import re
from unsloth import FastLanguageModel

class ViExToxicModel:
    """Giả lập Model Qwen 2.5-3B với khả năng Reasoning"""
    def __init__(self, model_name="Qwen 2.5-3B-Instruct"):
        self.model_name = model_name

    def predict(self, text):
        # Giả lập thời gian suy nghĩ của AI
        time.sleep(1.5) 
        
        # Mock dữ liệu trả về theo đúng format JSON yêu cầu
        is_toxic = random.choice([True, False])
        
        if is_toxic:
            return {
                "reasoning_scaffolding": {
                    "semantic_decoding": "Câu chứa từ ngữ thô tục và nhắm trực tiếp vào đối phương.",
                    "slang_interpretation": "Sử dụng từ lóng thô tục để nhấn mạnh sự tức giận.",
                    "contextual_conflict": "Không có mâu thuẫn, thái độ tiêu cực nhất quán.",
                    "target": "Người đối diện (Cá nhân)."
                },
                "thought_trace": "AI nhận thấy sự thù địch rõ ràng qua cách xưng hô và từ cảm thán. Không có dấu hiệu mỉa mai, đây là tấn công trực diện.",
                "final_label": "Explicit Hostility",
                "confidence_score": round(random.uniform(0.85, 0.99), 2)
            }
        else:
            return {
                "reasoning_scaffolding": {
                    "semantic_decoding": "Câu chia sẻ thông tin hoặc cảm xúc cá nhân tích cực.",
                    "slang_interpretation": "Sử dụng teencode Gen Z thân thiện.",
                    "contextual_conflict": "Cảm xúc và nội dung văn bản đồng nhất.",
                    "target": "Không có (Chia sẻ chung)."
                },
                "thought_trace": "Văn bản mang tính chất xây dựng, không chứa từ ngữ độc hại hay tấn công cá nhân.",
                "final_label": "Constructive/Clean",
                "confidence_score": round(random.uniform(0.9, 0.98), 2)
            }

class QwenInferenceService:
    def __init__(self, model_path="unsloth/Qwen2.5-3B-Instruct-bnb-4bit", max_seq_length=2048):
        print(f"🚀 Khởi tạo Service với Model: {model_path}...")
        self.max_seq_length = max_seq_length
        
        # 1. Load Model & Tokenizer
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name = model_path,
            max_seq_length = max_seq_length,
            load_in_4bit = True,
            cache_dir="/app/models"
        )
        
        # 2. Bật chế độ suy luận nhanh
        FastLanguageModel.for_inference(self.model)
        print("✅ Model đã sẵn sàng suy luận.")

    def _extract_json(self, text: str) -> dict:
        """Bóc tách JSON từ chuỗi kết quả (xử lý cả markdown code blocks)"""
        try:
            # Tìm cặp ngoặc nhọn {} hoặc ngoặc vuông []
            match = re.search(r'(\{.*\}|\[.*\])', text, re.DOTALL)
            if match:
                return json.loads(match.group(1))
            return {"error": "No JSON found", "raw": text}
        except Exception as e:
            return {"error": str(e), "raw": text}

    def predict_toxic(self, text: str, emotion: str) -> dict:
        """Hàm chuyên dụng cho bài toán Vi-ExToxic"""
        
        # 1. Chuẩn bị System Prompt (giống hệt lúc Train)
        system_prompt = (
            """
                You are a Vietnamese social media expert. Follow these steps to analyze the user's input:
                Step 1: Decode semantics, teencode, and slang.
                Step 2: Compare Text with Emotion to detect sarcasm or hidden intent.
                Step 3: Identify the target of the message.
                Step 4: Synthesize logic and assign one of these labels: 
                - Constructive/Clean
                - Implicit Toxicity
                - Explicit Hostility
                - Identity-Based Hate
                - Ambiguous/Noise
                
                Return ONLY a JSON object:
                {
                "reasoning_scaffolding": {
                    "semantic_decoding": "...",
                    "slang_interpretation": "...",
                    "contextual_conflict": "...",
                    "target": "..."
                },
                "thought_trace": "Brief step-by-step logic summary in Vietnamese",
                "final_label": "Label name"
                }
            """
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Input Text: '{text}'\nInput Emotion: '{emotion}'"}
        ]

        # 2. Tokenize theo template của Qwen
        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize = True,
            add_generation_prompt = True,
            return_tensors = "pt",
        ).to("cuda")

        # 3. Generate
        outputs = self.model.generate(
            input_ids = inputs,
            max_new_tokens = 512,
            temperature = 0.1,    
            top_p = 0.9,
            repetition_penalty = 1.1,
            use_cache = True
        )

        # 4. Decode và lấy phần trả lời của Assistant
        full_response = self.tokenizer.batch_decode(outputs)[0]
        assistant_part = full_response.split("<|im_start|>assistant\n")[-1].replace("<|im_end|>", "").strip()
        
        return self._extract_json(assistant_part)

if __name__ == '__main__':
    qwen_service = QwenInferenceService()

    test_cases = [
    {"text": "mng ơi mik mới mua cái đt mới xịn xò lắm lun 📱", "emotion": "happy"},
    {"text": "clgt sao m lại làm thế vs t 😡", "emotion": "angry"},
    {"text": "Giỏi quá vcl cả họ tự hào luôn nhé", "emotion": "smirk"}
    ]

    print("\n" + "="*50)
    print("🔍 TEST INFERENCE KẾT QUẢ")
    print("="*50)

    for i, case in enumerate(test_cases):
        print(f"\n[Case {i+1}] Input: {case['text']}")
        result = qwen_service.predict_toxic(case['text'], case['emotion'])
        
        # In ra kết quả JSON đẹp
        print(json.dumps(result, indent=2, ensure_ascii=False))
        print("-" * 50)

# py -m app.inference_service.model_engine