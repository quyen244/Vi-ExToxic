import time
import random

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
if __name__ == '__main__':
    print('hi')