import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Class quản lý toàn bộ cấu hình hệ thống"""
    
    # API Keys
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    
    # Model Settings
    DEFAULT_MODEL = "gpt-4o"
    TEMPERATURE = 0.2
    MAX_CONCURRENT_REQUESTS = 10
    
    # Hệ thống Nhãn (Final Labels) nâng cao
    LABELS = [
        "Constructive/Clean",
        "Implicit Toxicity",
        "Explicit Hostility",
        "Identity-Based Hate",
        "Ambiguous/Noise"
    ]
    
    # Cấu hình đường dẫn mặc định
    DEFAULT_PROMPT_PATH = "system_prompt.txt"
    DEFAULT_INPUT_PATH = "input_data.csv"
    DEFAULT_OUTPUT_PATH = "teacher_dataset.csv"

    @classmethod
    def validate_config(cls):
        """Kiểm tra xem các biến môi trường quan trọng đã có chưa"""
        missing = []
        if not cls.OPENAI_API_KEY:
            missing.append("OPENAI_API_KEY")
        
        if missing:
            raise ValueError(f"Thiếu cấu hình biến môi trường: {', '.join(missing)}")
        return True
