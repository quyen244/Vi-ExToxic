import pandas as pd
import re
import argparse
import logging
from abc import ABC, abstractmethod
import emoji
import string

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ==========================================
# 1. BASE CLASS
# ==========================================
class BaseProcessor(ABC):
    @abstractmethod
    def process(self, text: str) -> str:
        pass

# ==========================================
# 2. CÁC CẤP ĐỘ XỬ LÝ
# ==========================================

class Level1_BasicCleaner(BaseProcessor):
    """Cấp 1: Xóa các thành phần rác cơ bản (URL, HTML, Mentions, Hashtags)"""
    def process(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = re.sub(r'http\S+|www\.\S+', '', text)
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#\w+', '', text)
        return text

class Level2_SocialMediaCleaner(BaseProcessor):
    """Cấp 2: Xử lý Teencode, Ký tự kéo dài, Giữ nguyên Emoji"""
    def __init__(self, teencode_path: str = None):
        self.teencode_dict = {}
        if teencode_path:
            try:
                with open(teencode_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line and ':' in line:
                            key, value = line.split(':', 1)
                            self.teencode_dict[key.strip().lower()] = value.strip()
                logging.info(f"Đã tải {len(self.teencode_dict)} từ slang từ {teencode_path}")
            except Exception as e:
                logging.error(f"Lỗi khi đọc file slang: {e}")

    def process(self, text: str) -> str:
        # Chuẩn hóa ký tự kéo dài (vuiiiii -> vui, 😂😂😂 -> 😂)
        text = re.sub(r'(.)\1{2,}', r'\1', text)
        
        # Chuyển đổi Teencode
        words = text.split()
        normalized_words = [self.teencode_dict.get(word.lower(), word) for word in words]
        return ' '.join(normalized_words)

class Level3_VietnameseNLPCleaner(BaseProcessor):
    """Cấp 3: Chuẩn hóa NLP, Xóa dấu câu nhưng GIỮ EMOJI"""
    def process(self, text: str) -> str:
        # Chuyển về chữ thường (emoji không bị ảnh hưởng)
        text = text.lower()
        
        # Tạo pattern các dấu câu cần xóa (loại trừ emoji)
        # string.punctuation chứa các ký tự: !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
        punctuation_to_remove = string.punctuation
        pattern = re.compile(f"[{re.escape(punctuation_to_remove)}]")
        text = pattern.sub(' ', text)
        
        # Xóa khoảng trắng thừa
        text = re.sub(r'\s+', ' ', text).strip()
        return text

# ==========================================
# 3. PIPELINE ORCHESTRATOR
# ==========================================
class TextProcessingPipeline:
    def __init__(self, processors: list):
        self.processors = processors

    def run(self, text: str) -> str:
        for processor in self.processors:
            text = processor.process(text)
        return text

# ==========================================
# 4. HÀM CHÍNH & CLI
# ==========================================
def args_parser():
    parser = argparse.ArgumentParser(description="Vietnamese Social Media Data Processor Pipeline")
    parser.add_argument('--input', type=str, required=True, help="Đường dẫn tới file input (CSV hoặc Excel)")
    parser.add_argument('--output', type=str, required=True, help="Đường dẫn lưu file output (CSV)")
    parser.add_argument('--slang_file', type=str, default='slang.txt', help="Đường dẫn file chứa slang words")
    parser.add_argument('--text_col', type=str, default='Sentence', help="Tên cột chứa text cần xử lý")
    return parser.parse_args()
    
def main():
    args = args_parser()

    pipeline = TextProcessingPipeline([
        Level1_BasicCleaner(),
        Level2_SocialMediaCleaner(teencode_path=args.slang_file),
        Level3_VietnameseNLPCleaner()
    ])

    logging.info(f"Đang đọc dữ liệu từ: {args.input}")
    try:
        if args.input.endswith('.xlsx') or args.input.endswith('.xls'):
            df = pd.read_excel(args.input)
        else:
            df = pd.read_csv(args.input)
    except Exception as e:
        logging.error(f"Lỗi khi đọc file: {e}")
        return

    if args.text_col not in df.columns:
        logging.error(f"Không tìm thấy cột '{args.text_col}'")
        return

    df[f'{args.text_col}_cleaned'] = df[args.text_col].astype(str).apply(pipeline.run)

    try:
        if 'Unnamed: 0' in df.columns:
            df.drop(columns=['Unnamed: 0'], inplace=True)
        df.to_csv(args.output, index=False, encoding='utf-8-sig')
        logging.info(f"Xử lý xong! Kết quả lưu tại: {args.output}")
    except Exception as e:
        logging.error(f"Lỗi khi lưu file: {e}")

def test():
    # Giả định file slang nằm cùng thư mục hoặc đường dẫn cụ thể
    pipeline = TextProcessingPipeline([
        Level1_BasicCleaner(),
        Level2_SocialMediaCleaner('slang.txt'),
        Level3_VietnameseNLPCleaner()
    ])

    test_samples = [
        "Hôm nay t đi học trễ vcl 😂😂😂, đm thầy giáo gắt quá !!!",
        "Khum bít bao giờ mới đc đi chơi vs ny nhỉ ❤️✨",
        "mng ơi mik mới mua cái đt mới xịn xò lắm lun 📱📱📱",
        "clgt sao m lại làm thế vs t 😡",
        "chằm zn lun á mng ơi ét ô ét 🆘"
    ]

    print("\n" + "="*20 + " TEST PIPELINE RESULTS (EMOJI PRESERVED) " + "="*20)
    for original in test_samples:
        cleaned = pipeline.run(original)
        print(f"[-] Input:  {original}")
        print(f"[+] Output: {cleaned}")
        print("-" * 63)

if __name__ == "__main__":
    main()