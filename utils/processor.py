import pandas as pd
import re
import argparse
import logging
from abc import ABC, abstractmethod
import emoji

# Cấu hình logging chuyên nghiệp
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ==========================================
# 1. BASE CLASS (Lớp cơ sở trừu tượng)
# ==========================================
class BaseProcessor(ABC):
    @abstractmethod
    def process(self, text: str) -> str:
        """Phương thức bắt buộc các class con phải implement"""
        pass

# ==========================================
# 2. CÁC CẤP ĐỘ XỬ LÝ (Processing Levels)
# ==========================================

class Level1_BasicCleaner(BaseProcessor):
    """Cấp 1: Xóa các thành phần rác cơ bản của văn bản web/social"""
    def process(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        # Xóa URL
        text = re.sub(r'http\S+|www\.\S+', '', text)
        # Xóa HTML tags
        text = re.sub(r'<.*?>', '', text)
        # Xóa Mentions (@username)
        text = re.sub(r'@\w+', '', text)
        # Xóa Hashtags (#hashtag) - có thể giữ lại chữ nếu cần, ở đây chọn xóa cả cụm
        text = re.sub(r'#\w+', '', text)
        return text

class Level2_SocialMediaCleaner(BaseProcessor):
    """Cấp 2: Xử lý đặc thù ngôn ngữ mạng xã hội (Teencode, Ký tự kéo dài, Emoji)"""
    def __init__(self, teencode_path: str = None):
        self.teencode_dict = {}
        if teencode_path:
            try:
                with open(teencode_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line and ':' in line:
                            # Chia theo dấu : đầu tiên tìm thấy
                            key, value = line.split(':', 1)
                            self.teencode_dict[key.strip().lower()] = value.strip()
                logging.info(f"Đã tải {len(self.teencode_dict)} từ slang từ {teencode_path}")
            except Exception as e:
                logging.error(f"Lỗi khi đọc file slang: {e}")

    def process(self, text: str) -> str:
        # Xóa emoji
        text = emoji.replace_emoji(text, replace='')
        
        # Chuẩn hóa ký tự kéo dài (VD: "vuiiiii" -> "vui")
        text = re.sub(r'(.)\1{2,}', r'\1', text)
        
        # Chuyển đổi Teencode
        words = text.split()
        normalized_words = [self.teencode_dict.get(word.lower(), word) for word in words]
        return ' '.join(normalized_words)

class Level3_VietnameseNLPCleaner(BaseProcessor):
    """Cấp 3: Chuẩn hóa NLP cho tiếng Việt"""
    def process(self, text: str) -> str:
        text = text.lower()
        # Xóa dấu câu và ký tự đặc biệt, giữ lại chữ cái, số
        text = re.sub(r'[^\w\s\d_]', ' ', text)
        # Xóa khoảng trắng thừa
        text = re.sub(r'\s+', ' ', text).strip()
        return text

# ==========================================
# 3. PIPELINE ORCHESTRATOR (Trình điều phối)
# ==========================================
class TextProcessingPipeline:
    def __init__(self, processors: list):
        self.processors = processors

    def run(self, text: str) -> str:
        for processor in self.processors:
            text = processor.process(text)
        return text

# ==========================================
# 4. HÀM CHÍNH & ARGPARSE (CLI)
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

    # Khởi tạo Pipeline và truyền đường dẫn file slang vào Level 2
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

    # Áp dụng xử lý
    df[f'{args.text_col}_cleaned'] = df[args.text_col].astype(str).apply(pipeline.run)

    try:
        df.to_csv(args.output, index=False, encoding='utf-8-sig')
        logging.info(f"Xử lý xong! Kết quả lưu tại: {args.output}")
    except Exception as e:
        logging.error(f"Lỗi khi lưu file: {e}")

def test():
    pipeline = TextProcessingPipeline([
        Level1_BasicCleaner(),
        Level2_SocialMediaCleaner('/kaggle/input/datasets/quyenuit24/slang2/slang.txt'),
        Level3_VietnameseNLPCleaner()
    ])

    test_samples = [
        "Hôm nay t đi học trễ vcl, đm thầy giáo gắt vcl",
        "Khum bít bao giờ mới đc đi chơi vs ny nhỉ",
        "mng ơi mik mới mua cái đt mới xịn xò lắm lun",
        "clgt sao m lại làm thế vs t",
        "đcm cuộc đời như cc z",
        "rùi xong lun, chx bít lm j hết tr",
        "thik ghê á, đr đó hihi",
        "any và eny đang đi ăn cơm tiệm",
        "chằm zn lun á mng ơi ét ô ét",
        "ib báo giá đi bồ ơi, rep muộn v",
        "vcc cái btap này khó quá z"
    ]

    print("\n" + "="*20 + " TEST PIPELINE RESULTS " + "="*20)
    for original in test_samples:
        cleaned = pipeline.run(original)
        print(f"[-] Input:  {original}")
        print(f"[+] Output: {cleaned}")
        print("-" * 63)

if __name__ == "__main__":
    main() 
    # test()   # Chạy test để kiểm tra kết quả pipeline