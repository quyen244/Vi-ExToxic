import asyncio
import json
import logging
import os
import argparse
from typing import List, Dict, Any
import pandas as pd
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm
from config import Config


config = Config()

# logging information 
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("annotation.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)


class TeacherAnnotator:
    def __init__(self, prompt_path: str, model: str = "gpt-4o"):
        self.client = AsyncOpenAI(api_key=config.OPENAI_API_KEY)
        self.model = model
        self.prompt_path = prompt_path
        self.system_prompt = self._load_system_prompt()

    def _load_system_prompt(self) -> str:
        """Đọc prompt từ file .txt"""
        if not os.path.exists(self.prompt_path):
            raise FileNotFoundError(f"Không tìm thấy file prompt tại: {self.prompt_path}")
        with open(self.prompt_path, 'r', encoding='utf-8') as f:
            return f.read().strip()

    async def annotate_single(self, text: str, emotion: str, semaphore: asyncio.Semaphore) -> Dict:
        """Xử lý một dòng dữ liệu đơn lẻ"""
        async with semaphore:
            try:
                user_content = f"Input Text: '{text}'\nInput Emotion: '{emotion}'"
                
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": user_content}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.2
                )
                
                raw_res = response.choices[0].message.content
                result = json.loads(raw_res)
                
                # Làm phẳng cấu trúc JSON để lưu CSV dễ dàng hơn
                flat_res = {
                    "input_text": text,
                    "input_emotion": emotion,
                    "semantic_decoding": result.get("reasoning_scaffolding", {}).get("semantic_decoding"),
                    "slang_interpretation": result.get("reasoning_scaffolding", {}).get("slang_interpretation"),
                    "contextual_conflict": result.get("reasoning_scaffolding", {}).get("contextual_conflict"),
                    "target": result.get("reasoning_scaffolding", {}).get("target"),
                    "thought_trace": result.get("thought_trace"),
                    "final_label": result.get("final_label"),
                    "confidence_score": result.get("confidence_score"),
                    "suggested_action": result.get("suggested_action")
                }
                return flat_res

            except Exception as e:
                logging.error(f"Lỗi khi xử lý: {text[:30]}... | Lỗi: {e}")
                return {"input_text": text, "input_emotion": emotion, "error": str(e)}

    async def annotate_batch(self, data: List[Dict[str, str]], max_concurrent: int):
        """Xử lý một danh sách dữ liệu với semaphore giới hạn số request song song"""
        semaphore = asyncio.Semaphore(max_concurrent)
        tasks = [self.annotate_single(item['text'], item['emotion'], semaphore) for item in data]
        return await tqdm.gather(*tasks, desc="Đang xử lý batch", leave=False)

def parse_args():
    parser = argparse.ArgumentParser(description="Teacher Knowledge Generation Pipeline")
    parser.add_argument('--input', type=str, required=True, help="Path tới file CSV đầu vào")
    parser.add_argument('--output', type=str, required=True, help="Path lưu file kết quả")
    parser.add_argument('--prompt', type=str, default='system_prompt.txt', help="Path tới file prompt hệ thống")
    parser.add_argument('--model', type=str, default='gpt-4o', help="Model OpenAI (gpt-4o, gpt-4-turbo)")
    parser.add_argument('--batch_size', type=int, default=10, help="Số lượng request gửi song song (Max concurrent)")
    parser.add_argument('--text_col', type=str, default='text', help="Tên cột chứa nội dung text")
    parser.add_argument('--emotion_col', type=str, default='emotion', help="Tên cột chứa cảm xúc")
    parser.add_argument('--test', action='store_true', help="Chạy chế độ test với 5 dòng đầu tiên")
    return parser.parse_args()

async def test_logic_phase_1(args):
    """Hàm chạy test nhanh với các mẫu dữ liệu thực tế mạng xã hội để kiểm tra LLM Reasoning"""
    logging.info("--- ĐANG CHẠY TEST LOGIC NỘI BỘ (LLM REASONING) ---")
    
    # Khởi tạo annotator dành riêng cho test
    annotator = TeacherAnnotator(
        prompt_path=args.prompt, 
        model=args.model
    )

    # Dữ liệu test bao gồm text và emotion giả định
    test_samples = [
        {"text": "Hôm nay t đi học trễ vcl 😂😂😂, đm thầy giáo gắt quá !!!", "emotion": "angry"},
        {"text": "Khum bít bao giờ mới đc đi chơi vs ny nhỉ ❤️✨", "emotion": "sad"},
        {"text": "mng ơi mik mới mua cái đt mới xịn xò lắm lun 📱📱📱", "emotion": "happy"},
        {"text": "clgt sao m lại làm thế vs t 😡", "emotion": "angry"},
        {"text": "chằm zn lun á mng ơi ét ô ét 🆘", "emotion": "fear"}
    ]

    print("\n" + "="*30 + " TEST LLM REASONING RESULTS " + "="*30)
    
    # Chạy xử lý thông qua batch (với max_concurrent nhỏ cho test)
    results = await annotator.annotate_batch(test_samples, max_concurrent=2)

    for res in results:
        if "error" in res:
            print(f"[!] Lỗi: {res['error']}")
            continue
            
        print(f"\n[-] Input:  {res['input_text']}")
        print(f"[-] Emotion: {res['input_emotion']}")
        print(f"[>] Thought Trace: {res['thought_trace']}")
        print(f"[+] Final Label: {res['final_label']} (Score: {res['confidence_score']})")
        print("-" * 80)
        
    result_df = pd.DataFrame(results)
    result_df.to_csv(args.output, index=False, encoding='utf-8-sig')
    logging.info(f"Hoàn tất! Kết quả đã được lưu tại: {args.output}")

async def run_ppipeline_phase_1(args):
    # Khởi tạo Annotator
    annotator = TeacherAnnotator(
        prompt_path=args.prompt, 
        model=args.model
    )

    logging.info(f"Đang đọc dữ liệu từ {args.input}...")
    try:
        df = pd.read_csv(args.input)
    except Exception as e:
        logging.error(f"Không thể đọc file input: {e}")
        return
    
    # Chế độ lấy mẫu nếu cần (không ghi đè head(5) của test_logic)
    process_df = df.head(10) if args.test else df

    records = process_df.rename(
        columns={args.text_col: 'text', args.emotion_col: 'emotion'}
    ).to_dict('records')
    
    all_results = await annotator.annotate_batch(records, max_concurrent=args.batch_size)

    result_df = pd.DataFrame(all_results)
    result_df.to_csv(args.output, index=False, encoding='utf-8-sig')
    logging.info(f"Hoàn tất! Kết quả đã được lưu tại: {args.output}")

if __name__ == "__main__":
    args = parse_args()
    
    # Nếu có flag --test, chỉ chạy test_logic rồi thoát
    if args.test:
        asyncio.run(test_logic_phase_1(args))
    else:
        # Nếu không, chạy pipeline bình thường
        asyncio.run(run_ppipeline_phase_1(args))