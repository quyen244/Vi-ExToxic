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
from evaluator import EvaluatorData
from llm_providers import AnthropicProvider

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

# ======================= TeacherAnnotator =======================
# ============================================================

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

# ======================= DataVerifier =======================
# ============================================================
class DataVerifier:
    def __init__(self, anthropic_provider: Any, system_prompt_path: str):
        self.provider = anthropic_provider
        self.system_prompt = self._load_prompt(system_prompt_path)
        self.evaluator = EvaluatorData()
    
    def get_statistics(self, df: pd.DataFrame):
        return self.evaluator.get_statistics_metrics(df)

    def _load_prompt(self, path: str) -> str:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read().strip()

    def _format_batch_input(self, rows: List[Dict]) -> str:
        """Gom N mẫu vào 1 chuỗi theo format chuẩn định nghĩa trong Prompt"""
        count = len(rows)
        formatted_text = f"Hãy kiểm định chính xác {count} mẫu sau đây. Trả về một JSON ARRAY chứa đúng {count} đối tượng.\n\n"
        
        for i, row in enumerate(rows):
            # Gộp các bước phân tích nhỏ của Teacher thành Scaffolding
            scaffolding = (
                f"1. Semantic: {row.get('semantic_decoding', 'N/A')}; "
                f"2. Slang: {row.get('slang_interpretation', 'N/A')}; "
                f"3. Conflict: {row.get('contextual_conflict', 'N/A')}"
            )
            
            formatted_text += f"--- SAMPLE {i+1} ---\n"
            formatted_text += f"* **TEXT:** {row.get('input_text')}\n"
            formatted_text += f"* **EMOTION:** {row.get('input_emotion')}\n"
            formatted_text += f"* **TEACHER_reasoning_scaffolding:** {scaffolding}\n"
            formatted_text += f"* **TEACHER_thought_trace:** {row.get('thought_trace')}\n"
            formatted_text += f"* **TEACHER_LABEL:** {row.get('final_label')}\n\n"
            
        return formatted_text

    async def verify_chunk(self, chunk_rows: List[Dict]) -> List[Dict]:
        user_input = self._format_batch_input(chunk_rows)
       
        try:
            # Gọi Provider (Bây giờ đã dùng _extract_json mới)
            responses = await self.provider.generate_response(self.system_prompt, user_input)
            
            # Kiểm tra nếu responses là None hoặc không phải list
            if not isinstance(responses, list):
                responses = [responses] if responses else []

            verified_results = []
            for i, row in enumerate(chunk_rows):
                # Lấy kết quả từ AI, nếu AI trả thiếu thì gán giá trị mặc định
                v_res = responses[i] if i < len(responses) else {}
                
                verified_row = {
                    **row,
                    "verifier_label": v_res.get("verifier_label"),
                    "verifier_confidence": v_res.get("verifier_confidence", 0),
                    "hallucination_detected": v_res.get("hallucination_detected", False),
                    "logic_consistency_score": v_res.get("logic_consistency_score", 0),
                    "verifier_explanation": v_res.get("explanation", "No response from AI")
                }
                # Tính toán is_reliable dựa trên logic Evaluator
                verified_row["is_reliable"] = self.evaluator.calculate_reliability(verified_row)
                verified_results.append(verified_row)
                
            return verified_results

        except Exception as e:
            logging.error(f"Lỗi khi xử lý batch: {e}")
            return [{**r, "is_reliable": False, "error": str(e)} for r in chunk_rows]

    async def process_verification(self, input_csv: str, output_csv: str, samples_per_request: int = 5):
        df = pd.read_csv(input_csv)
        data = df.to_dict('records')
        all_results = []

        for i in range(0, len(data), samples_per_request):
            chunk = data[i : i + samples_per_request]
            logging.info(f"Đang xử lý mẫu {i} đến {i + len(chunk)}...")
            
            chunk_results = await self.verify_chunk(chunk)
            all_results.extend(chunk_results)
            
            if i + samples_per_request < len(data):
                await asyncio.sleep(5)

        verified_df = pd.DataFrame(all_results)
        verified_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
        
        # --- IN STATISTICS ---
        print("\n" + "="*50)
        print("📊 TỔNG KẾT KIỂM ĐỊNH (PHASE 2)")
        print("="*50)
        self.evaluator.get_statistics_metrics(verified_df)
        logging.info(f"✅ Hoàn tất! Đã lưu tại: {output_csv}")

def parse_args():
    parser = argparse.ArgumentParser(description="Teacher Knowledge Generation Pipeline")
    parser.add_argument('--input', type=str, required=True, help="Path tới file CSV đầu vào")
    parser.add_argument('--output', type=str, required=True, help="Path lưu file kết quả")
    parser.add_argument('--prompt', type=str, default='system_prompt.txt', help="Path tới file prompt hệ thống")
    parser.add_argument('--model', type=str, default='gpt-4o', help="Model OpenAI (gpt-4o, gpt-4-turbo)")
    parser.add_argument('--batch_size', type=int, default=10, help="Số lượng request gửi song song (Max concurrent)")
    parser.add_argument('--text_col', type=str, default='text', help="Tên cột chứa nội dung text")
    parser.add_argument('--emotion_col', type=str, default='emotion', help="Tên cột chứa cảm xúc")
    parser.add_argument('--size_pct', type=float, help="Lấy khoảng bao nhiêu dataset")
    parser.add_argument('--test', action='store_true', help="Chạy chế độ test với 5 dòng đầu tiên")
    return parser.parse_args()



# ======================= test_logic_phase_1 =======================
# ======================= test_logic_phase_1 =======================

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
    tmp = int(args.size_pct * len(process_df))
    process_df = process_df.iloc[:tmp]

    records = process_df.rename(
        columns={args.text_col: 'text', args.emotion_col: 'emotion'}
    ).to_dict('records')
    
    all_results = await annotator.annotate_batch(records, max_concurrent=args.batch_size)

    result_df = pd.DataFrame(all_results)
    result_df.to_csv(args.output, index=False, encoding='utf-8-sig')
    logging.info(f"Hoàn tất! Kết quả đã được lưu tại: {args.output}")


# ======================= test_logic_phase_2 =======================
# ======================= test_logic_phase_2 =======================

async def test_logic_phase_2():
    logging.info("--- ĐANG CHẠY TEST LOGIC PHASE 2 (VERIFIER) ---")
    
    # 1. Khởi tạo (Sử dụng model thực tế: claude-3-5-sonnet-latest)
    anthropic_llm = AnthropicProvider(
        api_key=os.environ['ANTHROPIC_API_KEY'], 
        model="claude-sonnet-4-6"
    )
    # Đảm bảo đường dẫn prompt đúng
    verifier = DataVerifier(anthropic_provider=anthropic_llm, system_prompt_path='/content/prompt_version_2.txt')

    mock_teacher_results = [
        # 1. MẪU CHUẨN: Đồng thuận giữa Teacher và Verifier
        {
            "input_text": "mng ơi mik mới mua cái đt mới xịn xò lắm lun",
            "input_emotion": "happy",
            "semantic_decoding": "Người dùng mua điện thoại mới",
            "slang_interpretation": "xịn xò = đồ tốt, chất lượng cao",
            "contextual_conflict": "Không có mâu thuẫn giữa lời nói và cảm xúc vui vẻ",
            "thought_trace": "Người dùng chia sẻ niềm vui mua đồ mới, mang tính xây dựng.",
            "final_label": "Constructive/Clean",
            "confidence_score": 0.95
        },
        
        # 2. MẪU SAI (HALLUCINATION): Teacher không hiểu từ lóng thô tục
        {
            "input_text": "clgt sao m lại làm thế vs t 😡",
            "input_emotion": "angry",
            "semantic_decoding": "Câu hỏi về hành động của người khác",
            "slang_interpretation": "clgt = viết tắt thông thường", # SAI: Teacher không giải mã được độ tục
            "contextual_conflict": "Cảm xúc giận dữ phù hợp với câu hỏi",
            "thought_trace": "Đây là lời hỏi thăm trong lúc nóng giận nhưng không thù địch.", # SAI: Logic yếu
            "final_label": "Constructive/Clean", # SAI: Đáng lẽ phải là Explicit Hostility
            "confidence_score": 0.88
        },

        # 3. MẪU SAI (SARCASM): Teacher bị đánh lừa bởi câu chữ "khen"
        {
            "input_text": "Giỏi quá vcl cả họ tự hào",
            "input_emotion": "smirk",
            "semantic_decoding": "Khen ngợi sự giỏi giang và niềm tự hào gia đình",
            "slang_interpretation": "vcl = vãi cả lồn (trạng từ nhấn mạnh)",
            "contextual_conflict": "Lời khen nhưng đi kèm biểu cảm cười nhếch mép (smirk)",
            "thought_trace": "Người nói đang khen ngợi đối phương một cách nhiệt tình.", # SAI: Bỏ qua tín hiệu mỉa mai
            "final_label": "Constructive/Clean", # SAI: Đáng lẽ phải là Implicit Toxicity
            "confidence_score": 0.82
        }
    ]


    # 3. Thực thi
    results = await verifier.verify_chunk(mock_teacher_results)

    df_res = pd.DataFrame(results)

    print("\n" + "="*30 + " CHI TIẾT KẾT QUẢ TEST " + "="*30)
    for res in results:
        print(f"\n[-] Text: {res['input_text']}")
        print(f"[-] Teacher: {res['final_label']} | Verifier: {res['verifier_label']}")
        print(f"[>] Logic Score: {res['logic_consistency_score']}/5 | Hallucination: {res['hallucination_detected']}")
        print(f"[!] Reliable: {'✅ YES' if res['is_reliable'] else '❌ NO'}")
        print(f"[*] Note: {res['verifier_explanation']}")
        print("-" * 60)

    # In thống kê
    verifier.get_statistics(df_res)
    return df_res

if __name__ == "__main__":
    args = parse_args()
    
    # Nếu có flag --test, chỉ chạy test_logic rồi thoát
    if args.test:
        asyncio.run(test_logic_phase_1(args))
    else:
        # Nếu không, chạy pipeline bình thường
        asyncio.run(run_ppipeline_phase_1(args))

# Constructive/Clean Constructive/Clean
# Constructive/Clean Implicit Toxicity
# Constructive/Clean Implicit Toxicity

# ============================== CHI TIẾT KẾT QUẢ TEST ==============================

# [-] Text: mng ơi mik mới mua cái đt mới xịn xò lắm lun
# [-] Teacher: Constructive/Clean | Verifier: Constructive/Clean
# [>] Logic Score: 5/5 | Hallucination: False
# [!] Reliable: ✅ YES
# [*] Note: Văn bản hoàn toàn tích cực: người dùng chia sẻ niềm vui mua điện thoại mới với cộng đồng ('mng ơi'). Slang 'xịn xò' được giải mã đúng (= chất lượng cao, tốt). Cảm xúc 'happy' nhất quán với nội dung. Không có dấu hiệu độc hại hay thù địch. Teacher AI phân tích chính xác, lập luận mạch lạc.
# ------------------------------------------------------------

# [-] Text: clgt sao m lại làm thế vs t 😡
# [-] Teacher: Constructive/Clean | Verifier: Implicit Toxicity
# [>] Logic Score: 2/5 | Hallucination: True
# [!] Reliable: ❌ NO
# [*] Note: Teacher AI mắc lỗi nghiêm trọng khi giải mã 'clgt': đây là viết tắt của 'cái lồn gì thế' – một cụm từ tục tĩu, mang tính xúc phạm ngầm, KHÔNG phải 'viết tắt thông thường' như Teacher mô tả. Kết hợp với 'm' (mày) và cảm xúc giận dữ 😡, câu này thể hiện sự đối đầu và xúc phạm ngầm đối với người nghe. Teacher AI đã hallucinate bằng cách bỏ qua nghĩa tục của 'clgt' và kết luận sai là 'Constructive/Clean'. Nhãn đúng phải là Implicit Toxicity (hoặc có thể tranh luận là Explicit Hostility tùy ngữ cảnh), nhưng chắc chắn không phải Clean.
# ------------------------------------------------------------

# [-] Text: Giỏi quá vcl cả họ tự hào
# [-] Teacher: Constructive/Clean | Verifier: Implicit Toxicity
# [>] Logic Score: 2/5 | Hallucination: True
# [!] Reliable: ❌ NO
# [*] Note: Teacher AI bỏ qua tín hiệu quan trọng nhất: cảm xúc 'smirk' (cười nhếch mép) kết hợp với cấu trúc khen ngợi cường điệu 'Giỏi quá vcl cả họ tự hào' là dấu hiệu điển hình của mỉa mai/châm biếm trong tiếng Việt mạng xã hội. Khi lời khen đi kèm smirk, đây gần như chắc chắn là lời khen giả tạo (sarcasm), ngụ ý chế giễu. Teacher nhận ra mâu thuẫn giữa lời khen và smirk trong phần scaffolding nhưng lại kết luận ngược lại ('khen ngợi nhiệt tình') trong thought_trace – đây là mâu thuẫn nội tại rõ ràng và có thể coi là hallucination logic. Nhãn phù hợp là Implicit Toxicity (châm biếm/mỉa mai ẩn).
# ------------------------------------------------------------

# ------------------------------
# 📊 THỐNG KÊ CHẤT LƯỢNG DATASET
# ------------------------------
# total_samples            : 3
# reliable_samples         : 1
# reliability_rate         : 0.3333
# avg_teacher_confidence   : 0.8833
# avg_verifier_confidence  : 0.8900
# hallucination_rate       : 0.6667
# avg_logic_score          : 3.0000
# agreement_rate           : 0.3333
# ------------------------------