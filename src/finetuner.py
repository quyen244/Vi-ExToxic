import torch
import argparse
import pandas as pd
import json
import matplotlib.pyplot as plt
from datetime import datetime
from unsloth import FastLanguageModel
from trl import SFTTrainer
from trl import SFTTrainer, SFTConfig 
from datasets import Dataset

# ==========================================
# 1. CLASS WRAPPER & INHERITANCE
# ==========================================
class QwenFineTuner:
    def __init__(self, model_name="unsloth/Qwen2.5-3B-Instruct-bnb-4bit", max_seq_length=2000):
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.model = None
        self.tokenizer = None
        self.history = None 

        # Khởi tạo Model và Tokenizer
        self._setup_model()

    def _setup_model(self):
        print(f"🚀 Loading model: {self.model_name}...")
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name = self.model_name,
            max_seq_length = self.max_seq_length,
            load_in_4bit = True,
        )
        
        # --- THAY BẰNG CẤU HÌNH CHUẨN ---
        self.tokenizer.padding_side = "right"
        self.tokenizer.truncation_side = "right"
        # Đảm bảo có pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def _apply_lora(self):
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r = 16,
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_alpha = 16,
            lora_dropout = 0,
            bias = "none",
            use_gradient_checkpointing = "unsloth",
            random_state = 3407,
        )

    def _prepare_data(self, csv_path):
        """Chuyển đổi CSV sang format ChatML cho Qwen"""
        df = pd.read_csv(csv_path)
        
        def format_chatml(row):
            # 1. Định nghĩa System Prompt khắt khe
            system_prompt = (
               """ You are a Vietnamese social media expert. Follow these steps to analyze the user's input:
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
            
            # 2. Xây dựng nội dung Assistant từ các cột của verified.csv
            # Chúng ta tái cấu trúc lại JSON giống hệt yêu cầu
            response_json = {
                "reasoning_scaffolding": {
                    "semantic_decoding": row.get("semantic_decoding", ""),
                    "slang_interpretation": row.get("slang_interpretation", ""),
                    "contextual_conflict": row.get("contextual_conflict", ""),
                    "target": row.get("target", "")
                },
                "thought_trace": row.get("thought_trace", ""),
                "final_label": row.get("final_label", "")
            }
    
            # Chuyển dict thành chuỗi JSON tiếng Việt chuẩn
            assistant_content = json.dumps(response_json, ensure_ascii=False, indent=2)
        
            # 3. Ghép vào ChatML
            text = f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            text += f"<|im_start|>user\nInput Text: '{row['input_text']}'\nInput Emotion: '{row['input_emotion']}'<|im_end|>\n"
            text += f"<|im_start|>assistant\n{assistant_content}<|im_end|>"
            
            return {"text": text}

        dataset = Dataset.from_pandas(df)
        dataset = dataset.map(format_chatml, remove_columns=dataset.column_names)
        return dataset

    # ==========================================
    # 2. TRAINING PROCESS
    # ==========================================
    def train(self, train_csv, output_dir, epochs=3, batch_size=2, lr=2e-4):
        self._apply_lora()
        dataset = self._prepare_data(train_csv)

        print(f"⚡ Starting Training for {epochs} epochs...")
        FastLanguageModel.for_training(model) # Enable for training!

        # 1. Sử dụng SFTConfig (Tránh lỗi AttributeError token)
        training_args = SFTConfig(
            per_device_train_batch_size = batch_size,
            gradient_accumulation_steps = 4,
            warmup_steps = 5,
            num_train_epochs = epochs,
            learning_rate = lr,
            logging_steps = 20,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "cosine",
            seed = 3407,
            output_dir = output_dir,
            report_to = "none",
            dataset_text_field = "text",
            max_seq_length = self.max_seq_length,
            save_steps = 400
        )

        # 2. Khởi tạo Trainer
        trainer = SFTTrainer(
            model = self.model,
            tokenizer = self.tokenizer,
            train_dataset = dataset,
            args = training_args,
            dataset_text_field = "text",
            max_seq_length = self.max_seq_length,
        )

        # 3. Chạy Training
        train_stats = trainer.train()
        self.history = trainer.state.log_history
        
        # Lưu model
        self.model.save_pretrained(f"{output_dir}/lora_model")
        self.tokenizer.save_pretrained(f"{output_dir}/lora_model")
        print("✅ Training Complete and Model Saved!")
        return train_stats

    # ==========================================
    # 4. REPORT & VISUALIZATION
    # ==========================================
    def export_reports(self, output_dir):
        """Xuất báo cáo ra CSV và JSONL"""
        log_df = pd.DataFrame([log for log in self.history if "loss" in log])
        
        # Lưu CSV
        csv_file = f"{output_dir}/train_logs.csv"
        log_df.to_csv(csv_file, index=False)
        
        # Lưu JSONL
        jsonl_file = f"{output_dir}/train_logs.jsonl"
        log_df.to_json(jsonl_file, orient='records', lines=True)
        
        print(f"📊 Reports saved to {csv_file} and {jsonl_file}")
        self._visualize(log_df, output_dir)

    def _visualize(self, df, output_dir):
        plt.figure(figsize=(10, 5))
        plt.plot(df['step'], df['loss'], label='Training Loss', color='blue')
        plt.title('Qwen-3B Fine-tuning Loss Curve')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{output_dir}/loss_curve.png")
        plt.show()
        print(f"🖼️ Visualization saved to {output_dir}/loss_curve.png")

# ==========================================
# 5. CLI ARGUMENT PARSER
# ==========================================
def args_parser():
    parser = argparse.ArgumentParser(description="Qwen-3B Fine-tuner Phase 3")
    parser.add_argument('--input', type=str, required=True, help="Path to verified.csv")
    parser.add_argument('--output_dir', type=str, default="./qwen_output", help="Output directory")
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=2e-4)
    return parser.parse_args()

# ==========================================
# MAIN EXECUTION
# ==========================================
async def main():
    # Giả lập sys.argv nếu chạy trong Notebook
    import sys
    if 'ipykernel' in sys.modules:
        args = argparse.Namespace(
            input="/kaggle/working/blanced_phase_1_dataset.csv", 
            output_dir="./trained_qwen_results",
            epochs=2,
            batch_size=3,
            lr=2e-5
        )
    else:
        args = args_parser()

    # Chạy quy trình
    qwen = QwenFineTuner()
    qwen.train(
        train_csv=args.input, 
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )
    
    # Xuất báo cáo
    qwen.export_reports(args.output_dir)