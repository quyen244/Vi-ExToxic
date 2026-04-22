"""
Evaluation Module
Evaluate and compare models for Multi-Label Classification
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Union
from datetime import datetime
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    hamming_loss, multilabel_confusion_matrix
)

from config import FINAL_LABELS, OUTPUT_DIR


class EvaluatorModel:
    """Evaluate and compare model results (Multi-label)"""
    
    def __init__(self, labels: List[str] = FINAL_LABELS):
        self.labels = labels
        self.results: List[Dict[str, Any]] = []
    
    def _convert_to_binary(self, y: List[List[str]]) -> np.ndarray:
        """Convert list of label strings to binary matrix"""
        binary = []
        for labels_list in y:
            row = [0] * len(self.labels)
            for label in labels_list:
                if label in self.labels:
                    idx = self.labels.index(label)
                    row[idx] = 1
            binary.append(row)
        return np.array(binary)
    
    def evaluate(self, y_true, y_pred, model_name: str, dataset_version: str) -> Dict[str, Any]:
        """
        Evaluate a model on a dataset version (Multi-label)
        
        Args:
            y_true: Ground truth labels (list of label strings or binary matrix)
            y_pred: Predicted labels (list of label strings or binary matrix)
        """
        # Convert to binary matrix if needed
        if len(y_true) > 0 and isinstance(y_true[0], list) and len(y_true[0]) > 0 and isinstance(y_true[0][0], str):
            y_true_binary = self._convert_to_binary(y_true)
        else:
            y_true_binary = np.array(y_true)
            
        if len(y_pred) > 0 and isinstance(y_pred[0], list) and len(y_pred[0]) > 0 and isinstance(y_pred[0][0], str):
            y_pred_binary = self._convert_to_binary(y_pred)
        else:
            y_pred_binary = np.array(y_pred)
        
        # Metrics
        subset_accuracy = accuracy_score(y_true_binary, y_pred_binary)
        h_loss = hamming_loss(y_true_binary, y_pred_binary)
        f1_samples = f1_score(y_true_binary, y_pred_binary, average='samples', zero_division=0)
        precision_samples = precision_score(y_true_binary, y_pred_binary, average='samples', zero_division=0)
        recall_samples = recall_score(y_true_binary, y_pred_binary, average='samples', zero_division=0)
        f1_macro = f1_score(y_true_binary, y_pred_binary, average='macro', zero_division=0)
        f1_micro = f1_score(y_true_binary, y_pred_binary, average='micro', zero_division=0)
        precision_macro = precision_score(y_true_binary, y_pred_binary, average='macro', zero_division=0)
        precision_micro = precision_score(y_true_binary, y_pred_binary, average='micro', zero_division=0)
        recall_macro = recall_score(y_true_binary, y_pred_binary, average='macro', zero_division=0)
        recall_micro = recall_score(y_true_binary, y_pred_binary, average='micro', zero_division=0)
        
        # Per-label F1, Precision, Recall
        f1_per_label = f1_score(y_true_binary, y_pred_binary, average=None, zero_division=0)
        precision_per_label = precision_score(y_true_binary, y_pred_binary, average=None, zero_division=0)
        recall_per_label = recall_score(y_true_binary, y_pred_binary, average=None, zero_division=0)
        
        f1_per_label_dict = {}
        precision_per_label_dict = {}
        recall_per_label_dict = {}
        
        for i, label in enumerate(self.labels):
            f1_per_label_dict[label] = float(f1_per_label[i])
            precision_per_label_dict[label] = float(precision_per_label[i])
            recall_per_label_dict[label] = float(recall_per_label[i])
        
        result = {
            'model': model_name,
            'dataset': dataset_version,
            'subset_accuracy': float(subset_accuracy),
            'hamming_loss': float(h_loss),
            'f1_samples': float(f1_samples),
            'f1_macro': float(f1_macro),
            'f1_micro': float(f1_micro),
            'precision_samples': float(precision_samples),
            'precision_macro': float(precision_macro),
            'precision_micro': float(precision_micro),
            'recall_samples': float(recall_samples),
            'recall_macro': float(recall_macro),
            'recall_micro': float(recall_micro),
            'f1_per_label': f1_per_label_dict,
            'precision_per_label': precision_per_label_dict,
            'recall_per_label': recall_per_label_dict,
            'timestamp': datetime.now().isoformat(),
            'n_samples': len(y_true),
        }
        
        self.results.append(result)
        return result
    
    def print_result(self, result: Dict[str, Any]):
        """Print formatted result"""
        print(f"\n{'='*70}")
        print(f" {result['model']} on Dataset {result['dataset']} (Multi-Label)")
        print(f"{'='*70}")
        print(f"  Subset Accuracy (Exact Match): {result['subset_accuracy']:.4f}")
        print(f"  Hamming Loss:                  {result['hamming_loss']:.4f}")
        print(f"  ")
        print(f"  F1 Samples:                    {result['f1_samples']:.4f}")
        print(f"  F1 Macro:                      {result['f1_macro']:.4f}")
        print(f"  F1 Micro:                      {result['f1_micro']:.4f}")
        print(f"  ")
        print(f"  Precision (Samples):           {result['precision_samples']:.4f}")
        print(f"  Precision (Macro):             {result['precision_macro']:.4f}")
        print(f"  Precision (Micro):             {result['precision_micro']:.4f}")
        print(f"  ")
        print(f"  Recall (Samples):              {result['recall_samples']:.4f}")
        print(f"  Recall (Macro):                {result['recall_macro']:.4f}")
        print(f"  Recall (Micro):                {result['recall_micro']:.4f}")
        
        print(f"\n  F1 per Label:")
        for label, f1 in result['f1_per_label'].items():
            if f1 > 0:
                print(f"    {label:25s}: {f1:.4f}")
    
    def get_comparison_matrix(self) -> pd.DataFrame:
        """Create comparison matrix"""
        if not self.results:
            return pd.DataFrame()
        df = pd.DataFrame(self.results)
        matrix = df.pivot(index='model', columns='dataset', values='f1_macro')
        return matrix
    
    def get_full_comparison(self) -> pd.DataFrame:
        """Create full comparison table"""
        if not self.results:
            return pd.DataFrame()
        rows = []
        for r in self.results:
            rows.append({
                'Model': r['model'],
                'Dataset': r['dataset'],
                'Subset Acc': f"{r['subset_accuracy']:.4f}",
                'Hamming': f"{r['hamming_loss']:.4f}",
                'F1 Samples': f"{r['f1_samples']:.4f}",
                'F1 Macro': f"{r['f1_macro']:.4f}",
                'F1 Micro': f"{r['f1_micro']:.4f}",
            })
        return pd.DataFrame(rows)
    
    def save_results(self, output_dir: Path = OUTPUT_DIR):
        """Save results to file"""
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / "experiment_results.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        print(f"Saved results to {json_path}")
        
        matrix = self.get_comparison_matrix()
        if not matrix.empty:
            csv_path = output_dir / "comparison_matrix.csv"
            matrix.to_csv(csv_path)
            print(f"Saved comparison matrix to {csv_path}")
        
        full_df = self.get_full_comparison()
        if not full_df.empty:
            csv_path = output_dir / "full_comparison.csv"
            full_df.to_csv(csv_path, index=False)
            print(f"Saved full comparison to {csv_path}")
    
    def generate_report(self) -> str:
        """Generate text report"""
        report = []
        report.append("=" * 80)
        report.append("MULTI-LABEL EXPERIMENT RESULTS REPORT")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 80)
        
        report.append("\n COMPARISON MATRIX (F1 Macro)")
        report.append("-" * 60)
        matrix = self.get_comparison_matrix()
        if not matrix.empty:
            report.append(matrix.to_string())
        
        report.append("\n\n FULL COMPARISON")
        report.append("-" * 60)
        full_df = self.get_full_comparison()
        if not full_df.empty:
            report.append(full_df.to_string(index=False))
        
        report.append("\n\n DETAILED RESULTS")
        report.append("-" * 60)
        for r in self.results:
            report.append(f"\n{r['model']} - Dataset {r['dataset']}:")
            report.append(f"  Subset Accuracy: {r['subset_accuracy']:.4f}")
            report.append(f"  Hamming Loss: {r['hamming_loss']:.4f}")
            report.append(f"  F1 Macro: {r['f1_macro']:.4f}")
        
        if self.results:
            best = max(self.results, key=lambda x: x['f1_macro'])
            report.append(f"\n\n ANALYSIS")
            report.append(f"Best Model: {best['model']} on Dataset {best['dataset']} (F1: {best['f1_macro']:.4f})")
        
        return "\n".join(report)

# ==========================================
# 3. ĐỘ ĐO & LỌC DỮ LIỆU (Metric Scoring)
# ==========================================
class EvaluatorData:
    """Đánh giá chất lượng dữ liệu và sự thống nhất giữa các Agent"""

    @staticmethod
    def calculate_reliability(row: Dict) -> bool:
        """
        Kiểm tra bản ghi có đủ độ tin cậy để làm nhãn chuẩn (Gold Label) hay không.
        """
        try:
            # 1. Kiểm tra sự khớp nhau giữa Teacher và Verifier
            t_label = set(row.get('final_label', [])) if isinstance(row.get('final_label'), list) else {row.get('final_label')}
            v_label = set(row.get('verifier_label', [])) if isinstance(row.get('verifier_label'), list) else {row.get('verifier_label')}
            is_match = t_label == v_label
            
            # 2. Kiểm tra ảo giác
            no_hallucination = not row.get('hallucination_detected', True)
            
            # 3. Điểm logic (Consistency)
            logic_score = row.get('logic_consistency_score', 0)
            logic_ok = float(logic_score) >= 4.0
            
            # 4. Độ tự tin (Confidence)
            avg_conf = (float(row.get('confidence_score', 0)) + float(row.get('verifier_confidence', 0))) / 2
            confidence_ok = avg_conf >= 0.8
            
            return is_match and no_hallucination and logic_ok and confidence_ok
        except Exception:
            return False

    @staticmethod
    def get_statistics_metrics(df: pd.DataFrame) -> Dict[str, Any]:
        """Thống kê chi tiết chất lượng của toàn bộ dataset"""
        total = len(df)
        if total == 0: return {}

        # Thêm cột tin cậy nếu chưa có
        if 'is_reliable' not in df.columns:
            df['is_reliable'] = df.apply(EvaluatorData.calculate_reliability, axis=1)

        stats = {
            'total_samples': total,
            'reliable_samples': int(df['is_reliable'].sum()),
            'reliability_rate': float(df['is_reliable'].mean()),
            'avg_teacher_confidence': float(df['confidence_score'].mean()),
            'avg_verifier_confidence': float(df.get('verifier_confidence', 0).mean()),
            'hallucination_rate': float(df.get('hallucination_detected', 0).mean()),
            # Tỷ lệ Teacher và Verifier cãi nhau
            'disagreement_rate': 1 - (df['final_label'] == df['verifier_label']).mean()
        }

        print("\n" + "-"*30)
        print("📊 THỐNG KÊ CHẤT LƯỢNG DATASET")
        print("-"*30)
        for k, v in stats.items():
            print(f"{k:25}: {v:.4f}" if isinstance(v, float) else f"{k:25}: {v}")
        
        return stats
    
def test_evaluator_data():
    print("=== ĐANG CHẠY TEST CASE CHO EVALUATOR DATA ===\n")

    # 1. TẠO DỮ LIỆU MOCK (Dựa trên CSV của bạn nhưng thêm các cột Verifier)
    # Chúng ta sẽ tạo 4 trường hợp điển hình:
    mock_data = [
        {
            "input_text": "mng ơi mik mới mua cái đt mới xịn xò lắm lun",
            "final_label": "Constructive/Clean",
            "verifier_label": "Constructive/Clean", # Khớp nhãn
            "confidence_score": 0.98,
            "verifier_confidence": 0.95,           # Trung bình > 0.8
            "hallucination_detected": False,       # Không ảo giác
            "logic_consistency_score": 5,          # Logic tốt
        }, # => KẾT QUẢ MONG ĐỢI: True (Reliable)
        
        {
            "input_text": "clgt sao m lại làm thế vs t 😡",
            "final_label": "Explicit Hostility",
            "verifier_label": "Constructive/Clean", # SAI KHÁC NHÃN
            "confidence_score": 0.9,
            "verifier_confidence": 0.8,
            "hallucination_detected": False,
            "logic_consistency_score": 4,
        }, # => KẾT QUẢ MONG ĐỢI: False (Disagreement)

        {
            "input_text": "Hôm nay t đi học trễ vcl 😂😂😂",
            "final_label": "Explicit Hostility",
            "verifier_label": "Explicit Hostility",
            "confidence_score": 0.7,               # ĐỘ TỰ TIN THẤP (0.7+0.7)/2 = 0.7 < 0.8
            "verifier_confidence": 0.7,
            "hallucination_detected": False,
            "logic_consistency_score": 4,
        }, # => KẾT QUẢ MONG ĐỢI: False (Low confidence)

        {
            "input_text": "Giỏi quá vcl cả họ tự hào smirk",
            "final_label": "Implicit Toxicity",
            "verifier_label": "Implicit Toxicity",
            "confidence_score": 0.95,
            "verifier_confidence": 0.9,
            "hallucination_detected": True,        # CÓ ẢO GIÁC
            "logic_consistency_score": 5,
        }  # => KẾT QUẢ MONG ĐỢI: False (Hallucination)
    ]

    df_test = pd.DataFrame(mock_data)

    # 2. TEST HÀM 1: calculate_reliability
    print("--- Test: calculate_reliability ---")
    for i, row in enumerate(mock_data):
        is_reliable = EvaluatorData.calculate_reliability(row)
        status = "✅ PASS" if (i == 0 and is_reliable) or (i > 0 and not is_reliable) else "❌ FAIL"
        print(f"Sample {i+1}: Reliable={is_reliable} | {status}")

    # 3. TEST HÀM 2: get_statistics_metrics
    print("\n--- Test: get_statistics_metrics ---")
    stats = EvaluatorData.get_statistics_metrics(df_test)
    
    # Kiểm tra các chỉ số quan trọng
    assert stats['total_samples'] == 4
    assert stats['reliable_samples'] == 1
    assert stats['disagreement_rate'] == 0.25 # 1/4 mẫu bị lệch nhãn
    print("\n=> Kiểm tra Assertions: Hoàn tất (Dữ liệu thống kê chính xác)")

if __name__ == "__main__":
    test_evaluator_data()