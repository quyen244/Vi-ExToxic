"""Prompt construction and supervised target formatting."""

from __future__ import annotations

import json

from .data import LABELS, RATIONALE_COLUMNS, canonical_target_mode

TARGET_MODES = ("no-cot", "cot", "cot-distilled")

LABEL_DEFINITIONS = {
    "Constructive/Clean": "Thông tin bình thường, phản hồi tích cực hoặc chia sẻ cảm xúc cá nhân vô hại.",
    "Implicit Toxicity": "Mỉa mai, châm biếm hoặc lời lẽ có vẻ tích cực nhưng nhằm xúc phạm hay công kích.",
    "Explicit Hostility": "Lăng mạ trực tiếp, chửi tục, ngôn từ xúc phạm rõ ràng hoặc đe dọa.",
    "Identity-Based Hate": "Miệt thị, thù ghét hoặc phân biệt đối xử dựa trên vùng miền, giới tính, tôn giáo hoặc ngoại hình/cơ thể.",
    "Ambiguous/Noise": "Quá ngắn, chỉ có emoji, sai chính tả nặng hoặc thiếu ngữ cảnh đến mức không thể diễn giải đáng tin cậy.",
}

_LABEL_GUIDE = "\n".join(f"- {label}: {LABEL_DEFINITIONS[label]}" for label in LABELS)
SYSTEM_PROMPT = (
    "Bạn là chuyên gia ngôn ngữ học về văn hóa mạng xã hội Việt Nam và chuyên gia kiểm duyệt nội dung. "
    "Bạn giỏi giải mã teencode, tiếng lóng Gen Z và nhận biết mỉa mai. "
    "Phân tích văn bản cùng cảm xúc được cung cấp và chọn đúng một nhãn. Cảm xúc là ngữ cảnh, "
    "không tự nó quyết định nhãn.\n\n"
    "Hệ thống nhãn:\n"
    f"{_LABEL_GUIDE}\n\n"
    "Trả lời bằng tiếng Việt và chỉ xuất JSON hợp lệ theo đúng schema được yêu cầu, không thêm markdown."
)


def _confidence(example: dict[str, str]) -> float:
    try:
        confidence = float(example["confidence_score"])
    except (KeyError, ValueError) as exc:
        raise ValueError("CoT targets require a numeric confidence_score") from exc
    if not 0.0 <= confidence <= 1.0:
        raise ValueError(f"confidence_score must be in [0, 1], got {confidence}")
    return confidence


def _target(example: dict[str, str], mode: str) -> dict[str, object]:
    if mode == "no-cot":
        return {"final_label": example["final_label"]}

    answer: dict[str, object] = {
        "thought_trace": example["thought_trace"],
        "final_label": example["final_label"],
        "confidence_score": _confidence(example),
    }
    if mode == "cot-distilled":
        scaffolding = {key: example[key] for key in RATIONALE_COLUMNS[:4]}
        answer = {
            "reasoning_scaffolding": scaffolding,
            **answer,
        }
    return answer


def _output_instruction(mode: str) -> str:
    if mode == "no-cot":
        return 'Chỉ phân loại, không trình bày lập luận. Schema JSON: {"final_label":"<nhãn>"}.'
    if mode == "cot":
        return (
            'Hãy nêu ngắn gọn lập luận tổng hợp trong "thought_trace" rồi kết luận. '
            'Schema JSON: {"thought_trace":"...","final_label":"<nhãn>",'
            '"confidence_score":0.0}. confidence_score là số từ 0.0 đến 1.0.'
        )
    return (
        'Hãy điền từng bước reasoning_scaffolding (semantic_decoding, slang_interpretation, '
        'contextual_conflict, target), sau đó tổng hợp thành thought_trace và kết luận. '
        'Schema JSON: {"reasoning_scaffolding":{"semantic_decoding":"...",'
        '"slang_interpretation":"...","contextual_conflict":"...","target":"..."},'
        '"thought_trace":"...","final_label":"<nhãn>","confidence_score":0.0}. '
        'confidence_score là số từ 0.0 đến 1.0.'
    )


def to_prompt_completion(example: dict[str, str], target_mode: str) -> dict[str, str]:
    """Format one row as a Qwen-compatible prompt/completion pair."""
    mode = canonical_target_mode(target_mode)
    user = (
        "Hãy phân tích nội dung người dùng dựa trên văn bản và cảm xúc được cung cấp.\n"
        f"Emotion: {example['input_emotion'] or 'Không rõ'}\n"
        f"Text: {example['input_text']}"
    )
    prompt = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}\n{_output_instruction(mode)}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
    )
    completion = json.dumps(_target(example, mode), ensure_ascii=False, separators=(",", ":"))
    return {"prompt": prompt, "completion": completion + "<|im_end|>"}
