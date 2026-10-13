"""Prompt construction and supervised target formatting."""

from __future__ import annotations

import json

from .data import LABELS, RATIONALE_COLUMNS

SYSTEM_PROMPT = (
    "Bạn là chuyên gia phân tích bình luận mạng xã hội tiếng Việt. "
    "Phân loại nội dung thành đúng một nhãn trong danh sách: "
    + "; ".join(LABELS)
    + ". Trả về duy nhất một đối tượng JSON hợp lệ, không thêm markdown."
)


def to_prompt_completion(example: dict[str, str], target_mode: str) -> dict[str, str]:
    """Format a row as a Qwen-compatible prompt/completion pair."""
    if target_mode not in {"label-only", "distilled"}:
        raise ValueError(f"Unsupported target mode: {target_mode}")
    user = (
        "Hãy phân tích bình luận sau.\n"
        f"Cảm xúc được cung cấp: {example['input_emotion'] or 'Không rõ'}\n"
        f"Bình luận: {example['input_text']}"
    )
    prompt = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
    )
    if target_mode == "label-only":
        answer = {"final_label": example["final_label"]}
    else:
        scaffolding = {key: example.get(key, "") for key in RATIONALE_COLUMNS[:4]}
        answer = {
            "reasoning_scaffolding": scaffolding,
            "explanation": example.get("thought_trace", ""),
            "final_label": example["final_label"],
        }
    completion = json.dumps(answer, ensure_ascii=False, separators=(",", ":"))
    return {"prompt": prompt, "completion": completion + "<|im_end|>"}
