"""Prompt construction and supervised target formatting."""

from __future__ import annotations

import json
from typing import Any

from .data import LABELS, RATIONALE_COLUMNS, canonical_target_mode

TARGET_MODES = ("no-cot", "cot", "cot-distilled", "cot-en")
PROMPT_VERSION = "phase4-vi-en-cot-v1"

LABEL_DEFINITIONS = {
    "Constructive/Clean": "Thông tin bình thường, phản hồi tích cực hoặc chia sẻ cảm xúc cá nhân vô hại.",
    "Implicit Toxicity": "Mỉa mai, châm biếm hoặc lời lẽ có vẻ tích cực nhưng nhằm xúc phạm hay công kích.",
    "Explicit Hostility": "Lăng mạ trực tiếp, chửi tục, ngôn từ xúc phạm rõ ràng hoặc đe dọa.",
    "Identity-Based Hate": "Miệt thị, thù ghét hoặc phân biệt đối xử dựa trên vùng miền, giới tính, tôn giáo hoặc ngoại hình/cơ thể.",
    "Ambiguous/Noise": "Quá ngắn, chỉ có emoji, sai chính tả nặng hoặc thiếu ngữ cảnh đến mức không thể diễn giải đáng tin cậy.",
}

_LABEL_GUIDE_VI = "\n".join(f"- {label}: {LABEL_DEFINITIONS[label]}" for label in LABELS)
_LABEL_DEFINITIONS_EN = {
    "Constructive/Clean": "Harmless information, positive responses, or personal feelings.",
    "Implicit Toxicity": "Sarcasm or seemingly positive wording intended to insult or attack.",
    "Explicit Hostility": "Direct insults, profanity, offensive language, or threats.",
    "Identity-Based Hate": "Hate or discrimination based on region, gender, religion, appearance, or body.",
    "Ambiguous/Noise": "Content too short, noisy, or unclear to interpret reliably.",
}
_LABEL_GUIDE_EN = "\n".join(f"- {label}: {_LABEL_DEFINITIONS_EN[label]}" for label in LABELS)
SYSTEM_PROMPT_VI = (
    "Bạn là chuyên gia ngôn ngữ học về văn hóa mạng xã hội Việt Nam và chuyên gia kiểm duyệt nội dung. "
    "Bạn giỏi giải mã teencode, tiếng lóng Gen Z và nhận biết mỉa mai. "
    "Phân tích văn bản cùng cảm xúc được cung cấp và chọn đúng một nhãn. Cảm xúc là ngữ cảnh, "
    "không tự nó quyết định nhãn.\n\n"
    "Hệ thống nhãn:\n"
    f"{_LABEL_GUIDE_VI}\n\n"
    "Trả lời bằng tiếng Việt và chỉ xuất JSON hợp lệ theo đúng schema được yêu cầu, không thêm markdown."
)
SYSTEM_PROMPT_EN = (
    "You are an expert in Vietnamese social media language and content moderation. "
    "You can interpret Vietnamese teencode, slang, and sarcasm. Analyze the text together "
    "with the supplied emotion and select exactly one label. Emotion provides context and "
    "does not determine the label by itself.\n\n"
    "Label definitions:\n"
    f"{_LABEL_GUIDE_EN}\n\n"
    "Return only valid JSON according to the requested schema, without Markdown. "
    "Write natural-language field values in Vietnamese because the task and teacher annotations "
    "are Vietnamese. Keep final_label exactly equal to one of the listed label names."
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
            '2. Quy trình suy luận: Hãy phân tích theo các bước sau. '
            'Bước 1: Giải mã ngữ nghĩa và tiếng lóng. Giải mã tiếng lóng, teencode, '
            'nghĩa đen và nghĩa bóng của câu. '
            'Bước 2: Đối chiếu ngữ cảnh. So sánh nội dung văn bản với cảm xúc đi kèm. '
            'Nếu hai yếu tố mâu thuẫn, hãy cân nhắc khả năng văn bản mang ý mỉa mai. '
            'Bước 3: Xác định đối tượng. Xác định câu nói hướng đến người nói, một cá nhân khác, '
            'hay một nhóm cụ thể. '
            '"thought_trace" là lập luận logic cuối cùng tổng hợp các bước trên trước khi gán nhãn. '
            '"confidence_score" là mức độ tự tin vào nhãn đã chọn, biểu diễn bằng số từ 0.0 đến 1.0. '
            '3. Định dạng đầu ra (JSON): Chỉ trả về JSON hợp lệ theo mẫu: '
            '{"thought_trace":"Lập luận logic cuối cùng trước khi gán nhãn...",'
            '"final_label":"<nhãn>","confidence_score":0.0}.'
        )
    return (
        '2. Quy trình suy luận (Reasoning Scaffolding): Bắt buộc phân tích theo các bước sau. '
        'Bước 1: Giải mã ngữ nghĩa và tiếng lóng. Giải mã tiếng lóng, teencode, nghĩa đen hoặc '
        'nghĩa bóng của câu. Ghi cách dùng từ và cấu trúc câu vào semantic_decoding; ghi tiếng lóng '
        'hoặc teencode vào slang_interpretation. '
        'Bước 2: Đối chiếu ngữ cảnh. So sánh nội dung văn bản với cảm xúc đi kèm. Ví dụ, nếu cảm xúc '
        'là tức giận nhưng văn bản lại khen ngợi, hãy cân nhắc khả năng mỉa mai. Ghi phân tích vào '
        'contextual_conflict. '
        'Bước 3: Xác định đối tượng. Xác định câu nói hướng đến người nói, một cá nhân khác, hay một '
        'nhóm cụ thể. Ghi đối tượng vào target. '
        'Bước 4: Lập luận tổng hợp. Tổng hợp logic từ các bước trên thành chuỗi lập luận trước khi gán '
        'nhãn và ghi vào thought_trace. '
        'confidence_score là mức độ tự tin vào nhãn đã chọn, biểu diễn bằng số từ 0.0 đến 1.0. '
        '3. Định dạng đầu ra (JSON): Chỉ trả về JSON hợp lệ theo mẫu: '
        '{"reasoning_scaffolding":{"semantic_decoding":"Giải mã cách dùng từ và cấu trúc câu...",'
        '"slang_interpretation":"Giải thích tiếng lóng hoặc teencode...",'
        '"contextual_conflict":"Phân tích sự phù hợp giữa văn bản và cảm xúc...",'
        '"target":"Xác định đối tượng..."},'
        '"thought_trace":"Lập luận logic cuối cùng trước khi gán nhãn...",'
        '"final_label":"<nhãn>","confidence_score":0.0}.'
    )


def _user_message(example: dict[str, Any], language: str) -> str:
    if language == "en":
        return (
            "Analyze the user's content based on the supplied text and emotion.\n"
            f"Emotion: {example['input_emotion'] or 'Unknown'}\n"
            f"Text: {example['input_text']}"
        )
    return (
        "Hãy phân tích nội dung người dùng dựa trên văn bản và cảm xúc được cung cấp.\n"
        f"Emotion: {example['input_emotion'] or 'Không rõ'}\n"
        f"Text: {example['input_text']}"
    )


def to_prompt_completion(
    example: dict[str, Any],
    target_mode: str,
    few_shot_examples: list[dict[str, Any]] | None = None,
) -> dict[str, str]:
    """Format one row as a Qwen-compatible prompt/completion pair."""
    mode = canonical_target_mode(target_mode)
    prompt_language = "en" if target_mode == "cot-en" else "vi"
    if prompt_language == "en":
        instruction = _output_instruction_en(mode)
        system_prompt = SYSTEM_PROMPT_EN
    else:
        instruction = _output_instruction(mode)
        system_prompt = SYSTEM_PROMPT_VI
    examples = few_shot_examples or []
    if examples and target_mode == "cot-en":
        raise ValueError("Few-shot evaluation is enabled only for Vietnamese prompt modes")
    prompt = f"<|im_start|>system\n{system_prompt}\n{instruction}<|im_end|>\n"
    for demonstration in examples:
        demo_user = _user_message(demonstration, prompt_language)
        demo_target = json.dumps(_target(demonstration, mode), ensure_ascii=False, separators=(",", ":"))
        prompt += (
            f"<|im_start|>user\n{demo_user}<|im_end|>\n"
            f"<|im_start|>assistant\n{demo_target}<|im_end|>\n"
        )
    user = _user_message(example, prompt_language)
    prompt += f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
    completion = json.dumps(_target(example, mode), ensure_ascii=False, separators=(",", ":"))
    return {"prompt": prompt, "completion": completion + "<|im_end|>"}


def _output_instruction_en(mode: str) -> str:
    if mode != "cot":
        raise ValueError("English prompt language is currently supported only for the cot experiment")
    return (
        "2. Reasoning Process: Analyze through the following steps. "
        "Step 1: Semantic & Slang Decoding. Decode slang, teencode, and the literal or figurative "
        "meanings of the sentence. "
        "Step 2: Contextual Conflict. Compare the text with the associated emotion. If they conflict, "
        "consider whether the text may be sarcastic. "
        "Step 3: Target Identification. Identify whether the statement is directed at the speaker, "
        "another individual, or a specific group. "
        '"thought_trace" is the final logical reasoning that synthesizes these steps before assigning '
        'a label. "confidence_score" is your confidence in the assigned label, represented as a number '
        "from 0.0 to 1.0. "
        "3. Output Format (JSON): Return only valid JSON: "
        '{"thought_trace":"Final logical reasoning before assigning label...",'
        '"final_label":"<one exact label from the list>","confidence_score":0.0}.'
    )
