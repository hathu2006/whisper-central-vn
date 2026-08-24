"""
data_utils.py — Các thành phần dùng chung giữa 03_finetune_whisper.py và
04_evaluate_wer.py, tách riêng để 2 script không phải copy-paste code.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Data collator cho bài toán speech-to-text của Whisper.

    Vì sao cần collator RIÊNG (không dùng default collator của Trainer)?
    input_features (spectrogram) và labels (token ids) có cách "pad" khác
    nhau và cần xử lý riêng biệt:
      - input_features: mỗi mẫu đã được feature_extractor pad sẵn về đúng
        1 độ dài cố định (30s) ở bước 02, nên chỉ cần gom lại thành tensor
        (feature_extractor.pad lo phần này).
      - labels: độ dài câu nói khác nhau -> cần pad về cùng độ dài trong
        1 batch bằng pad_token_id, SAU ĐÓ đổi các vị trí pad thành -100
        (giá trị PyTorch CrossEntropyLoss dùng để "bỏ qua, không tính loss").
        Nếu không đổi thành -100, model sẽ bị ép học dự đoán ra token pad
        vô nghĩa ở cuối câu, làm nhiễu quá trình học.
    """

    processor: Any  # WhisperProcessor (gồm feature_extractor + tokenizer)
    decoder_start_token_id: int

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        # --- xử lý phần input_features (audio) ---
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # --- xử lý phần labels (text) ---
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # thay pad_token bằng -100 để loss function bỏ qua các vị trí này
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # Whisper tự thêm decoder_start_token vào đầu chuỗi khi generate,
        # nên nếu label đã có sẵn token này ở đầu (do tokenizer thêm), ta cắt
        # bỏ để tránh bị lặp/mất lệch 1 vị trí khi tính loss.
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


# Các ký tự dấu câu cần bỏ trước khi tính WER/CER. CHỈ liệt kê dấu câu
# (không đụng tới chữ cái có dấu thanh điệu tiếng Việt như á, à, ả, ã, ạ...
# — những ký tự đó là 1 phần của TỪ, khác hoàn toàn với dấu câu ở đây).
_PUNCTUATION_PATTERN = re.compile(r"[.,!?;:\"'“”‘’()\[\]{}…\-–—]")


def normalize_text(text: str) -> str:
    """
    Chuẩn hóa text trước khi tính WER/CER: lowercase + bỏ dấu câu + xóa
    khoảng trắng thừa.

    Lý do cần bỏ dấu câu (không chỉ lowercase): Whisper có thể sinh ra dấu
    câu ở vị trí hơi khác transcript gốc (vd. model ra "đó," còn transcript
    gốc là "đó" — cùng 1 từ, chỉ khác dấu phẩy dính liền). Nếu không bỏ dấu
    câu, WER sẽ tính "đó," và "đó" là 2 TỪ KHÁC NHAU hoàn toàn (thay vì nhận
    ra đây là cùng 1 từ đúng), làm WER bị thổi phồng giả tạo — không phản
    ánh đúng chất lượng nhận diện NỘI DUNG (mục tiêu chính của bài toán ASR
    này), chỉ phản ánh sai khác về dấu câu vốn không quan trọng bằng.

    Áp dụng chuẩn hóa NHƯ NHAU cho cả reference lẫn hypothesis để so sánh
    công bằng.
    """
    text = text.lower()
    text = _PUNCTUATION_PATTERN.sub(" ", text)
    return " ".join(text.split())
