"""
data_utils.py — Các thành phần dùng chung giữa 03_finetune_whisper.py và
04_evaluate_wer.py, tách riêng để 2 script không phải copy-paste code.
"""

from __future__ import annotations

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


def normalize_text(text: str) -> str:
    """
    Chuẩn hóa text tối thiểu trước khi tính WER: lowercase + xóa khoảng
    trắng thừa. WER vốn nhạy cảm với khác biệt hoa/thường và dấu câu — nếu
    không chuẩn hóa, một câu đúng nội dung nhưng khác cách viết hoa/chấm câu
    sẽ bị tính là "sai", khiến con số WER không phản ánh đúng chất lượng
    nhận diện thực sự. Áp dụng chuẩn hóa NHƯ NHAU cho cả reference lẫn
    hypothesis để so sánh công bằng.
    """
    return " ".join(text.lower().split())
