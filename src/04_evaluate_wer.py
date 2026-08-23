"""
04_evaluate_wer.py
====================
BƯỚC 4: Đo WER (Word Error Rate) trên tập TEST cho:
    (a) model Whisper GỐC (openai/whisper-small, chưa fine-tune)
    (b) model đã FINE-TUNE (output của bước 03)
rồi in bảng so sánh.

WER là gì (nhắc lại ngắn gọn): tỉ lệ lỗi ở mức từ, tính bằng
    WER = (Substitutions + Deletions + Insertions) / (số từ trong câu đúng)
WER càng THẤP càng tốt. WER = 0% nghĩa là nhận diện hoàn hảo.

Chạy: python src/04_evaluate_wer.py
"""

from __future__ import annotations

import json
import os
import sys

import evaluate
import torch
from datasets import load_from_disk
from transformers import WhisperForConditionalGeneration, WhisperProcessor

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from data_utils import normalize_text

wer_metric = evaluate.load("wer")


def transcribe_dataset(model, processor, test_ds, device, batch_size=8):
    """
    Chạy model.generate() theo từng batch trên tập test, trả về
    (list câu dự đoán, list câu đúng) đã chuẩn hóa (normalize_text).

    Lưu ý: test_ds ở đây là dataset ĐÃ tiền xử lý (có sẵn input_features và
    labels dạng token id từ bước 02), nên ta decode labels lại thành text để
    so sánh, thay vì đọc lại text thô — cách này đảm bảo model gốc và model
    fine-tune được đánh giá trên cùng 1 bộ tham chiếu tuyệt đối giống nhau.
    """
    model.eval()
    model.to(device)

    predictions, references = [], []

    for start in range(0, len(test_ds), batch_size):
        batch = test_ds[start : start + batch_size]

        input_features = torch.tensor(batch["input_features"]).to(device)
        if input_features.dtype != torch.float32:
            input_features = input_features.float()

        with torch.no_grad():
            generated_ids = model.generate(
                input_features,
                max_new_tokens=225,
                language=config.LANGUAGE,
                task=config.TASK,
            )

        pred_str = processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        # decode label ids thành text tham chiếu (labels không chứa -100 ở
        # đây vì đây là dữ liệu thô từ bước 02, chưa qua DataCollator)
        label_str = processor.tokenizer.batch_decode(batch["labels"], skip_special_tokens=True)

        predictions.extend(normalize_text(s) for s in pred_str)
        references.extend(normalize_text(s) for s in label_str)

        done = min(start + batch_size, len(test_ds))
        print(f"  ... đã transcribe {done}/{len(test_ds)} mẫu")

    return predictions, references


def evaluate_model(model_name_or_path: str, test_ds, device):
    print(f"\nLoad model: {model_name_or_path}")
    processor = WhisperProcessor.from_pretrained(
        model_name_or_path, language=config.LANGUAGE, task=config.TASK
    )
    model = WhisperForConditionalGeneration.from_pretrained(model_name_or_path)

    predictions, references = transcribe_dataset(model, processor, test_ds, device)
    wer = 100 * wer_metric.compute(predictions=predictions, references=references)

    return {
        "wer": wer,
        "predictions": predictions,
        "references": references,
    }


def print_comparison_table(results: dict):
    print("\n" + "=" * 70)
    print("BẢNG SO SÁNH WER TRÊN TẬP TEST (miền Trung)")
    print("=" * 70)
    print(f"{'Model':<45}{'WER (%)':>15}")
    print("-" * 70)
    for name, r in results.items():
        print(f"{name:<45}{r['wer']:>14.2f}%")
    print("=" * 70)

    base_wer = results["Whisper gốc (chưa fine-tune)"]["wer"]
    ft_wer = results["Whisper đã fine-tune (miền Trung)"]["wer"]
    if base_wer > 0:
        improvement = 100 * (base_wer - ft_wer) / base_wer
        print(f"\n>>> Cải thiện tương đối: {improvement:.1f}% (WER giảm từ {base_wer:.2f}% xuống {ft_wer:.2f}%)")


def print_qualitative_examples(results: dict, n=5):
    print(f"\nVí dụ minh họa ({n} câu đầu tiên trong tập test):")
    refs = results["Whisper đã fine-tune (miền Trung)"]["references"]
    base_preds = results["Whisper gốc (chưa fine-tune)"]["predictions"]
    ft_preds = results["Whisper đã fine-tune (miền Trung)"]["predictions"]

    for i in range(min(n, len(refs))):
        print(f"\n[{i+1}] Đúng      : {refs[i]}")
        print(f"    Model gốc : {base_preds[i]}")
        print(f"    Fine-tune : {ft_preds[i]}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Thiết bị dùng để đánh giá: {device}")
    if device == "cpu":
        print("[Lưu ý] Đánh giá trên CPU sẽ chậm hơn đáng kể so với GPU.")

    print(f"Load tập test đã tiền xử lý từ: {config.PROCESSED_DIR}")
    ds = load_from_disk(config.PROCESSED_DIR)
    test_ds = ds["test"]
    print(f"  {len(test_ds)} mẫu test.")

    if not os.path.isdir(config.FINETUNED_MODEL_DIR):
        raise FileNotFoundError(
            f"Không tìm thấy model đã fine-tune tại {config.FINETUNED_MODEL_DIR}. "
            f"Hãy chạy 03_finetune_whisper.py trước."
        )

    results = {}
    results["Whisper gốc (chưa fine-tune)"] = evaluate_model(config.MODEL_NAME, test_ds, device)
    results["Whisper đã fine-tune (miền Trung)"] = evaluate_model(
        config.FINETUNED_MODEL_DIR, test_ds, device
    )

    print_comparison_table(results)
    print_qualitative_examples(results)

    # Lưu kết quả ra file để dùng lại (vd. điền vào README) mà không cần
    # chạy lại toàn bộ script tốn thời gian
    summary = {
        name: {"wer": r["wer"]} for name, r in results.items()
    }
    with open(config.EVAL_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nĐã lưu kết quả tóm tắt vào: {config.EVAL_RESULTS_PATH}")


if __name__ == "__main__":
    main()
