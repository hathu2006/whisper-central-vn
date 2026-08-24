"""
04_evaluate_wer.py
====================
BƯỚC 4: Đo chất lượng nhận diện trên tập TEST cho:
    (a) model Whisper GỐC (openai/whisper-small, chưa fine-tune)
    (b) model đã FINE-TUNE (output của bước 03)
rồi in các bảng so sánh: WER/CER tổng, WER theo từng tỉnh, và top các lỗi
nhận diện hay gặp nhất (confusion analysis).

WER là gì (nhắc lại ngắn gọn): tỉ lệ lỗi ở mức TỪ, tính bằng
    WER = (Substitutions + Deletions + Insertions) / (số từ trong câu đúng)
CER tương tự nhưng tính ở mức KÝ TỰ thay vì từ — với tiếng Việt (nhiều từ
đơn âm tiết, từ láy/ghép), CER là chỉ số bổ sung hữu ích vì 1 lỗi nhỏ ở mức
ký tự (vd. sai dấu thanh) có thể làm cả từ bị tính sai trong WER, trong khi
CER phản ánh mức độ sai đó nhẹ hơn WER thể hiện.
Cả 2 chỉ số càng THẤP càng tốt. 0% nghĩa là nhận diện hoàn hảo.

Chạy: python src/04_evaluate_wer.py
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict

import evaluate
import jiwer
import torch
from datasets import load_from_disk
from transformers import WhisperForConditionalGeneration, WhisperProcessor

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from data_utils import normalize_text

wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")

# Số mẫu test tối thiểu 1 tỉnh cần có để báo cáo WER riêng cho tỉnh đó.
# Tỉnh có ít mẫu hơn ngưỡng này sẽ bị gộp chung vào 1 dòng "Khác" — vì WER
# tính trên quá ít mẫu (vd. 3-5 câu) dao động rất mạnh, dễ gây hiểu lầm nếu
# đọc riêng lẻ (1 câu sai/3 câu đã kéo WER lên tới 33%).
MIN_SAMPLES_PER_PROVINCE = 15


def transcribe_dataset(model, processor, test_ds, device, batch_size=8):
    """
    Chạy model.generate() theo từng batch trên tập test, trả về
    (predictions, references, provinces) — mỗi list cùng độ dài, cùng thứ tự
    với test_ds, đã chuẩn hóa text (normalize_text).

    Lưu ý: test_ds ở đây là dataset ĐÃ tiền xử lý (có sẵn input_features và
    labels dạng token id từ bước 02), nên ta decode labels lại thành text để
    so sánh, thay vì đọc lại text thô — cách này đảm bảo model gốc và model
    fine-tune được đánh giá trên cùng 1 bộ tham chiếu tuyệt đối giống nhau.
    """
    model.eval()
    model.to(device)

    predictions, references, provinces = [], [], []
    has_province = "province_name" in test_ds.column_names

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
        if has_province:
            provinces.extend(batch["province_name"])

        done = min(start + batch_size, len(test_ds))
        print(f"  ... đã transcribe {done}/{len(test_ds)} mẫu")

    if not has_province:
        print(
            "  [Lưu ý] Dataset không có cột 'province_name' (có thể bạn đang "
            "dùng dữ liệu tiền xử lý từ bản code cũ) -> bỏ qua breakdown theo "
            "tỉnh. Chạy lại 02_preprocess_data.py với bản code mới để có cột "
            "này."
        )
        provinces = [None] * len(predictions)

    return predictions, references, provinces


def evaluate_model(model_name_or_path: str, test_ds, device):
    print(f"\nLoad model: {model_name_or_path}")
    processor = WhisperProcessor.from_pretrained(
        model_name_or_path, language=config.LANGUAGE, task=config.TASK
    )
    model = WhisperForConditionalGeneration.from_pretrained(model_name_or_path)

    predictions, references, provinces = transcribe_dataset(model, processor, test_ds, device)
    wer = 100 * wer_metric.compute(predictions=predictions, references=references)
    cer = 100 * cer_metric.compute(predictions=predictions, references=references)

    return {
        "wer": wer,
        "cer": cer,
        "predictions": predictions,
        "references": references,
        "provinces": provinces,
    }


def print_comparison_table(results: dict):
    print("\n" + "=" * 70)
    print("BẢNG SO SÁNH WER/CER TRÊN TẬP TEST (miền Trung)")
    print("=" * 70)
    print(f"{'Model':<45}{'WER (%)':>12}{'CER (%)':>12}")
    print("-" * 70)
    for name, r in results.items():
        print(f"{name:<45}{r['wer']:>11.2f}%{r['cer']:>11.2f}%")
    print("=" * 70)

    base_wer = results["Whisper gốc (chưa fine-tune)"]["wer"]
    ft_wer = results["Whisper đã fine-tune (miền Trung)"]["wer"]
    if base_wer > 0:
        improvement = 100 * (base_wer - ft_wer) / base_wer
        print(f"\n>>> Cải thiện tương đối (WER): {improvement:.1f}% (giảm từ {base_wer:.2f}% xuống {ft_wer:.2f}%)")


def per_province_wer(predictions: list[str], references: list[str], provinces: list[str]) -> list[tuple]:
    """
    Gộp nhóm theo tỉnh, tính WER riêng cho từng tỉnh có đủ
    MIN_SAMPLES_PER_PROVINCE mẫu trở lên; các tỉnh còn lại gộp vào 1 nhóm
    "Khác" để tránh báo cáo WER dựa trên quá ít mẫu (không đáng tin cậy).
    Trả về list (tên_tỉnh, số_mẫu, wer), sắp xếp theo số mẫu giảm dần.
    """
    groups = defaultdict(lambda: {"preds": [], "refs": []})
    for pred, ref, prov in zip(predictions, references, provinces):
        key = prov if prov is not None else "(không rõ tỉnh)"
        groups[key]["preds"].append(pred)
        groups[key]["refs"].append(ref)

    rows = []
    small_preds, small_refs, small_n, small_provinces = [], [], 0, []
    for prov, data in groups.items():
        n = len(data["preds"])
        if n >= MIN_SAMPLES_PER_PROVINCE:
            wer = 100 * wer_metric.compute(predictions=data["preds"], references=data["refs"])
            rows.append((prov, n, wer))
        else:
            small_preds.extend(data["preds"])
            small_refs.extend(data["refs"])
            small_n += n
            small_provinces.append(prov)

    if small_n > 0:
        wer = 100 * wer_metric.compute(predictions=small_preds, references=small_refs)
        label = f"Khác ({len(small_provinces)} tỉnh, <{MIN_SAMPLES_PER_PROVINCE} mẫu/tỉnh, gộp lại)"
        rows.append((label, small_n, wer))

    rows.sort(key=lambda r: -r[1])
    return rows


def print_province_breakdown(rows: list[tuple]):
    print("\n" + "=" * 70)
    print(f"WER THEO TỈNH (model đã fine-tune) — tỉnh có >= {MIN_SAMPLES_PER_PROVINCE} mẫu test")
    print("=" * 70)
    print(f"{'Tỉnh':<45}{'Số mẫu':>10}{'WER (%)':>12}")
    print("-" * 70)
    for prov, n, wer in rows:
        print(f"{prov:<45}{n:>10}{wer:>11.2f}%")
    print("=" * 70)


def top_confusions(predictions: list[str], references: list[str], top_n: int = 15) -> list[tuple]:
    """
    Phân tích lỗi định lượng: dùng jiwer để căn chỉnh (align) từng câu dự
    đoán với câu đúng ở mức từ, tìm các cặp (từ_đúng -> từ_bị_nhận_nhầm)
    xuất hiện nhiều lần nhất trên toàn tập test. Giúp biến quan sát kiểu
    "hình như hay nhầm từ X" thành số liệu có dẫn chứng cụ thể, thay vì chỉ
    dựa vào vài ví dụ đơn lẻ.
    """
    output = jiwer.process_words(references, predictions)
    counter = Counter()

    for sent_idx, alignment_chunks in enumerate(output.alignments):
        ref_words = output.references[sent_idx]
        hyp_words = output.hypotheses[sent_idx]
        for chunk in alignment_chunks:
            if chunk.type != "substitute":
                continue
            # substitute chunk: đoạn từ_đúng[ref_start:ref_end] bị nhận nhầm
            # thành từ_dự_đoán[hyp_start:hyp_end] — 2 đoạn luôn cùng độ dài
            for ref_i, hyp_i in zip(
                range(chunk.ref_start_idx, chunk.ref_end_idx),
                range(chunk.hyp_start_idx, chunk.hyp_end_idx),
            ):
                counter[(ref_words[ref_i], hyp_words[hyp_i])] += 1

    return counter.most_common(top_n)


def print_top_confusions(confusions: list[tuple]):
    print("\n" + "=" * 70)
    print("TOP LỖI NHẬN DIỆN HAY GẶP NHẤT (model đã fine-tune, tập test)")
    print("=" * 70)
    print(f"{'Từ đúng':<20}{'-> Bị nhận nhầm thành':<25}{'Số lần':>10}")
    print("-" * 70)
    for (ref_word, hyp_word), count in confusions:
        print(f"{ref_word:<20}{'-> ' + hyp_word:<25}{count:>10}")
    print("=" * 70)


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

    ft_result = results["Whisper đã fine-tune (miền Trung)"]
    province_rows = []
    if any(p is not None for p in ft_result["provinces"]):
        province_rows = per_province_wer(
            ft_result["predictions"], ft_result["references"], ft_result["provinces"]
        )
        print_province_breakdown(province_rows)

    confusions = top_confusions(ft_result["predictions"], ft_result["references"])
    print_top_confusions(confusions)

    print_qualitative_examples(results)

    # Lưu kết quả ra file để dùng lại (vd. điền vào README) mà không cần
    # chạy lại toàn bộ script tốn thời gian
    summary = {
        "overall": {name: {"wer": r["wer"], "cer": r["cer"]} for name, r in results.items()},
        "per_province_wer_finetuned": [
            {"province": prov, "n_samples": n, "wer": wer} for prov, n, wer in province_rows
        ],
        "top_confusions_finetuned": [
            {"correct_word": ref, "confused_as": hyp, "count": count}
            for (ref, hyp), count in confusions
        ],
    }
    with open(config.EVAL_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nĐã lưu kết quả tóm tắt vào: {config.EVAL_RESULTS_PATH}")


if __name__ == "__main__":
    main()
