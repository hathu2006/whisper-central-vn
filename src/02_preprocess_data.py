"""
02_preprocess_data.py
======================
BƯỚC 2: Biến dữ liệu đã lọc (audio + text thô) thành định dạng Whisper cần:
    - input_features: log-mel spectrogram 80 kênh, trích từ audio 16kHz
    - labels: chuỗi token id từ transcript (qua WhisperTokenizer)

Đầu vào : data/vimd_central_raw/{split}.jsonl + các file .wav (từ bước 01)
Đầu ra  : data/vimd_central_processed/ (định dạng HuggingFace `datasets`,
          load lại bằng datasets.load_from_disk())

VỀ VIỆC CHIA TRAIN/TEST:
------------------------
Ta KHÔNG tự random-split lại từ đầu. ViMD đã có sẵn 3 split train/valid/test
được tác giả chia theo dataset gốc. Giữ nguyên split gốc quan trọng vì:
  1) Tránh rò rỉ dữ liệu (data leakage): nếu tự trộn rồi chia lại ngẫu nhiên,
     rất dễ để giọng của cùng 1 speaker xuất hiện cả ở train và test, khiến
     model "học thuộc" giọng đó và WER trên test bị đánh giá ảo cao hơn thực tế.
  2) Kết quả có thể so sánh với các nghiên cứu khác cũng dùng ViMD.
Trong dự án này: train = học, valid = theo dõi trong lúc train (early
stopping / chọn checkpoint tốt nhất), test = CHỈ dùng 1 lần cuối cùng ở
bước 04 để đánh giá khách quan.
"""

from __future__ import annotations

import json
import os
import sys

from datasets import Audio, Dataset, DatasetDict

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config

from transformers import WhisperFeatureExtractor, WhisperTokenizer


def load_raw_split(split_name: str) -> Dataset:
    """Đọc file {split}.jsonl (metadata) sinh ra từ bước 01 thành Dataset."""
    meta_path = os.path.join(config.RAW_CENTRAL_DIR, f"{split_name}.jsonl")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"Không tìm thấy {meta_path}. Hãy chạy 01_load_and_filter_data.py trước."
        )

    records = []
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    if len(records) == 0:
        raise ValueError(
            f"Split '{split_name}' rỗng (0 mẫu). Không thể tiếp tục tiền xử lý."
        )

    ds = Dataset.from_list(records)

    # Cast cột audio_path -> Audio feature. Đây là bước RESAMPLE VỀ 16kHZ:
    # `datasets` sẽ tự động resample mỗi khi audio được truy cập (lazy),
    # dùng backend librosa/soundfile, không cần code resample tay.
    ds = ds.rename_column("audio_path", "audio")
    ds = ds.cast_column("audio", Audio(sampling_rate=config.TARGET_SAMPLING_RATE))
    return ds


def build_prepare_fn(feature_extractor: WhisperFeatureExtractor, tokenizer: WhisperTokenizer):
    """
    Trả về hàm `prepare_example` dùng cho Dataset.map().
    Tách thành hàm factory để feature_extractor/tokenizer được "đóng gói"
    (closure) mà không phải truyền lại qua map() một cách phức tạp.
    """

    def prepare_example(batch):
        audio = batch["audio"]  # đã được resample 16kHz nhờ cast_column ở trên

        # log-mel spectrogram 80 kênh — input thực sự Whisper encoder nhận vào
        batch["input_features"] = feature_extractor(
            audio["array"], sampling_rate=audio["sampling_rate"]
        ).input_features[0]

        # tokenize transcript thành label ids (Whisper decoder học sinh ra
        # đúng chuỗi token này)
        batch["labels"] = tokenizer(batch["text"]).input_ids

        # ghi lại độ dài audio tính bằng số sample sau resample, hữu ích để
        # lọc / debug sau này nếu cần
        batch["input_length"] = len(audio["array"]) / audio["sampling_rate"]
        return batch

    return prepare_example


def main():
    print(f"Model dùng để lấy feature extractor & tokenizer: {config.MODEL_NAME}")
    feature_extractor = WhisperFeatureExtractor.from_pretrained(config.MODEL_NAME)
    tokenizer = WhisperTokenizer.from_pretrained(
        config.MODEL_NAME, language=config.LANGUAGE, task=config.TASK
    )
    prepare_example = build_prepare_fn(feature_extractor, tokenizer)

    processed = DatasetDict()
    for split_name in config.SPLITS:
        print(f"\n[Split: {split_name}] Đang load & cast audio -> 16kHz...")
        ds = load_raw_split(split_name)
        print(f"  {len(ds)} mẫu. Đang trích input_features + labels (Dataset.map)...")

        # num_proc>1 chạy song song trên nhiều tiến trình -> nhanh hơn khi
        # dataset lớn. Tuy nhiên trên Colab free (RAM giới hạn), mỗi tiến
        # trình con phải decode audio riêng, rất dễ bị OOM-killer giết ngầm
        # (lỗi "subprocess abruptly died", không có traceback rõ ràng). Vì
        # dataset ở đây chỉ vài nghìn mẫu, num_proc=1 (không multiprocessing)
        # đã đủ nhanh và an toàn hơn. Nếu bạn có runtime nhiều RAM (vd. Colab
        # Pro/High-RAM) và muốn tăng tốc, có thể thử num_proc=2.
        # Giữ lại cột province_name (bỏ hết cột thô còn lại: audio nặng nhất,
        # text/speakerID/gender/province_code/duration_sec không cần cho các
        # bước sau) — để bước 04 có thể phân tích WER theo TỪNG TỈNH thay vì
        # chỉ 1 con số trung bình toàn tập, insight sâu hơn nhiều.
        columns_to_drop = [c for c in ds.column_names if c != "province_name"]
        ds = ds.map(
            prepare_example,
            remove_columns=columns_to_drop,  # bỏ cột thô (audio, text,...),
                                              # giữ lại province_name +
                                              # các cột ta add thêm trong hàm
            num_proc=1,
            # writer_batch_size: số dòng gom lại trong RAM trước khi ghi 1
            # lần xuống đĩa (mặc định 1000). input_features là spectrogram
            # 80x3000 float32 (~960KB/mẫu) -> gom 1000 mẫu cùng lúc có thể
            # cần thêm nhiều GB RAM ngay lúc "chốt" (finalize) file, dễ vượt
            # trần RAM 12GB của Colab free và bị OOM-killer giết ngầm (không
            # có traceback). Hạ xuống 50 để ghi xuống đĩa thường xuyên hơn,
            # giữ RAM đỉnh điểm thấp hơn nhiều — đánh đổi bằng việc ghi đĩa
            # (I/O) nhiều lần hơn 1 chút, không đáng kể so với lợi ích ổn định.
            writer_batch_size=50,
            desc=f"Trích features [{split_name}]",
        )
        processed[split_name] = ds
        print(f"  Xong split '{split_name}'.")

    print(f"\nLưu dataset đã xử lý vào: {config.PROCESSED_DIR}")
    processed.save_to_disk(config.PROCESSED_DIR)

    print("\nTóm tắt số mẫu sau tiền xử lý:")
    for split_name in config.SPLITS:
        print(f"  - {split_name}: {len(processed[split_name])} mẫu")

    print(
        "\nHoàn tất bước 2. Ở bước 03, load lại dataset này bằng:\n"
        "    from datasets import load_from_disk\n"
        f"    ds = load_from_disk('{config.PROCESSED_DIR}')"
    )


if __name__ == "__main__":
    main()
