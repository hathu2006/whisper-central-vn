"""
01_load_and_filter_data.py
===========================
BƯỚC 1: Tải dataset ViMD từ Hugging Face, lọc riêng các mẫu region="Central",
và in thống kê cơ bản.

TẠI SAO DÙNG STREAMING THAY VÌ load_dataset() THÔNG THƯỜNG?
-------------------------------------------------------------
Toàn bộ ViMD (cả North/Central/South) nặng khoảng 59GB audio. Nếu tải bình
thường, Hugging Face sẽ cache nguyên 59GB đó xuống đĩa trước, RỒI mới lọc —
trong khi ta chỉ cần giữ lại ~1/3 (phần Central). Trên Colab free (đĩa có hạn,
phiên làm việc có thể bị ngắt bất cứ lúc nào), việc này rất rủi ro.

Giải pháp: dùng `streaming=True`. Dataset sẽ được đọc theo kiểu "chảy qua"
(giống đọc file lớn từng dòng), lọc ngay trong lúc đọc, và ta chỉ ghi xuống
đĩa những mẫu đạt điều kiện region="Central". Nhược điểm: vẫn phải tải qua
mạng gần như toàn bộ dữ liệu (vì việc lọc diễn ra ở phía client, không phải
server), nên bước này vẫn tốn thời gian mạng tương đương bản đầy đủ — nhưng
đĩa thì tiết kiệm được rất nhiều.

KHUYẾN NGHỊ KHI CHẠY LẦN ĐẦU TRÊN COLAB:
Hãy chạy thử với --max_samples_per_split 200 trước để chắc chắn pipeline
chạy đúng (đường dẫn, quyền ghi, thư viện...) trước khi chạy full (có thể
mất hàng chục phút tới vài giờ tùy tốc độ mạng và dataset bị ngắt kết nối).
"""

from __future__ import annotations  # cho phép dùng cú pháp type hint mới (list[dict], X | None)
                                     # trên cả Python 3.9 (Colab thường có 3.10+ nhưng để an toàn)

import argparse
import json
import os
import sys
import time

import soundfile as sf
from datasets import load_dataset

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config


def parse_args():
    parser = argparse.ArgumentParser(description="Lọc dữ liệu ViMD theo vùng miền Trung")
    parser.add_argument(
        "--max_samples_per_split",
        type=int,
        default=None,
        help="Giới hạn số mẫu Central lấy ra mỗi split (dùng để test nhanh). "
             "Mặc định None = lấy hết.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=config.SPLITS,
        help=f"Danh sách split cần xử lý. Mặc định: {config.SPLITS}",
    )
    return parser.parse_args()


def filter_one_split(split_name: str, max_samples: int | None) -> list[dict]:
    """
    Duyệt qua 1 split ở chế độ streaming, giữ lại các mẫu region == Central,
    ghi file audio (.wav, giữ nguyên sample rate gốc) xuống đĩa, trả về danh
    sách metadata (dict) tương ứng.
    """
    print(f"\n[Split: {split_name}] Bắt đầu tải & lọc (streaming)...")
    t0 = time.time()

    ds_stream = load_dataset(config.HF_DATASET_ID, split=split_name, streaming=True)

    audio_out_dir = os.path.join(config.RAW_CENTRAL_DIR, "audio", split_name)
    os.makedirs(audio_out_dir, exist_ok=True)

    records = []
    n_seen = 0
    n_kept = 0
    n_skipped_duration = 0
    n_skipped_empty_text = 0

    for example in ds_stream:
        n_seen += 1

        # --- điều kiện lọc 1: đúng vùng miền Trung ---
        if example["region"] != config.REGION_FILTER:
            continue

        text = (example["text"] or "").strip()
        audio = example["audio"]  # dict: {"array": np.ndarray, "sampling_rate": int}
        duration_sec = len(audio["array"]) / audio["sampling_rate"]

        # --- điều kiện lọc 2: loại transcript rỗng ---
        if len(text) == 0:
            n_skipped_empty_text += 1
            continue

        # --- điều kiện lọc 3: loại audio quá ngắn/quá dài (xem giải thích
        #     trong config.py — Whisper chỉ nhận cửa sổ cố định 30s) ---
        if not (config.MIN_AUDIO_DURATION_SEC <= duration_sec <= config.MAX_AUDIO_DURATION_SEC):
            n_skipped_duration += 1
            continue

        # Ghi audio ra file wav (giữ nguyên sample rate gốc; việc resample về
        # 16kHz sẽ làm ở bước 02 để tách bạch rõ ràng "lọc dữ liệu" và
        # "tiền xử lý cho model")
        filename = example["filename"]
        out_path = os.path.join(audio_out_dir, filename)
        sf.write(out_path, audio["array"], audio["sampling_rate"])

        records.append(
            {
                "audio_path": out_path,
                "text": text,
                "duration_sec": round(duration_sec, 3),
                "province_code": example["province_code"],
                "province_name": example["province_name"],
                "speakerID": example["speakerID"],
                "gender": example["gender"],
            }
        )
        n_kept += 1

        if n_kept % 100 == 0:
            print(f"  ... đã giữ {n_kept} mẫu Central (đã duyệt qua {n_seen} mẫu tổng)")

        if max_samples is not None and n_kept >= max_samples:
            print(f"  Đã đạt max_samples_per_split={max_samples}, dừng sớm split này.")
            break

    elapsed = time.time() - t0
    print(
        f"[Split: {split_name}] Xong sau {elapsed/60:.1f} phút. "
        f"Duyệt {n_seen} mẫu -> giữ {n_kept} mẫu Central "
        f"(bỏ {n_skipped_empty_text} do rỗng text, "
        f"{n_skipped_duration} do thời lượng nằm ngoài "
        f"[{config.MIN_AUDIO_DURATION_SEC}, {config.MAX_AUDIO_DURATION_SEC}]s)."
    )
    return records


def print_summary(all_records: dict[str, list[dict]]):
    print("\n" + "=" * 70)
    print("THỐNG KÊ TẬP DỮ LIỆU MIỀN TRUNG (ViMD - Central)")
    print("=" * 70)

    total_samples = 0
    total_hours = 0.0
    all_provinces = set()

    for split_name, records in all_records.items():
        n = len(records)
        hours = sum(r["duration_sec"] for r in records) / 3600
        provinces = sorted({r["province_name"] for r in records})
        all_provinces.update(provinces)

        total_samples += n
        total_hours += hours

        print(f"\n- Split '{split_name}': {n} mẫu, {hours:.2f} giờ")
        print(f"  Số tỉnh xuất hiện: {len(provinces)} -> {provinces}")

    print(f"\n>>> TỔNG: {total_samples} mẫu, {total_hours:.2f} giờ audio")
    print(f">>> Tổng số tỉnh miền Trung có trong tập lọc: {len(all_provinces)}")
    print(f">>> Danh sách: {sorted(all_provinces)}")
    print("=" * 70)


def main():
    args = parse_args()

    print(f"Nguồn dataset: {config.HF_DATASET_ID}")
    print(f"Lọc theo region = '{config.REGION_FILTER}'")
    print(f"Ghi dữ liệu đã lọc vào: {config.RAW_CENTRAL_DIR}")
    if args.max_samples_per_split:
        print(
            f"[CHẾ ĐỘ TEST] Giới hạn {args.max_samples_per_split} mẫu/split "
            f"— bỏ cờ này khi chạy full."
        )

    all_records = {}
    for split_name in args.splits:
        all_records[split_name] = filter_one_split(split_name, args.max_samples_per_split)

        # Lưu metadata ra file .jsonl (mỗi dòng 1 mẫu) để bước 02 đọc lại
        meta_path = os.path.join(config.RAW_CENTRAL_DIR, f"{split_name}.jsonl")
        with open(meta_path, "w", encoding="utf-8") as f:
            for r in all_records[split_name]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  Đã lưu metadata: {meta_path}")

    print_summary(all_records)

    # Cảnh báo nếu 1 split nào đó rỗng — có thể do dataset đổi format, lỗi
    # mạng, hoặc tên split/region không còn đúng như tài liệu mô tả.
    for split_name, records in all_records.items():
        if len(records) == 0:
            print(
                f"\n[CẢNH BÁO] Split '{split_name}' không có mẫu Central nào. "
                f"Hãy kiểm tra lại: (1) tên split có đúng không, "
                f"(2) giá trị region trong dataset có đúng là 'Central' không "
                f"(in thử example['region'] để xác nhận), "
                f"(3) kết nối mạng/log lỗi phía trên."
            )


if __name__ == "__main__":
    main()
