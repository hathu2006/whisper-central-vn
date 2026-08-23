"""
config.py — Nơi tập trung TẤT CẢ hằng số và đường dẫn dùng chung cho cả dự án.

Lý do tách riêng file này: các script 01 -> 05 đều cần những giá trị giống nhau
(tên dataset, tên model, sample rate, đường dẫn thư mục...). Nếu mỗi script tự khai
báo riêng, khi cần đổi (vd. đổi từ whisper-small sang whisper-base) sẽ phải sửa
nhiều nơi và rất dễ sót. Sửa 1 chỗ trong file này là đủ.
"""

import os

# ---------------------------------------------------------------------------
# 1) Thông tin dataset gốc (ViMD) trên Hugging Face
# ---------------------------------------------------------------------------
# Đã kiểm tra trực tiếp qua HuggingFace Datasets Server API (không đoán):
#   - Các cột thực tế: audio, text, filename, speakerID, region,
#     province_code, province_name, gender
#   - region nhận 1 trong 3 giá trị chuỗi: "North", "Central", "South"
#   - Splits: train (~15.2k mẫu / ~81.4h), valid (~1.9k mẫu / ~10.3h),
#     test (~2.0k mẫu / ~10.9h)
#   - Tổng dung lượng audio toàn bộ dataset (cả 3 vùng miền) ~59GB.
#     => Đây là lý do 01_load_and_filter_data.py dùng streaming=True
#        thay vì tải nguyên bộ về đĩa rồi mới lọc.
#   - License: CC-BY-NC-ND-4.0 (Non-Commercial, No-Derivatives).
#     => Chỉ dùng cho mục đích cá nhân/học tập/portfolio, KHÔNG public model
#        đã fine-tune lên nơi khác, KHÔNG dùng thương mại. Xem README.md
#        mục "Lưu ý bản quyền" trước khi public repo.
HF_DATASET_ID = "nguyendv02/ViMD_Dataset"
DATASET_CITATION = (
    "Dinh et al., \"Multi-Dialect Vietnamese: Task, Dataset, Baseline Models "
    "and Challenges\", EMNLP 2024. https://aclanthology.org/2024.emnlp-main.426"
)

REGION_FILTER = "Central"          # vùng miền muốn giữ lại
SPLITS = ["train", "valid", "test"]  # tên split đúng như trên Hugging Face

# Một số mẫu trong ViMD dài tới ~30.8s, trong khi Whisper chỉ xử lý cửa sổ
# 30s cố định (audio dài hơn sẽ bị cắt nhưng transcript vẫn giữ nguyên toàn bộ
# -> lệch nhãn). Vì vậy ta loại các mẫu vượt ngưỡng này ở bước lọc dữ liệu.
MAX_AUDIO_DURATION_SEC = 30.0
MIN_AUDIO_DURATION_SEC = 0.5   # loại audio quá ngắn/lỗi (gần như im lặng)

# ---------------------------------------------------------------------------
# 2) Model Whisper
# ---------------------------------------------------------------------------
# Vì sao chọn "whisper-small" (244M tham số) thay vì base (74M) hay large (1.5B):
#   - base: nhẹ, train nhanh, nhưng chất lượng nhận diện tiếng Việt gốc (chưa
#     fine-tune) khá yếu -> điểm xuất phát thấp, khó cải thiện thuyết phục.
#   - large/large-v3: chất lượng gốc tốt nhất, nhưng ~1.5 tỷ tham số đòi hỏi
#     VRAM lớn (>15GB chỉ để load, chưa tính gradient/optimizer states) và
#     train rất chậm -> KHÔNG khả thi để full fine-tune trên GPU T4 16GB
#     (Colab free tier) trong thời gian hợp lý.
#   - small: cân bằng tốt nhất cho bài toán này — đủ nhỏ để fine-tune full
#     (không cần LoRA/PEFT) trên T4 free với batch nhỏ + gradient
#     accumulation, nhưng vẫn đủ mạnh để thấy WER cải thiện rõ rệt sau khi
#     fine-tune trên vài giờ dữ liệu miền Trung.
MODEL_NAME = "openai/whisper-small"
LANGUAGE = "vietnamese"     # dùng cho WhisperTokenizer / generation_config
TASK = "transcribe"         # transcribe (giữ nguyên tiếng Việt), không dịch

TARGET_SAMPLING_RATE = 16000  # Whisper luôn yêu cầu input 16kHz mono

# ---------------------------------------------------------------------------
# 3) Đường dẫn thư mục (tất cả nằm trong data/ và outputs/, đã .gitignore)
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_CENTRAL_DIR = os.path.join(DATA_DIR, "vimd_central_raw")   # output bước 1
PROCESSED_DIR = os.path.join(DATA_DIR, "vimd_central_processed")  # output bước 2

OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
FINETUNED_MODEL_DIR = os.path.join(OUTPUTS_DIR, "whisper-small-central-vn")
EVAL_RESULTS_PATH = os.path.join(OUTPUTS_DIR, "wer_comparison.json")

for _dir in (DATA_DIR, RAW_CENTRAL_DIR, PROCESSED_DIR, OUTPUTS_DIR):
    os.makedirs(_dir, exist_ok=True)
