# Fine-tune Whisper cho giọng nói miền Trung Việt Nam

Dự án cá nhân: fine-tune model Whisper (small) để cải thiện độ chính xác nhận
diện giọng nói (ASR) cho phương ngữ miền Trung Việt Nam, đo lường cải thiện
bằng WER (Word Error Rate), và đóng gói thành demo Gradio.

## 1. Vấn đề

Các model ASR đa ngôn ngữ (kể cả Whisper) thường được huấn luyện trên dữ liệu
tiếng Việt chủ yếu là giọng miền Bắc/miền Nam (phổ biến trong dữ liệu web,
tin tức, phụ đề). Giọng miền Trung — với hệ thống nguyên âm, thanh điệu và
từ vựng địa phương khác biệt — thường bị nhận diện sai nhiều hơn. Dự án này
kiểm chứng giả thuyết: **fine-tune Whisper trên một lượng nhỏ dữ liệu giọng
miền Trung có gán nhãn sẽ cải thiện WER đáng kể so với model gốc**, mà không
cần train lại từ đầu.

## 2. Cách làm

### 2.1. Dữ liệu

Dùng [ViMD Dataset](https://huggingface.co/datasets/nguyendv02/ViMD_Dataset)
— 102.56 giờ audio tiếng Việt có gán nhãn vùng miền (North/Central/South),
tỉnh, giới tính người nói. Dự án chỉ dùng phần `region == "Central"`.

Đã xác minh trực tiếp qua Hugging Face Datasets Server API (không dùng theo
mô tả suông) các thông tin sau trước khi viết code:

| Thuộc tính | Giá trị thực tế |
|---|---|
| Cột dữ liệu | `audio`, `text`, `filename`, `speakerID`, `region`, `province_code`, `province_name`, `gender` |
| Giá trị `region` | chuỗi: `"North"`, `"Central"`, `"South"` |
| Splits | `train` (~15.2k mẫu / ~81.4h), `valid` (~1.9k mẫu / ~10.3h), `test` (~2.0k mẫu / ~10.9h) |
| Dung lượng audio toàn bộ dataset | ~59GB (cả 3 vùng miền) |
| License | **CC-BY-NC-ND-4.0** |

> ⚠️ **Lưu ý bản quyền:** ViMD dùng license CC-BY-NC-ND-4.0 (Non-Commercial,
> No-Derivatives). Dự án này chỉ dùng cho mục đích **cá nhân / học tập /
> portfolio**. Nếu public repo này lên GitHub để đính vào CV: **đừng commit
> dữ liệu audio hoặc publish weight model đã fine-tune** (vì license không
> cho phép phân phối bản phái sinh) — chỉ nên chia sẻ code + phương pháp +
> số liệu WER. Luôn trích dẫn paper gốc khi nhắc tới dataset (xem mục Tài
> liệu tham khảo bên dưới).

Vì audio Central chỉ là một phần của 59GB đó và Colab free có đĩa hạn chế,
bước tải dữ liệu dùng chế độ `streaming=True` của thư viện `datasets` để lọc
ngay trong lúc đọc, chỉ ghi xuống đĩa phần Central — xem giải thích chi tiết
trong [src/01_load_and_filter_data.py](src/01_load_and_filter_data.py).

### 2.2. Pipeline

| Bước | File | Việc chính |
|---|---|---|
| 1 | [src/01_load_and_filter_data.py](src/01_load_and_filter_data.py) | Tải ViMD (streaming), lọc `region="Central"`, thống kê số giờ/số mẫu/danh sách tỉnh |
| 2 | [src/02_preprocess_data.py](src/02_preprocess_data.py) | Resample 16kHz, trích log-mel `input_features` + tokenize `labels` cho Whisper, giữ nguyên train/valid/test gốc |
| 3 | [src/03_finetune_whisper.py](src/03_finetune_whisper.py) | Fine-tune `openai/whisper-small` bằng `Seq2SeqTrainer` |
| 4 | [src/04_evaluate_wer.py](src/04_evaluate_wer.py) | Tính WER trên test set: model gốc vs. model fine-tune |
| 5 | [src/05_app_gradio.py](src/05_app_gradio.py) | Demo Gradio: upload/thu âm → transcript |

File dùng chung: [src/config.py](src/config.py) (hằng số, đường dẫn),
[src/data_utils.py](src/data_utils.py) (data collator, chuẩn hóa text cho WER).

### 2.3. Vì sao chọn Whisper-small?

- `whisper-base` (74M tham số): nhẹ, nhưng chất lượng gốc trên tiếng Việt
  khá yếu → điểm xuất phát thấp, khó chứng minh cải thiện thuyết phục.
- `whisper-large`/`large-v3` (1.5B tham số): chất lượng gốc tốt nhất, nhưng
  full fine-tune đòi hỏi VRAM lớn hơn nhiều so với GPU T4 16GB miễn phí của
  Colab → không khả thi trong scope dự án cá nhân này.
- `whisper-small` (244M tham số): điểm cân bằng — đủ nhỏ để full fine-tune
  trên T4 free (batch nhỏ + gradient accumulation + fp16 + gradient
  checkpointing), nhưng đủ mạnh để thấy khác biệt WER rõ ràng.

Giải thích chi tiết từng hyperparameter (learning rate, warmup, batch size,
gradient accumulation, fp16, gradient checkpointing...) nằm ngay trong
comment của [src/03_finetune_whisper.py](src/03_finetune_whisper.py) — mục
tiêu là đọc code cũng hiểu được lý do, không chỉ chạy được.

## 3. Kết quả

Đo trên tập **test** (609 mẫu, giữ nguyên split gốc của ViMD, model chưa
từng thấy trong lúc train), bằng `src/04_evaluate_wer.py`:

| Model | WER trên test set (miền Trung) |
|---|---|
| Whisper-small gốc (chưa fine-tune) | 41.28% |
| Whisper-small đã fine-tune (miền Trung) | **21.19%** |

**→ Giảm WER tương đối 48.7%** (41.28% → 21.19%) chỉ với ~30.5 giờ dữ liệu
fine-tune (4617 mẫu train), 2000 step (~6.9 epoch), fine-tune trên 1 GPU T4
free của Colab trong ~3h9p.

Quá trình train: WER trên tập valid giảm đều qua các lần đánh giá
(24.11% → 22.99% → 22.79% → 22.48%) mà không có dấu hiệu overfit ngược lại
— cho thấy 2000 step là điểm dừng hợp lý, không lãng phí compute.

> Ví dụ transcript minh họa (model gốc vs. fine-tune) được in ở cuối
> `04_evaluate_wer.py` khi chạy — nên chọn 2-3 câu tiêu biểu (đặc biệt câu
> có từ vựng/phát âm đặc trưng miền Trung) để đưa thêm vào README/CV cho
> trực quan.

## 4. Hướng dẫn chạy lại

### Trên Google Colab (khuyến nghị — có GPU free)

1. Push repo này lên GitHub.
2. Mở [notebooks/colab_quickstart.ipynb](notebooks/colab_quickstart.ipynb)
   trong Colab (hoặc Open in Colab từ GitHub).
3. Runtime > Change runtime type > GPU (T4).
4. Sửa biến `REPO_URL` trong cell đầu tiên, chạy tuần tự các cell.
5. **Khuyến nghị**: chạy bước 1 với `--max_samples_per_split 200` trước để
   kiểm tra toàn bộ pipeline (1→5) chạy đúng, trước khi chạy full dataset
   (tốn nhiều thời gian tải + train hơn, và phiên Colab free có thể bị ngắt
   giữa chừng).

### Chạy local (cần Python 3.10+, khuyến nghị có GPU NVIDIA)

```bash
pip install -r requirements.txt

python src/01_load_and_filter_data.py --max_samples_per_split 200  # test nhanh
python src/01_load_and_filter_data.py                              # full data

python src/02_preprocess_data.py
python src/03_finetune_whisper.py
python src/04_evaluate_wer.py
python src/05_app_gradio.py
```

## 5. Cấu trúc thư mục

```
.
├── src/
│   ├── config.py                  # hằng số & đường dẫn dùng chung
│   ├── data_utils.py              # DataCollator, chuẩn hóa text cho WER
│   ├── 01_load_and_filter_data.py
│   ├── 02_preprocess_data.py
│   ├── 03_finetune_whisper.py
│   ├── 04_evaluate_wer.py
│   └── 05_app_gradio.py
├── notebooks/
│   └── colab_quickstart.ipynb     # vỏ notebook chạy tuần tự 5 bước trên Colab
├── data/                          # output bước 1 & 2 (gitignored)
├── outputs/                       # checkpoint model + kết quả eval (gitignored)
├── requirements.txt
└── README.md
```

## 6. Hạn chế & hướng phát triển tiếp

- Dữ liệu miền Trung trong ViMD tuy có gán nhãn tỉnh nhưng không chắc phân
  bố đều giữa các tỉnh — nên kiểm tra thống kê in ra ở bước 1 để biết tỉnh
  nào chiếm đa số, tránh model chỉ "học tốt" 1-2 tỉnh cụ thể.
- Chưa thử kỹ thuật tiết kiệm tài nguyên hơn nữa như LoRA/PEFT — có thể là
  hướng mở rộng nếu muốn fine-tune model lớn hơn (base/large) trên free GPU.
- WER là thước đo ở mức từ; với tiếng Việt (đơn âm tiết, nhiều từ láy/ghép),
  nên cân nhắc thêm CER (Character Error Rate) để đánh giá toàn diện hơn.

## 7. Tài liệu tham khảo

- Dataset: Dinh et al., "Multi-Dialect Vietnamese: Task, Dataset, Baseline
  Models and Challenges", EMNLP 2024.
  https://aclanthology.org/2024.emnlp-main.426
- Model: [openai/whisper-small](https://huggingface.co/openai/whisper-small)
  — Radford et al., "Robust Speech Recognition via Large-Scale Weak
  Supervision", 2022.
