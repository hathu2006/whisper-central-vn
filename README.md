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
từng thấy trong lúc train), bằng `src/04_evaluate_wer.py` (text đã chuẩn hóa
lowercase + bỏ dấu câu trước khi so sánh — xem `src/data_utils.py`):

| Model | WER | CER |
|---|---|---|
| Whisper-small gốc (chưa fine-tune) | 38.10% | 23.35% |
| Whisper-small đã fine-tune (miền Trung) | **17.95%** | **11.48%** |

**→ Giảm WER tương đối 52.9%** (38.10% → 17.95%), CER giảm tương đối ~50.8%
(23.35% → 11.48%), chỉ với ~30.5 giờ dữ liệu fine-tune (4617 mẫu train),
2000 step (~6.9 epoch), fine-tune trên 1 GPU T4 free của Colab trong ~3h9p.

Quá trình train: WER trên tập valid giảm đều qua các lần đánh giá
(24.11% → 22.99% → 22.79% → 22.48%) mà không có dấu hiệu overfit ngược lại
— cho thấy 2000 step là điểm dừng hợp lý, không lãng phí compute.

### WER theo từng tỉnh (model đã fine-tune)

Tất cả 19 tỉnh trong tập test đều có ≥15 mẫu, đủ để báo cáo riêng (không
cần gộp nhóm "Khác" vì thiếu mẫu):

| Tỉnh | Số mẫu | WER |
|---|---|---|
| NgheAn | 43 | 13.61% |
| QuangNgai | 38 | 25.72% |
| QuangBinh | 36 | 22.47% |
| NinhThuan | 34 | 12.14% |
| HaTinh | 33 | 22.45% |
| DaNang | 33 | 16.46% |
| LamDong | 33 | 20.09% |
| BinhThuan | 33 | 20.38% |
| ThuaThienHue | 32 | 20.67% |
| GiaLai | 32 | 16.82% |
| DakLak | 31 | 15.57% |
| PhuYen | 31 | 20.59% |
| KonTum | 30 | 19.17% |
| QuangTri | 29 | 14.18% |
| KhanhHoa | 29 | 11.99% |
| QuangNam | 29 | 16.70% |
| ThanhHoa | 28 | 13.99% |
| DakNong | 28 | 14.79% |
| BinhDinh | 27 | 20.98% |

WER dao động khá rộng giữa các tỉnh (11.99% – 25.72%), không lệch hẳn về
riêng 1-2 tỉnh chiếm đa số dữ liệu train — cho thấy model không chỉ "học
thuộc" giọng của tỉnh có nhiều mẫu nhất (NgheAn, 43 mẫu test nhưng WER
13.61%, không phải thấp nhất). KhanhHoa/NinhThuan (Nam Trung Bộ) có WER thấp
nhất; QuangNgai/QuangBinh có WER cao nhất — có thể do đặc trưng phát âm
riêng, hoặc chỉ do nhiễu thống kê với cỡ mẫu ~30/tỉnh (không đủ lớn để kết
luận chắc chắn, cần thêm dữ liệu để kiểm chứng).

### Top lỗi nhận diện hay gặp nhất (phân tích bằng jiwer alignment)

| Từ đúng | Bị nhận nhầm thành | Số lần |
|---|---|---|
| cây | cái | 17 |
| thành | thanh | 10 |
| là | thì | 7 |
| trồng | trong | 7 |
| quang | quan | 7 |
| là | và | 7 |
| tình | tinh | 6 |
| nó | là | 6 |
| các | có | 6 |
| với | cái | 6 |
| mà | và | 6 |
| có | cái | 6 |
| con | còn | 6 |
| nấm | nắm | 6 |
| chưng | trưng | 6 |

Pattern rõ rệt nhất: **mất dấu thanh điệu / âm cuối** (thành→thanh,
trồng→trong, tình→tinh, nấm→nắm, chưng→trưng, quang→quan) — đúng loại lỗi
kinh điển khi ASR xử lý tiếng Việt, một ngôn ngữ đơn âm tiết có thanh điệu.
Nhóm lỗi thứ hai là nhầm các từ chức năng tần suất cao (là/thì/và, có/cái) —
dấu hiệu model thiên về từ phổ biến hơn khi không chắc chắn.

**Ví dụ thực tế** — test qua demo Gradio (bước 5) với 1 đoạn video giọng Huế
ngoài tập test (không nằm trong quá trình train/eval):

> **Model fine-tune:** "Thì từ đầu tiên mà mình muốn chia sẻ với các bạn đó
> là từ rửa. Không phải là chỉ đơn quần là kết là từ rưỡi không thôi mà phải
> gắn vào với từ án. À rữa, đó á rữa. Ví dụ như các bạn về huế, các bạn ăn
> mục tô bú, tô cơm hến, chẳng hạn. Đó khi mong người bưng ra, đó thì cái từ
> át rữa đó được dự hình như thế."
>
> **Thực tế:** "Thì từ đầu tiên mà mình muốn chia sẻ với các bạn đó là từ
> rứa. Không phải là chỉ đơn thuần là cái là từ rứa không thôi mà phải gắn
> vào với từ à. À rứa, đó à rứa. Ví dụ như các bạn về huế, các bạn ăn một tô
> bún, tô cơm hến, chẳng hạn. Đó khi mọi người bưng ra, đó thì cái từ à rứa
> đó được được hiểu như thế."

Nhận diện đúng phần lớn câu, kể cả các từ địa phương ít gặp ("huế", "cơm
hến"). Lỗi còn lại tập trung ở đúng nhóm từ khó nhất: chính từ cảm thán đặc
trưng miền Trung **"rứa"** bị nhầm thành "rửa/rưỡi/rữa" nhiều lần trong cùng
1 đoạn — cho thấy model vẫn chưa nắm chắc từ vựng đặc thù này dù đã cải
thiện đáng kể so với model gốc (xem mục Hạn chế bên dưới).

## 4. Hướng dẫn chạy lại

### Trên Google Colab (khuyến nghị — có GPU free)

1. Mở [notebooks/colab_quickstart.ipynb](notebooks/colab_quickstart.ipynb)
   trong Colab (hoặc Open in Colab từ GitHub) — đã có sẵn `REPO_URL` trỏ
   đúng repo này, không cần sửa gì thêm.
2. Runtime > Change runtime type > GPU (T4).
3. Chạy tuần tự các cell, **bao gồm cell mount Google Drive** ngay đầu
   notebook — quan trọng vì Colab free có thể ngắt phiên bất cứ lúc nào,
   trong khi model checkpoint (bước 3) cần lưu ở nơi bền vững để không mất
   công train lại từ đầu (script tự resume từ checkpoint gần nhất nếu tìm
   thấy trên Drive).
4. **Khuyến nghị**: chạy bước 1 với `--max_samples_per_split 50` trước để
   kiểm tra toàn bộ pipeline (1→5) chạy đúng (1-2 phút), trước khi chạy full
   dataset (tốn nhiều thời gian tải + train hơn).
5. Nếu phiên Colab bị ngắt/hết hạn giữa chừng: mount lại Drive, set lại
   biến `WHISPER_PROJECT_OUTPUTS_DIR`, `git clone` lại repo (vì `/content`
   bị xóa sạch khi phiên mới), rồi chạy lại đúng script đang dang dở —
   không cần làm lại từ bước 1 nếu model/data đã có trên Drive.

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

- Đã kiểm tra WER theo từng tỉnh (mục 3) — dao động 11.99%–25.72%, không
  lệch hẳn về 1-2 tỉnh chiếm đa số dữ liệu, nhưng cỡ mẫu mỗi tỉnh còn nhỏ
  (~30 mẫu test/tỉnh) nên chưa đủ để kết luận chắc chắn tỉnh nào thực sự khó
  hơn hay chỉ là nhiễu thống kê — cần thêm dữ liệu để kiểm chứng.
- Chưa thử kỹ thuật tiết kiệm tài nguyên hơn nữa như LoRA/PEFT — có thể là
  hướng mở rộng nếu muốn fine-tune model lớn hơn (base/large) trên free GPU.
- Model vẫn yếu nhất ở đúng nhóm từ cảm thán đặc trưng miền Trung (mô, ri,
  răng, rứa...) — hợp lý vì tần suất xuất hiện trong tập train không nhiều;
  đây là hướng cải thiện rõ ràng nếu có thêm dữ liệu.
- Dữ liệu train chỉ gồm các đoạn có giọng nói rõ ràng (đã lọc ≤30s ở bước
  1), không có đoạn nhạc nền/im lặng — nên khi demo gặp audio dài có xen
  đoạn không lời (nhạc intro...), model có thể hallucinate (sinh lặp từ vô
  nghĩa). Đã giảm thiểu bằng cách tắt `condition_on_prev_tokens` + bật các
  ngưỡng phát hiện no-speech/logprob/compression-ratio chuẩn của Whisper
  (xem `src/05_app_gradio.py`), nhưng không loại bỏ hoàn toàn 100%. Demo
  hoạt động tốt nhất với audio ngắn, giọng nói rõ ràng — đúng phạm vi dữ
  liệu đã train.

## 7. Tài liệu tham khảo

- Dataset: Dinh et al., "Multi-Dialect Vietnamese: Task, Dataset, Baseline
  Models and Challenges", EMNLP 2024.
  https://aclanthology.org/2024.emnlp-main.426
- Model: [openai/whisper-small](https://huggingface.co/openai/whisper-small)
  — Radford et al., "Robust Speech Recognition via Large-Scale Weak
  Supervision", 2022.
