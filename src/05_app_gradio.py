"""
05_app_gradio.py
==================
BƯỚC 5: Demo Gradio — người dùng upload file audio hoặc thu âm trực tiếp,
app trả về transcript bằng model đã fine-tune (output của bước 03).

Chạy: python src/05_app_gradio.py
Sau đó mở link (thường là http://127.0.0.1:7860) hiện ra trong terminal.
Trên Colab: gọi launch(share=True) để có link public tạm thời xem trên
điện thoại/máy khác.
"""

from __future__ import annotations

import os
import sys

import gradio as gr
import torch
from transformers import pipeline

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config

# ---------------------------------------------------------------------------
# Load model 1 LẦN DUY NHẤT khi app khởi động (không load lại mỗi lần người
# dùng bấm nút) — dùng transformers.pipeline("automatic-speech-recognition")
# vì nó tự lo luôn việc: đọc file audio (qua ffmpeg), resample về 16kHz,
# cắt audio dài thành từng đoạn 30s nếu cần — đỡ phải tự viết lại các bước
# tiền xử lý đã làm ở 02_preprocess_data.py cho input runtime.
# ---------------------------------------------------------------------------
device = 0 if torch.cuda.is_available() else -1  # pipeline dùng device=0 nghĩa là GPU đầu tiên

if not os.path.isdir(config.FINETUNED_MODEL_DIR):
    raise FileNotFoundError(
        f"Không tìm thấy model đã fine-tune tại {config.FINETUNED_MODEL_DIR}.\n"
        f"Hãy chạy 03_finetune_whisper.py trước, hoặc sửa MODEL_PATH bên dưới "
        f"thành đường dẫn model bạn muốn dùng (vd. model đã upload lên Hugging Face Hub)."
    )

print(f"Đang load model từ: {config.FINETUNED_MODEL_DIR} (device={'GPU' if device == 0 else 'CPU'})")
asr_pipeline = pipeline(
    task="automatic-speech-recognition",
    model=config.FINETUNED_MODEL_DIR,
    device=device,
    # Whisper chỉ xử lý cửa sổ cố định 30s/lần. Audio user upload có thể dài
    # hơn 30s (khác với dữ liệu train, vốn đã lọc <=30s ở bước 01).
    #
    # ĐÃ THỬ chunk_length_s=30 trước đó nhưng transformers tự cảnh báo cách
    # này "very experimental with seq2seq models" — pipeline cắt audio thành
    # từng đoạn 30s ĐỘC LẬP rồi ghép lại, dễ mất câu ở ranh giới đoạn.
    # Đổi sang return_timestamps=True: cơ chế "long-form generation" CHÍNH
    # THỨC xây sẵn trong WhisperForConditionalGeneration.generate() (đúng
    # giải pháp được gợi ý trong lỗi gốc "> 30 seconds..."), tự nối ngữ cảnh
    # qua nhiều cửa sổ 30s bằng token timestamp.
    return_timestamps=True,
    generate_kwargs={
        "language": config.LANGUAGE,
        "task": config.TASK,
        # Mặc định Whisper lấy CHÍNH text đã sinh ra ở cửa sổ 30s trước làm
        # ngữ cảnh cho cửa sổ sau (condition_on_prev_tokens=True). Nếu cửa
        # sổ đầu là nhạc nền/im lặng và model (đặc biệt model đã fine-tune,
        # chỉ thấy giọng nói rõ ràng lúc train, không quen với "không có
        # tiếng nói") hallucinate ra 1 từ lặp, ngữ cảnh sai đó bị đẩy tiếp
        # sang mọi cửa sổ sau -> lặp vô hạn xuyên suốt cả bài (đúng hiện
        # tượng "CÁI CÁI CÁI..." gặp phải). Tắt hẳn cơ chế này để mỗi cửa sổ
        # 30s được sinh độc lập, 1 đoạn hallucinate sẽ không lây lan.
        "condition_on_prev_tokens": False,
        # 3 ngưỡng dưới đây là cơ chế fallback gốc của Whisper (OpenAI) để
        # phát hiện đoạn không có tiếng nói / audio nhiễu và bỏ qua thay vì
        # cố sinh chữ: no_speech_threshold (xác suất "không có ai nói" đủ
        # cao thì coi là im lặng), logprob_threshold (model không đủ tự tin
        # về cả câu thì coi là generate lỗi), compression_ratio_threshold
        # (transcript bị lặp/nén bất thường, dấu hiệu kinh điển của
        # hallucination lặp từ).
        "no_speech_threshold": 0.6,
        "logprob_threshold": -1.0,
        "compression_ratio_threshold": 2.4,
        # Chặn thêm ở mức decoding: không cho phép lặp lại cùng 1 cụm 3 từ
        # liên tiếp — lưới an toàn cuối cùng chống lặp cụm từ như
        # "video tiếp theo của mình" lặp hàng chục lần.
        "no_repeat_ngram_size": 3,
    },
)
print("Model đã sẵn sàng.")


def transcribe(audio_path: str | None) -> str:
    """Hàm callback chính: nhận đường dẫn file audio, trả về text."""
    if audio_path is None:
        return "(Chưa có audio — hãy upload file hoặc thu âm trước khi bấm Submit)"

    result = asr_pipeline(audio_path)
    return result["text"].strip()


demo = gr.Interface(
    fn=transcribe,
    inputs=gr.Audio(sources=["upload", "microphone"], type="filepath", label="Audio đầu vào"),
    outputs=gr.Textbox(label="Transcript"),
    title="Whisper Fine-tuned — Nhận diện giọng nói miền Trung Việt Nam",
    description=(
        "Model Whisper-small đã fine-tune trên tập ViMD (region=Central). "
        "Upload file audio hoặc thu âm trực tiếp bằng micro để thử."
    ),
    examples=None,  # có thể thêm list đường dẫn file .wav mẫu vào đây nếu muốn
    # Lưu ý: KHÔNG truyền allow_flagging/flagging_mode ở đây — tên tham số này
    # đã đổi giữa các phiên bản Gradio (allow_flagging ở bản cũ, flagging_mode
    # từ Gradio 5.x). Bỏ hẳn tham số (mặc định flagging vẫn bật nhưng không
    # ảnh hưởng tới demo) để code không phụ thuộc version Gradio cụ thể.
)

if __name__ == "__main__":
    # Tự động bật share=True khi chạy trên Colab (link local 127.0.0.1 không
    # mở được từ trình duyệt thật vì Colab chạy trên máy ảo riêng, cần link
    # public tạm thời gradio.live). Khi chạy ở máy local thì không cần share.
    #
    # Lưu ý: script này thường được chạy qua `!python src/05_app_gradio.py`,
    # tức là 1 TIẾN TRÌNH CON hoàn toàn mới, khác với tiến trình Python của
    # chính notebook — nên kiểm tra `"google.colab" in sys.modules` sẽ luôn
    # ra False (module đó chỉ được import sẵn trong kernel notebook, không
    # tự có trong subprocess). Thay vào đó, kiểm tra biến môi trường Colab
    # đặt cho toàn bộ máy ảo (mọi tiến trình con đều kế thừa được).
    running_on_colab = "COLAB_RELEASE_TAG" in os.environ or "COLAB_GPU" in os.environ
    demo.launch(share=running_on_colab)
