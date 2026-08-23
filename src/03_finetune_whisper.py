"""
03_finetune_whisper.py
========================
BƯỚC 3: Fine-tune openai/whisper-small trên tập train (miền Trung) đã tiền
xử lý ở bước 02, dùng transformers.Seq2SeqTrainer.

Chạy: python src/03_finetune_whisper.py
(Trên Colab: đổi Runtime -> Change runtime type -> GPU trước khi chạy)
"""

from __future__ import annotations

import os
import sys

import evaluate
import numpy as np
import torch
from datasets import load_from_disk
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)
from transformers.trainer_utils import get_last_checkpoint

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from data_utils import DataCollatorSpeechSeq2SeqWithPadding, normalize_text

wer_metric = evaluate.load("wer")


def load_data():
    print(f"Load dataset đã tiền xử lý từ: {config.PROCESSED_DIR}")
    ds = load_from_disk(config.PROCESSED_DIR)
    print(f"  train: {len(ds['train'])} mẫu | valid: {len(ds['valid'])} mẫu")
    return ds


def build_model_and_processor():
    print(f"Load processor & model gốc: {config.MODEL_NAME}")
    processor = WhisperProcessor.from_pretrained(
        config.MODEL_NAME, language=config.LANGUAGE, task=config.TASK
    )
    model = WhisperForConditionalGeneration.from_pretrained(config.MODEL_NAME)

    # Cấu hình generation: ép model luôn sinh ra tiếng Việt + task transcribe
    # (mặc định Whisper tự đoán ngôn ngữ từ vài giây đầu audio, có thể đoán
    # sai với giọng địa phương -> cố định trước cho chắc).
    model.generation_config.language = config.LANGUAGE
    model.generation_config.task = config.TASK
    model.generation_config.forced_decoder_ids = None  # API cũ, để None cho đúng chuẩn mới

    return model, processor


def build_compute_metrics(processor: WhisperProcessor):
    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # đưa các vị trí -100 (đã set ở DataCollator) trở lại pad_token_id
        # để tokenizer.decode không lỗi
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        pred_str = [normalize_text(s) for s in pred_str]
        label_str = [normalize_text(s) for s in label_str]

        wer = 100 * wer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}

    return compute_metrics


def main():
    has_gpu = torch.cuda.is_available()
    print(f"GPU khả dụng: {has_gpu}" + (f" ({torch.cuda.get_device_name(0)})" if has_gpu else ""))
    if not has_gpu:
        print(
            "[CẢNH BÁO] Không phát hiện GPU. Fine-tune trên CPU sẽ RẤT chậm "
            "(có thể mất nhiều giờ/ngày ngay cả với vài nghìn mẫu). Nếu đang "
            "ở Colab: Runtime > Change runtime type > chọn GPU (T4)."
        )

    ds = load_data()
    model, processor = build_model_and_processor()

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )
    compute_metrics = build_compute_metrics(processor)

    # -------------------------------------------------------------------
    # GIẢI THÍCH CÁC HYPERPARAMETER QUAN TRỌNG
    # (đây là phần bạn nên đọc kỹ để hiểu VÌ SAO chọn giá trị này, không chỉ
    # copy số mặc định)
    # -------------------------------------------------------------------
    training_args = Seq2SeqTrainingArguments(
        output_dir=config.FINETUNED_MODEL_DIR,

        # --- Batch size & gradient accumulation ---
        # GPU T4 free trên Colab chỉ có ~15GB VRAM. Whisper-small + activations
        # cho audio 30s đã chiếm khá nhiều bộ nhớ, nên batch size thực tế (per
        # device) phải nhỏ (4-8) để không bị lỗi CUDA out of memory.
        # Để bù lại (batch nhỏ -> gradient noisy, train không ổn định), ta
        # dùng gradient_accumulation_steps: cộng dồn gradient qua N batch nhỏ
        # trước khi update trọng số 1 lần -> "batch size hiệu dụng" =
        # per_device_train_batch_size * gradient_accumulation_steps = 8*2=16,
        # gần với batch size hay dùng khi fine-tune Whisper trong paper gốc,
        # mà vẫn vừa VRAM.
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        per_device_eval_batch_size=8,

        # --- Learning rate & warmup ---
        # 1e-5 là learning rate tiêu chuẩn khi FINE-TUNE (không phải train từ
        # đầu) Whisper — model đã học tốt tiếng Việt nói chung, ta chỉ cần
        # "chỉnh" nhẹ theo giọng miền Trung, nên learning rate phải nhỏ để
        # không phá vỡ (catastrophic forgetting) kiến thức đã học trước đó.
        # warmup_steps: tăng dần learning rate từ 0 lên giá trị đích trong
        # ~10% tổng số step đầu, giúp tránh update quá mạnh khi model/optimizer
        # chưa "khởi động" ổn định (đặc biệt quan trọng với learning rate nhỏ
        # + optimizer Adam). Tính theo % thay vì số cố định để tự co giãn
        # đúng tỉ lệ mỗi khi đổi MAX_TRAIN_STEPS trong config.py.
        learning_rate=1e-5,
        warmup_steps=int(0.1 * config.MAX_TRAIN_STEPS),

        # --- Độ dài quá trình train ---
        # Dùng max_steps thay vì num_train_epochs vì: với dataset nhỏ (Central
        # chỉ là 1 phần của ViMD), số step/epoch có thể ít, và ta muốn kiểm
        # soát chính xác tổng công sức train (dễ ước lượng thời gian trên
        # Colab free có giới hạn phiên làm việc). Giá trị lấy từ
        # config.MAX_TRAIN_STEPS (xem giải thích + cách đo tốc độ thực tế ở
        # đó) — theo dõi wer trên valid (log bên dưới) để biết có cần train
        # thêm/ít hơn không; nếu wer ngừng giảm (hoặc tăng lại = overfitting)
        # thì nên dừng sớm.
        max_steps=config.MAX_TRAIN_STEPS,

        # --- Tiết kiệm bộ nhớ ---
        # gradient_checkpointing: KHÔNG lưu toàn bộ activations trong forward
        # pass, mà tính lại 1 phần khi backward -> giảm đáng kể VRAM dùng,
        # đánh đổi bằng train chậm hơn ~20%. Trên GPU VRAM nhỏ như T4, đây là
        # đánh đổi đáng giá để tránh out-of-memory.
        gradient_checkpointing=True,

        # --- Mixed precision (fp16) ---
        # fp16=True: tính toán ở dạng số thực 16-bit thay vì 32-bit mặc định.
        # Lợi ích: nhanh hơn ~2x và giảm ~50% VRAM trên GPU có Tensor Cores
        # (T4 có). CHỈ bật khi có GPU hỗ trợ (torch.cuda.is_available()) —
        # trên CPU, fp16 không có tác dụng và có thể gây lỗi.
        fp16=has_gpu,

        # --- Đánh giá & lưu checkpoint trong lúc train ---
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,        # chỉ giữ 2 checkpoint gần nhất -> đỡ tốn đĩa
        predict_with_generate=True,  # bắt buộc để compute_metrics tính được WER
                                      # (cần model THỰC SỰ generate ra câu, không
                                      # chỉ tính loss)
        generation_max_length=225,   # độ dài tối đa câu sinh ra khi eval

        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,     # WER càng THẤP càng tốt (ngược logic accuracy)

        logging_steps=25,
        report_to=["none"],  # tắt wandb/tensorboard mặc định để khỏi cần API key;
                              # đổi thành ["tensorboard"] nếu muốn xem biểu đồ loss
        push_to_hub=False,   # xem README nếu muốn tự bật + đăng nhập huggingface-cli
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=ds["train"],
        eval_dataset=ds["valid"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        # Các bản transformers mới (>=4.46) đổi tên tham số này từ `tokenizer`
        # thành `processing_class` (nhận feature_extractor/tokenizer/processor
        # đều được — Trainer chỉ dùng nó để lưu kèm checkpoint, không ảnh
        # hưởng tới việc train).
        processing_class=processor.feature_extractor,
    )

    # --- Tự resume từ checkpoint gần nhất nếu có ---
    # Nếu output_dir đã chứa checkpoint từ lần chạy trước (vd. phiên Colab bị
    # ngắt giữa chừng), tự động tiếp tục từ đó thay vì train lại từ đầu.
    # CHỈ có tác dụng thật sự nếu output_dir trỏ tới nơi bền vững (Google
    # Drive/Kaggle output) — xem hướng dẫn WHISPER_PROJECT_OUTPUTS_DIR trong
    # config.py. Nếu output_dir nằm trên đĩa tạm của máy ảo, checkpoint cũng
    # mất theo khi ngắt phiên nên sẽ không tìm thấy gì để resume.
    last_checkpoint = get_last_checkpoint(training_args.output_dir) if os.path.isdir(
        training_args.output_dir
    ) else None
    if last_checkpoint is not None:
        print(f"\nTìm thấy checkpoint cũ tại: {last_checkpoint}")
        print("Sẽ tiếp tục train từ checkpoint này thay vì bắt đầu lại từ đầu.")
    else:
        print("\nKhông tìm thấy checkpoint cũ, bắt đầu train từ đầu.")

    print("Bắt đầu fine-tune... (theo dõi cột 'wer' trong log mỗi 500 step)")
    trainer.train(resume_from_checkpoint=last_checkpoint)

    print(f"\nLưu model đã fine-tune (bản tốt nhất theo valid WER) vào: {config.FINETUNED_MODEL_DIR}")
    trainer.save_model(config.FINETUNED_MODEL_DIR)
    processor.save_pretrained(config.FINETUNED_MODEL_DIR)

    print("Hoàn tất bước 3. Chạy tiếp 04_evaluate_wer.py để đánh giá trên tập test.")


if __name__ == "__main__":
    main()
