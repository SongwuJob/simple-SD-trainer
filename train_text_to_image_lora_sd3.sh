export MODEL_NAME="/path/to/your/stable-diffusion-3-medium-diffusers"
export OUTPUT_DIR="lora/rank32"
export TRAIN_DIR="/path/to/your/data"
export JSON_FILE="/path/to/your/data/data.json"

accelerate launch ./stable_diffusion/train_text_to_image_lora_sd3.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$TRAIN_DATA_DIR \
  --output_dir=$OUTPUT_DIR \
  --json_file=$JSON_FILE \
  --mixed_precision="fp16" \
  --height=1024 --width=1024  \
  --random_flip \
  --train_batch_size=2 \
  --checkpointing_steps=1000 \
  --gradient_accumulation_steps=2 \
  --learning_rate=1e-4 \
  --text_encoder_lr=5e-6 \
  --rank=64 --text_encoder_rank=8 \
  --lr_scheduler="constant_with_warmup" --lr_warmup_steps=500 \
  --num_train_epochs=30 \
  --scale_lr --train_text_encoder \
  --seed=1337
