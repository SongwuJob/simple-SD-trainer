export MODEL_NAME="/path/to/your/model"
export OUTPUT_DIR="lora/rank32"
export TRAIN_DIR="/path/to/your/data"
export JSON_FILE="/path/to/your/data/data.json"

accelerate launch  ./SDXL/train_text_to_image_lora_sdxl.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$TRAIN_DIR \
  --output_dir=$OUTPUT_DIR \
  --json_file=$JSON_FILE \
  --height=1024 --width=1024  \
  --train_batch_size=2 \
  --random_flip \
  --rank=32 --text_encoder_rank=8 \
  --gradient_accumulation_steps=2 \
  --num_train_epochs=30 --repeats=5 \
  --checkpointing_steps=2000 \
  --learning_rate=1e-4 \
  --text_encoder_lr=1e-5 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=500 \
  --mixed_precision="fp16" \
  --train_text_encoder \
  --seed=1337 \
