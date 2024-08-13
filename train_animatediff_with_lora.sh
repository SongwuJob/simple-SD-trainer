export MODEL_NAME="/path/to/your/Realistic_Vision_V5.1_noVAE"
export MOTION_MODULE="/path/to/your/animatediff-motion-adapter-v1-5-2"
export OUTPUT_DIR="animatediff"
export TRAIN_DIR="webvid"
export JSON_FILE="webvid/data.json"

accelerate launch  ./SDXL/AnimateDiff/train_animatediff_with_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --motion_module=$MOTION_MODULE \
  --train_data_dir=$TRAIN_DIR \
  --output_dir=$OUTPUT_DIR \
  --json_file=$JSON_FILE \
  --resolution=512  \
  --train_batch_size=1 \
  --rank=8 \
  --gradient_accumulation_steps=2 \
  --num_train_epochs=10 \
  --checkpointing_steps=10000 \
  --learning_rate=1e-5 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=500 \
  --mixed_precision="fp16" \
  --seed=1337 \
