export MODEL_DIR="/path/to/your/model"
export OUTPUT_DIR="controlnet"
export TRAIN_DIR="controlnet_data"
export JSON_FILE="controlnet_data/data.json"

accelerate launch ./stable_diffusion/train_controlnet_sdxl.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --train_data_dir=$TRAIN_DIR \
 --output_dir=$OUTPUT_DIR \
 --json_file=$JSON_FILE \
 --mixed_precision="fp16" \
 --width=1024 --height=1024\
 --learning_rate=1e-5 \
 --checkpointing_steps=1000 \
 --num_train_epochs=5 \
 --lr_scheduler="constant_with_warmup" \
 --lr_warmup_steps=500 \
 --train_batch_size=1 --dataloader_num_workers=4\
 --gradient_accumulation_steps=2 \
 --seed=1337 \
