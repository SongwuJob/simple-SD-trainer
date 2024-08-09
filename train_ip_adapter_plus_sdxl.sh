export MODEL_NAME="/path/to/your/stable-diffusion-xl-base-1.0"
export PRETRAIN_IP_ADAPTER_PATH="/path/to/your/.../sdxl_models/ip-adapter-plus_sdxl_vit-h.bin"
export IMAGE_ENCODER_PATH="/path/to/your/.../models/image_encoder"
export OUTPUT_DIR="ip-adapter"
export TRAIN_DIR="images"
export JSON_FILE="images/data.json"

accelerate launch ./SDXL/train_ip_adapter_plus_sdxl.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --image_encoder_path=$IMAGE_ENCODER_PATH \
  --pretrained_ip_adapter_path=$PRETRAIN_IP_ADAPTER_PATH \
  --data_json_file=$JSON_FILE \
  --data_root_path=$TRAIN_DIR \
  --mixed_precision="fp16" \
  --height=1024 --width=1024\
  --train_batch_size=2 \
  --dataloader_num_workers=4 \
  --learning_rate=1e-05 \
  --weight_decay=0.01 \
  --output_dir=$OUTPUT_DIR \
  --save_steps=10000
