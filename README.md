# simple-SD-trainer
As a AIGC rookie, I want to go ahead and try to reproduce some basic abilities of the text-to-image model, including Lora, ControlNet, IP-adapter, where you can use these abilities to realize a range of interesting AIGC plays! (if you want, we could add some detailed notes for easy reading)

To this end, we will keep a complete record of how these abilities are trained, and our future plans can be broadly categorized into:

- [Data Process](#data-process)
- [SDXL](#sdxl)
  - [Lora](#lora)
  - [ControlNet](#controlnet)
  - [IP-adapter]
  - [AnimateDiff]
- [DiT]
  - [AuraFlow]
  - [PixArt]
  - [Stable Diffusion 3]
- [Application]
  - [Personalized avatar]
  - [Style Transformation]
  - [Virtual tryon]

## Data Process

### Image caption
Image caption is an important part of training text-to-image models, which can be used in Lora, ControlNet, etc. Common caption methods can be broadly categorized into two types:

- **SDWebUI Tagger**: This method involves using a tagger in the web UI, which essentially functions as a multi-classification model, to generate captions.
- **VLM**: VLM offers a better understanding of the dense semantics within an image，and is capable of providing detailed captions, which is our recommended approach.

In our experiments, we use [GLM-4v-9b](https://github.com/THUDM/GLM-4) to caption our trained images. Specifically, we use ``query = "please describe this image into prompt words, and reply us with keywords like \"xxx, xxx, xxx, xxx\""`` prompt the VLM to output the image caption. For example, we can employ GLM-4v to caption the single image as follows:

```python
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda"

tokenizer = AutoTokenizer.from_pretrained("THUDM/glm-4v-9b", trust_remote_code=True)

query = "please describe this image into prompt words, and reply us with keywords like \"xxx, xxx, xxx, xxx\""
image = Image.open("your image").convert('RGB')
inputs = tokenizer.apply_chat_template([{"role": "user", "image": image, "content": query}],
                                       add_generation_prompt=True, tokenize=True, return_tensors="pt",
                                       return_dict=True)  # chat mode

inputs = inputs.to(device)
model = AutoModelForCausalLM.from_pretrained(
    "THUDM/glm-4v-9b",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True
).to(device).eval()

gen_kwargs = {"max_new_tokens": 77, "do_sample": True, "top_k": 1}
with torch.no_grad():
    outputs = model.generate(**inputs, **gen_kwargs)
    outputs = outputs[:, inputs['input_ids'].shape[1]:]
    caption = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(caption)
```

For large amounts of training data, we can use ``caption.py`` in the ``data_process`` directory. Specifically, you can use ``use_buckets = True`` to conduct [ARB (Aspect Ratio Bucket)](https://civitai.com/articles/2056), there are two ways to preprocess trained images:

- **single resolution**: just provide the single resolution images, don't need conduct ARB.
```bash
python data_process/caption.py --train_image_dir "/path/to/your/data" --trigger_word "KongFu Panda"
```

- **multiple resolution**: provide multiple resolution images, which might need enough trained images and average size distribution.
```bash
python data_process/caption.py --train_image_dir "/path/to/your/data" --use_buckets --trigger_word "KongFu Panda"
```

The processed ``data.json`` format as follows:
```json
[
    {
        "image": "1.jpg",
        "text": "white hair, anime style, pink background, long hair, jacket, black and red top, earrings, rosy cheeks, large eyes, youthful, fashion, illustration, manga, character design, vibrant colors, hairstyle, clothing, accessories, earring design, artistic, contemporary, youthful fashion, graphic novel, digital drawing, pop art influence, soft shading, detailed rendering, feminine aesthetic"
    },
    {
        "image": "2.jpg",
        "text": "cute, anime-style, girl, long, wavy, hair, green, plaid, blazer, blush, big, expressive, eyes, hoop, earrings, soft, pastel, colors, youthful, innocent, charming, fashionable"
    }
]
```

## SDXL
Stable Diffusion XL (SDXL) is an advanced variant of [IDM](https://arxiv.org/abs/2112.10752) designed to generate high-quality images from textual descriptions. Building upon the original SD1.5(2.1), SDXL offers enhanced capabilities and improved performance, making it a powerful tool for various applications in the field of generative AI.

### Lora
Our Lora training code [train_text_to_image_lora_sdxl.py](/SDXL/train_text_to_image_lora_sdxl.py) is modified from [diffusers](https://github.com/huggingface/diffusers/tree/main/examples/text_to_image) and [kohya-ss](https://github.com/kohya-ss/sd-scripts). 

- We rewrite the dataset as ``BaseDataset.py`` and ``ARBDataset.py`` in ``dataset`` directory.
- We remove some parameters inside the diffusers for simplying training process, and adjust some settings.

After captioning the complete trained images, we can conduct ``sh train_text_to_image_lora_sdxl.sh`` to train your lora model:
```bash
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
  --checkpointing_steps=1000 \
  --learning_rate=1e-4 \
  --text_encoder_lr=1e-5 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=500 \
  --mixed_precision="fp16" \
  --train_text_encoder \
  --seed=1337 \
```

### ControlNet
Our ControlNet training code [train_controlnet_sdxl.py](/SDXL/train_controlnet_sdxl.py) is modified from [diffusers](https://github.com/huggingface/diffusers/tree/main/examples/text_to_image). 

- We rewrite the dataset as ``ControlNetDataset.py`` in ``dataset`` directory.
- We rewrite the data load process, and remove some parameters inside the diffusers for simplying training process.

To test training your controlnet, you can download a controlnet data on hugging face, like [controlnet_sdxl_animal](https://huggingface.co/datasets/HZ0504/controlnet_sdxl_animal/tree/main). Meanwhile, you need to simply preprocess these training data as follows:

- Training data directory structure
  ```shell
   controlnet_data
      ├──images/  (image files)
      │  ├──0.png
      │  ├──1.png
      │  ├──......
      ├──conditioning_images/  (conditioning image files)
      │  ├──0.png
      │  ├──1.png
      │  ├──......
      ├──data.json
   ```

- The ``data.json`` format
```json
[
    {
        "text": "a person walking a dog on a leash",
        "image": "images/1.png",
        "conditioning_image": "conditioning_images/1.png"
    },
    {
        "text": "a woman walking her dog in the park",
        "image": "images/2.png",
        "conditioning_image": "conditioning_images/2.png"
    }
]
```

After preparing the complete trained images, we can conduct ``sh train_controlnet_sdxl.sh`` to train your controlnet model:
```bash
export MODEL_DIR="/path/to/your/model"
export OUTPUT_DIR="controlnet"
export TRAIN_DIR="controlnet_data"
export JSON_FILE="controlnet_data/data.json"

accelerate launch ./SDXL/train_controlnet_sdxl.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --train_data_dir=$TRAIN_DIR \
 --output_dir=$OUTPUT_DIR \
 --json_file=$JSON_FILE \
 --mixed_precision="fp16" \
 --width=1024 --height=1024\
 --learning_rate=1e-5 \
 --lr_scheduler="constant_with_warmup" \
 --lr_warmup_steps=500 \
 --train_batch_size=1 \
 --gradient_accumulation_steps=2 \
 --seed=1337 \
```


