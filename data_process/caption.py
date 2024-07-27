import torch
import argparse
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

import os
import json
import ast
from transformers import AutoModelForCausalLM, AutoTokenizer

# GLM-4v output
# ```json
# {
#   "prompt": "A white-haired anime girl, wearing a white baseball cap, a red and black sports jacket, and a red turtleneck sweater. She has a calm and serious expression, with a single earring visible. The background is a uniform pink color."
# }
# ```
def refine_caption(caption):
    text = caption.replace("```json", "").replace("```","")
    return text.lower().rstrip('.')

def resize_image(image, target_sizes):
    original_width, original_height = image.size
    original_ratio = original_width / original_height

    closest_size = None
    min_diff = float('inf')

    for target_width, target_height in target_sizes:
        target_ratio = target_width / target_height
        diff = abs(target_ratio - original_ratio)
        if diff < min_diff:
            min_diff = diff
            closest_size = (target_width, target_height)

    if closest_size is None:
        raise ValueError("Not found the suitable size!")

    target_width, target_height = closest_size
    if original_width < original_height:
        new_width = target_width
        new_height = int(target_width / original_ratio)
    else:
        new_height = target_height
        new_width = int(target_height * original_ratio)

    if new_width < target_width or new_height < target_height:
        if new_width < target_width:
            new_width = target_width
            new_height = int(target_width / original_ratio)
        if new_height < target_height:
            new_height = target_height
            new_width = int(target_height * original_ratio)

    resized_image = image.resize((new_width, new_height), Image.BICUBIC)
    return resized_image

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_image_dir', type=str, default="images", help= "save data.json path")
    parser.add_argument('--query', type=str, default="please describe this image into prompt words, and reply us with keywords like \"xxx, xxx, xxx, xxx\"", help= "the query for vlm model")
    parser.add_argument('--use_buckets', type=bool, default=False, help= "perform aspect ratio bucket")
    parser.add_argument('--device', type=str, default="cuda", help= "device")
    parser.add_argument('--max_sequence_length', type=int, default=77, help="clip max length is 77")
    parser.add_argument('--model_path', type=str, default="/pretrain_models/glm-4v-9b", help="we recommend using glm-4v")
    parser.add_argument('--trigger_word', type=str, default="HkStyle", help="trigger the model to generate a specific object")

    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if args.use_buckets:
        target_sizes = [(1024, 1024), (1152, 896), (896, 1152), (1216, 832), (832, 1216), (1344, 768), (768, 1344), (1280, 768), (768, 1280)]
    
    # transformers==4.40.2
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).to(args.device).eval()
    gen_kwargs = {"max_new_tokens": args.max_sequence_length, "do_sample": True, "top_k": 1}

    # Default reads all image files under a dir and stores data.json in the directory
    json_file = []
    image_extensions = ('.jpg', '.jpeg', '.png')

    for file in os.listdir(args.train_image_dir):
        if file.lower().endswith(image_extensions):
            image = Image.open(os.path.join(args.train_image_dir, file)).convert('RGB')

            # if you use ARB (Aspect Ratio Bucket), adjusts to the size of the nearest bucket
            if args.use_buckets:
                image = resize_image(image, target_sizes)
                image.save(file)

            inputs = tokenizer.apply_chat_template([{"role": "user", "image": image, "content": args.query}],
                                        add_generation_prompt=True, tokenize=True, return_tensors="pt",
                                        return_dict=True).to(args.device)

            with torch.no_grad():   
                outputs = model.generate(**inputs, **gen_kwargs)
                outputs = outputs[:, inputs['input_ids'].shape[1]:]
                caption = tokenizer.decode(outputs[0], skip_special_tokens=True)
                text = refine_caption(caption) if args.trigger_word is None else args.trigger_word + ", " + refine_caption(caption)

            sample = {
                "image": file,
                "text": text
            }

            print(sample)
            json_file.append(sample)
            
    with open(os.path.join(args.train_image_dir, 'data.json'), 'w', encoding='utf-8') as f:
        json.dump(json_file, f, ensure_ascii=False, indent=4)