# simple-SD-trainer
As a AIGC rookie, I want to go ahead and try to reproduce some basic abilities of the text-to-image model, including Lora, ControlNet, IP-adapter, where you can use these abilities to realize a range of interesting AIGC plays!

To this end, we will keep a complete record of how these abilities are trained, and our future plans can be broadly categorized into:

- [Data Process](#data-process)
- [SDXL]
  - [Lora]
  - [ControlNet]
  - [IP-adapter]
  - [AnimateDiff]
- [DiT]
  - [PixArt]
  - [Stable Diffusion 3]
- [Application]
  - [Personalized avatar]
  - [Style Transformation]
  - [Virtual tryon]

## Data Process
Image caption is an important part of training text-to-image models, which can be used in Lora, ControlNet, etc. Common caption methods can be broadly categorized into two types:

- **SDWebUI Tagger**: You can use some tagger in webui (essentially a multi-classification model) to caption.
- **VLM**: VLM provides a better understanding of the dense semantics in the image, and is able to caption the image in detail, which is our recommended way.

In our experiments, we use [GLM-4v-9b](https://github.com/THUDM/GLM-4) to caption our training images, we use ``query = "please describe this image into prompt words, and reply us with keywords like \"xxx, xxx, xxx, xxx\""`` to make VLM output the image caption. For example, we can employ GLM-4v to caption the single image as follows:

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

gen_kwargs = {"max_length": 2500, "do_sample": True, "top_k": 1}
with torch.no_grad():
    outputs = model.generate(**inputs, **gen_kwargs)
    outputs = outputs[:, inputs['input_ids'].shape[1]:]
    caption = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(caption)
```
