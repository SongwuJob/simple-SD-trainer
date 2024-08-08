import os
import json
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class ControlNetDataset(Dataset):
    """
    A dataset to prepare images with the conditional image for training the ControlNet model.
    It pre-processes the images.
    """
    def __init__(
        self,
        train_data_dir,
        json_file,
        text_encoders,
        tokenizers,
        proportion_empty_prompts,
        device,
        height=1024,
        width=1024,
        crops_coords_top_left_h=0,
        crops_coords_top_left_w=0,
        is_train=True,
    ):  
        self.text_encoders = text_encoders
        self.tokenizers = tokenizers
        self.proportion_empty_prompts = proportion_empty_prompts
        self.size = (height, width)
        self.crops_coords_top_left = (crops_coords_top_left_h, crops_coords_top_left_w)
        self.train_data_dir = train_data_dir
        self.is_train = is_train
        self.device = device

        # Load training data
        with open(json_file, 'r') as f:
            self.data = json.load(f)

        # Define image transformations
        self.image_transforms = transforms.Compose([
            transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(self.size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

        self.conditioning_image_transforms = transforms.Compose([
            transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(self.size),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        text = item["text"]
        image = Image.open(os.path.join(self.train_data_dir, item["image"])).convert("RGB")
        conditioning_image = Image.open(os.path.join(self.train_data_dir, item["conditioning_image"])).convert("RGB")

        image = self.image_transforms(image)
        conditioning_image = self.conditioning_image_transforms(conditioning_image)

        embeddings = self.compute_embeddings(text)

        return {
            "pixel_values": image,
            "conditioning_pixel_values": conditioning_image,
            "prompt_embeds": embeddings["prompt_embeds"],
            "text_embeds": embeddings["text_embeds"],
            "time_ids": embeddings["time_ids"]
        }
    
    def compute_embeddings(self, prompt):
        original_size = self.size
        target_size = self.size
        crops_coords_top_left = self.crops_coords_top_left

        prompt_embeds, pooled_prompt_embeds = self.encode_prompt(prompt)
        add_text_embeds = pooled_prompt_embeds

        # Adapted from pipeline.StableDiffusionXLPipeline._get_add_time_ids
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_time_ids = torch.tensor([add_time_ids])
        
        prompt_embeds = prompt_embeds.to(self.device)
        add_text_embeds = add_text_embeds.to(self.device)
        add_time_ids = add_time_ids.to(self.device, dtype=prompt_embeds.dtype)

        return {
            "prompt_embeds": prompt_embeds,
            "text_embeds": add_text_embeds,
            "time_ids": add_time_ids
        }
    
    def encode_prompt(self, prompt):
        prompt_embeds_list = []
        
        if random.random() < self.proportion_empty_prompts:
            prompt = ""
        elif isinstance(prompt, (list, np.ndarray)):
            # take a random caption if there are multiple
            prompt = random.choice(prompt) if self.is_train else prompt[0]

        with torch.no_grad():
            for tokenizer, text_encoder in zip(self.tokenizers, self.text_encoders):
                text_input_ids = tokenizer(
                    prompt,
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                ).input_ids
                prompt_embeds = text_encoder(
                    text_input_ids.to(text_encoder.device),
                    output_hidden_states=True,
                )

                # We are only interested in the pooled output of the final text encoder
                pooled_prompt_embeds = prompt_embeds[0]
                prompt_embeds = prompt_embeds.hidden_states[-2]
                bs_embed, seq_len, _ = prompt_embeds.shape
                prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
                prompt_embeds_list.append(prompt_embeds)

            prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
            pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
            return prompt_embeds, pooled_prompt_embeds
