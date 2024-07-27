# the dataset used to process single resolution, can be used for lora training
import os
import json
import itertools
import random
import numpy as np

from PIL import Image
from PIL.ImageOps import exif_transpose
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import crop


# scales equally the image closed to the target size, and new size > target size
class CustomResize:
    def __init__(self, size):
        self.target_short, self.target_long = min(size), max(size)

    def __call__(self, img):
        width, height = img.size
        short, long = min(width, height), max(width, height)
        
        scale = self.target_short / short
        new_short = self.target_short
        new_long = int(long * scale)
        
        if new_long < self.target_long:
            scale = self.target_long / new_long
            new_short = int(new_short * scale)
            new_long = self.target_long
        new_size = (new_short, new_long) if width < height else (new_long, new_short)
        
        return img.resize(new_size, Image.BICUBIC)


class BaseDataset(Dataset):
    """
    A dataset to prepare images with the prompts for fine-tuning the model.
    It pre-processes the images.
    """

    def __init__(
        self,
        train_data_dir,
        json_file,
        tokenizer, 
        tokenizer_2,
        height=1024,
        width=1024,
        repeats=1, # you can set as 5
        center_crop=False,
        random_flip=False,
    ):
        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2
        self.size = (height, width) # Note that the PIL size is (w, h), while the size in transforms is (h, w)
        self.height = height
        self.width = width
        self.center_crop = center_crop
        self.random_flip = random_flip
        self.train_data_dir = train_data_dir

        # load train datas
        self.json_file = json_file
        self.data = json.load(open(json_file)) # list of dict: [{"image": "1.png", "text": "A dog"}]

        # loda images and prompts
        train_images = [os.path.join(self.train_data_dir, item["image"]) for item in self.data]
        train_prompts = [item["text"]for item in self.data]
        assert len(train_images) == len(train_prompts)

        self.train_images = []
        self.train_prompts = []
        # repeat train samples
        for img in train_images:
            self.train_images.extend(itertools.repeat(img, repeats))
        for txt in train_prompts:
            self.train_prompts.extend(itertools.repeat(txt, repeats))

        self._length = len(self.train_images)

        # preprocess the train images
        self.train_resize = CustomResize(self.size)
        self.train_crop = transforms.CenterCrop(self.size) if self.center_crop else transforms.RandomCrop(self.size)
        self.train_flip = transforms.RandomHorizontalFlip(p=1.0)
        self.train_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        text = self.train_prompts[index]
        image_file = self.train_images[index]

        # read image
        raw_image = Image.open(image_file).convert("RGB")
        
        # original size
        original_width, original_height = raw_image.size
        original_sizes = torch.tensor([original_height, original_width])

        image = self.train_resize(raw_image)
        if self.random_flip and random.random() < 0.5:
            # flip
            image = self.train_flip(image)
        if self.center_crop:
            y1 = max(0, int(round((image.height - self.height) / 2.0)))
            x1 = max(0, int(round((image.width - self.width) / 2.0)))
            image = self.train_crop(image)
        else:
            y1, x1, h, w = self.train_crop.get_params(image, self.size)
            image = crop(image, y1, x1, h, w)
        crop_top_lefts = torch.tensor([y1, x1])
        image = self.train_transforms(image)
        
        # get text and tokenize
        text_input_ids = self.tokenizer(
            text,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids
        
        text_input_ids_2 = self.tokenizer_2(
            text,
            max_length=self.tokenizer_2.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids
        
        return {
            "pixel_values": image,
            "input_ids_one": text_input_ids,
            "input_ids_two": text_input_ids_2,
            "original_sizes": original_sizes,
            "crop_top_lefts": crop_top_lefts
        }