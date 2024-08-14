# the dataset used to process single resolution, can be used for lora training
import os
import json
import itertools
import random

from PIL import Image
from PIL.ImageOps import exif_transpose
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import crop

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
        height=1024,
        width=1024,
        center_crop=False,
        random_flip=False,
    ):
        self.size = (height, width)
        self.height = height
        self.width = width
        self.center_crop = center_crop
        self.random_flip = random_flip
        self.train_data_dir = train_data_dir 
        self.data = json.load(open(json_file)) # list of dict: [{"image": "1.png", "text": "A dog"}]

        self.train_resize = CustomResize(self.size)
        self.train_crop = transforms.CenterCrop(self.size) if center_crop else transforms.RandomCrop(self.size)
        self.train_flip = transforms.RandomHorizontalFlip(p=1.0)
        self.train_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        text = item["text"]
        image = Image.open(os.path.join(self.train_data_dir, item["image"]))

        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = self.train_resize(image)
        if self.random_flip and random.random() < 0.5:
            image = self.train_flip(image)
        if self.center_crop:
            y1 = max(0, int(round((image.height - args.resolution) / 2.0)))
            x1 = max(0, int(round((image.width - args.resolution) / 2.0)))
            image = self.train_crop(image)
        else:
            y1, x1, h, w = self.train_crop.get_params(image, self.size)
            image = crop(image, y1, x1, h, w)
        image = self.train_transforms(image)
    
        examples = {}
        examples["images"] = image
        examples["prompts"] = text
        return examples
