##TODO: load content images in batches
import os
from pathlib import Path
from utils.config import Config
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

config = Config()

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

INPUT_SIZE = config.get('settings', 'input_size')
VGG_MEAN = config.get('settings', 'vgg_mean')
VGG_STD = config.get('settings', 'vgg_std')

class TrainingDataset(Dataset):
    def __init__(self, filepaths):
        self.filepaths = list(Path(filepaths).iterdir())
        self.transforms = transforms.Compose([
            transforms.Resize(INPUT_SIZE),
            transforms.CenterCrop(INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(VGG_MEAN, VGG_STD)
        ])

    def __getitem__(self, index):
        with Image.open(self.filepaths[index]).convert("RGB") as image:
            image = self.transforms(image)
            return image

    def __len__(self):
        return len(self.filepaths)





