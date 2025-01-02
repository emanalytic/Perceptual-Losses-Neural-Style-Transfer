##TODO: load content images in batches
from utils.config import Config

from PIL import Image
import os

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

config = Config()

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])
content_dir = config.get('paths', 'content_dir')
style_image = config.get('paths', 'style_image')
batch_size = int(config.get('settings', 'batch_size'))

class Data(Dataset):
    def __init__(self,
                 content_dir,
                 style_image,
                 transform=None):

        self.content_images = [os.path.join(content_dir, file)
                               for file in os.listdir(content_dir)
                               if file.lower().endswith(('.jpg', '.png', 'jpeg'))]
        self.style_image = style_image
        self.transform = transform

    def __len__(self):
        return len(self.content_images)

    def __getitem__(self, idx):

        content_img = Image.open(self.content_images[idx]).convert('RGB')
        style_img = Image.open(self.style_image).convert('RGB')
        if self.transform:
            content_img = self.transform(content_img)
            style_img = self.transform(style_img)
        return content_img, style_img

dataset = Data(content_dir, style_image, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


## sanity check
# if __name__ == "__main__":
#     content, style = dataset[0]
#     print(content.shape, style.shape)
#     for cb, sb in dataloader:
#         print('Content Batch Shape:', cb.shape)
#         print('Style Batch Shape:', sb.shape)
#         break





