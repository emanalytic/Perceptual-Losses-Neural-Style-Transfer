from torchvision.transforms import transforms
import torch

from utils.config import Config
from PIL import Image

config = Config()
VGG_MEAN = config.get('settings', 'vgg_mean')
VGG_STD = config.get('settings', 'vgg_std')
TARGET_SIZE = config.get('settings', 'target_size')
MODEL_KEY = config.get('keys', 'model_key')
OPTIMIZER_KEY = config.get('keys', 'optimizer_key')
EPOCH_KEY = config.get('keys', 'epoch_key')
MODEL_CHECKPOINT_FILEPATH = config.get('paths', 'model_checkpoints')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
STYLED_IMAGE_FILEPATH = config.get('paths', 'styled_image')


def show_image(image):
    image_transform = transforms.ToPILImage()
    image = image.squeeze().cpu()

    image = (image * VGG_STD[:, None, None]) + VGG_MEAN[:, None, None]

    image = image.clamp(0, 1)
    return image_transform(image)

def load_image(image):
    image = Image.open(image).convert('RGB')
    image_transforms = transforms.Compose([
        transforms.Resize(TARGET_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(VGG_MEAN, VGG_STD)
    ])
    image = image_transforms(image)
    return image


def save_model_checkpoint(model, optimizer, epoch, filename):
    checkpoint = {MODEL_KEY: model.state_dict(),
                  OPTIMIZER_KEY: optimizer.state_dict(),
                  EPOCH_KEY: epoch}
    torch.save(checkpoint, MODEL_CHECKPOINT_FILEPATH.format(filename))


def load_model_checkpoint(model, optimizer, filename):
    model_checkpoint = torch.load(MODEL_CHECKPOINT_FILEPATH.format(filename))
    model_state_dict, optimizer_state_dict = (model_checkpoint[MODEL_KEY],
                                              model_checkpoint[OPTIMIZER_KEY])
    model.load_state_dict(model_state_dict)
    optimizer.load_state_dict(optimizer_state_dict)


def style_image(model, image, filename=None):
    model.eval()

    image = image.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        styled_image = model(image)

    styled_image = show_image(styled_image)
    if filename != None:
        styled_image.save(STYLED_IMAGE_FILEPATH.format(filename))

    return styled_image
