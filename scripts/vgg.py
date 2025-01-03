from torchvision.models import vgg16, VGG16_Weights
from utils.config import Config
import torch

config = Config()
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ForwardHookManager:
    def __init__(self, model, layer_indices):
        vgg = vgg16(weights=VGG16_Weights.DEFAULT).features.to(DEVICE).eval()
        for params in vgg.parameters():
            params.requires_grad = False
        self.hooks = [list(model.children())[index].register_forward_hook(self.forward_hook) for index in layer_indices]
        self.clear_outputs()

    def __del__(self):
        for hook in self.hooks:
            hook.remove()

    def forward_hook(self, module, input, output):
        self.layer_outputs.append(output)

    def clear_outputs(self):
        self.layer_outputs = []