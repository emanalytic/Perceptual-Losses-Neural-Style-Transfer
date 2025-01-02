import torch
import torch.nn as nn
from torchvision.models import vgg16, VGG16_Weights


class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = vgg16(weights=VGG16_Weights).eval()

        self.content_layer = '9'  # relu2_2
        self.style_layers = ['4', '9', '16', '23']  # relu1_2, relu2_2, relu3_3, relu4_3

        self.slices = nn.ModuleList()
        i = 0
        for layer in vgg.features.children():
            if isinstance(layer, nn.ReLU):
                i += 1
                layer = nn.ReLU(inplace=False)
            self.slices.append(layer)
            if str(i) == self.style_layers[-1]:
                break

    def forward(self, x, layer_ids):
        feat = []
        for i, layer in enumerate(self.slices):
            x = layer(x)
            if str(i) in layer_ids:
                feat.append(x)
        return feat

def gram_matrix(x):
    b, c, h, w = x.size()
    f = x.view(b, c, h * w)
    gram = torch.bmm(f, f.transpose(2, 1))
    return gram / (c * h * w)

