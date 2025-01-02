from utils.utils import *

import torch
import torch.nn as nn
import torch.nn.functional as F

from configparser import ConfigParser
config = ConfigParser()

class PerceptualLoss(nn.Module):
    def __init__(self, device=None):
        super().__init__()
        self.feature_extractor = FeatureExtractor()
        self.feature_extractor.eval()

        if device is None:
            self.device = next(self.feature_extractor.parameters()).device
        else:
            self.device = device
            self.feature_extractor = self.feature_extractor.to(device)

        # VGG normalization constants
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

        self.content_weight = config['settings']['content_weight']
        self.style_weight = config['settings']['style_weight']
        self.style_weights = [0.2, 0.2, 0.3, 0.3]

        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def style_loss(self, style_feat_x, style_feat_y):
        total_loss = 0
        for i, (fx, fy, weight) in enumerate(zip(style_feat_x,
                                                 style_feat_y,
                                                 self.style_weights)):
            gram_x = gram_matrix(fx)
            gram_y = gram_matrix(fy)
            total_loss += weight * F.mse_loss(gram_x, gram_y)
        return total_loss

    def content_loss(self, x, y):
        return F.mse_loss(x, y)


    def forward(self, x, y_content, y_style):
        try:
            x = (x - self.mean) / self.std
            y_content = (y_content - self.mean) / self.std
            y_style = (y_style - self.mean) / self.std

            ### < Content loss: compares generated image with content image > ##
            content_feat_x = self.feature_extractor(x, [self.feature_extractor.content_layer])
            content_feat_y = self.feature_extractor(y_content, [self.feature_extractor.content_layer])
            content_loss = content_loss(content_feat_x, content_feat_y)

            ### < Style loss: compares generated image with style image > ##
            style_feat_x = self.feature_extractor(x, self.feature_extractor.style_layers)
            style_feat_y = self.feature_extractor(y_style, self.feature_extractor.style_layers)
            style_loss = self.compute_style_loss(style_feat_x, style_feat_y)

            ### ------ Weight and combine all losses ---------##
            final_content_loss = self.content_weight * content_loss
            final_style_loss = self.style_weight * style_loss

            return final_content_loss, final_style_loss

        except Exception as e:
            raise e