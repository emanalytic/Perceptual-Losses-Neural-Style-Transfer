from utils.utils import *

import torch
import torch.nn as nn
import torch.nn.functional as F

from configparser import ConfigParser
config = ConfigParser()

FEATURE_INDEX = config.get('settings', 'feature_index')


class FeatureReconstructionLoss(nn.Module):
    def __init__(self):
        super(FeatureReconstructionLoss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, y_hat, y):
        return self.mse_loss(y_hat, y)


class StyleReconstructionLoss(nn.Module):
    def __init__(self):
        super(StyleReconstructionLoss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, prediction, target):
        prediction_gram_matrix = self.gram_matrix(prediction)
        target_gram_matrix = self.gram_matrix(target)

        return self.mse_loss(prediction_gram_matrix, target_gram_matrix)

    def gram_matrix(self, y):
        b, ch, h, w = y.size()
        features = y.view(b, ch, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (ch * h * w)
        return gram

class PerceptualLoss(nn.Module):
    def __init__(self, alpha, beta):
        super(PerceptualLoss, self).__init__()

        self.alpha = alpha
        self.beta = beta
        self.style_loss = StyleReconstructionLoss()
        self.feature_loss = FeatureReconstructionLoss()

    def forward(self, y_hat, y_style, y_content):
        style_loss = 0.0
        feature_loss = self.alpha * self.feature_loss(y_hat[FEATURE_INDEX], y_content[FEATURE_INDEX].detach())

        for current_y_hat, current_y_style in zip(y_hat, y_style):
            style_loss += self.style_loss(current_y_hat, current_y_style[:len(current_y_hat)].detach())

        style_loss *= self.beta

        return feature_loss + style_loss