import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision.models import vgg19
import math


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        vgg19_model = vgg19(pretrained=True)
        # Keep the shallow VGG layers used by the original perceptual loss path.
        self.feature_extractor = nn.Sequential(*list(vgg19_model.features.children())[:18])
        # if not requires_grad:
            # for param in self.parameters():
                # param.requires_grad = False
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def forward(self, img):
        return self.feature_extractor(img)
