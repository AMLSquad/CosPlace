
import torch
import logging
import torchvision
from torch import nn
import copy
from torch.autograd import Function
from model.layers import Flatten, L2Norm, GeM


CHANNELS_NUM_IN_LAST_CONV = {
        "resnet18": 512,
        "resnet50": 2048,
        "resnet101": 2048,
        "resnet152": 2048,
        "vgg16": 512,
    }

CHANNELS_AFTER_AVGPOOLING = {
    "resnet18": 512,
    "resnet50": 2048,
    "resnet101": 2048,
    "resnet152": 2048,
    "vgg16": 512,
}

class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.clone()
    @staticmethod
    def backward(ctx, grads):
        dx = -grads.new_tensor(1) * grads
        return dx, None

class GradientReversal(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = torch.nn.functional.adaptive_avg_pool2d(x, (1,1))
        x = x.view(x.shape[0], -1)
        return GradientReversalFunction.apply(x)

def get_discriminator(input_dim, num_classes=2):
    discriminator = nn.Sequential(
        GradientReversal(),
        nn.Linear(input_dim, 50),
        nn.ReLU(),
        nn.Linear(50, 20),
        nn.ReLU(),
        nn.Linear(20, num_classes)
    )
    return discriminator


class GeoLocalizationNet(nn.Module):
    def __init__(self, backbone, fc_output_dim, domain_adaptation = False):
        super().__init__()
        self.backbone, features_dim, _ = get_backbone(backbone)
        self.aggregation = nn.Sequential(
                L2Norm(),
                # For each channel, get only one value
                GeM(),
                Flatten(),
                nn.Linear(features_dim, fc_output_dim),
                L2Norm()
            )
        # Domain adaptation
        self.discriminator = get_discriminator(features_dim) if domain_adaptation == True else None
        
        


    
    def forward(self, x, grl=False):
        features = self.backbone(x)
        
        if grl==True:
            # perform adaptation round
            # logits output dim is num_domains
            return self.discriminator(features)

        return self.aggregation(features)


def get_backbone(backbone_name):
    if backbone_name.startswith("resnet"):
        if backbone_name == "resnet18":
            backbone = torchvision.models.resnet18(pretrained=True)
        elif backbone_name == "resnet50":
            backbone = torchvision.models.resnet50(pretrained=True)
        elif backbone_name == "resnet101":
            backbone = torchvision.models.resnet101(pretrained=True)
        elif backbone_name == "resnet152":
            backbone = torchvision.models.resnet152(pretrained=True)
        
        for name, child in backbone.named_children():
            if name == "layer3":  # Freeze layers before conv_3
                break
            for params in child.parameters():
                params.requires_grad = False
        logging.debug(f"Train only layer3 and layer4 of the {backbone_name}, freeze the previous ones")
        avg_layer = list(backbone.children())[-2:-1]
        layers = list(backbone.children())[:-2]  # Remove avg pooling and FC layer
    
    elif backbone_name == "vgg16":
        backbone = torchvision.models.vgg16(pretrained=True)
        avg_layer = list(backbone.features.children())[-2:-1]
        layers = list(backbone.features.children())[:-2]  # Remove avg pooling and FC layer
        for layer in layers[:-5]:
            for p in layer.parameters():
                p.requires_grad = False
        logging.debug("Train last layers of the VGG-16, freeze the previous ones")
    
    backbone = torch.nn.Sequential(*layers)
    
    features_dim = CHANNELS_NUM_IN_LAST_CONV[backbone_name]
    
    return backbone, features_dim, avg_layer
