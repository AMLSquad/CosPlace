
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

class ReverseLayerF(Function):
    # Forwards identity
    # Sends backward reversed gradients
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class GeoLocalizationNet(nn.Module):
    def __init__(self, backbone, fc_output_dim):
        super().__init__()
        self.backbone, features_dim, avg_layer = get_backbone(backbone)
        self.aggregation = nn.Sequential(
                L2Norm(),
                # For each channel, get only one value
                GeM(),
                Flatten(),
                nn.Linear(features_dim, fc_output_dim),
                L2Norm()
            )

        # Domain adaptation
        num_domains = 2
        self.avg_layer = torch.nn.Sequential(*avg_layer)
        self.domain_classifier = nn.Sequential(
            self.avg_layer,
            L2Norm(),
            # For each channel, get only one value
            GeM(),
            Flatten(),
            nn.Linear(features_dim, num_domains),
        )
        


    
    def forward(self, x, alpha=None, flag_domain=False):
        features = self.backbone(x)
        
        if alpha is not None and flag_domain==True:
            
            # perform adaptation round
            # logits output dim is num_domains
            features = ReverseLayerF.apply(features, alpha)
            return self.domain_classifier(features)

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
