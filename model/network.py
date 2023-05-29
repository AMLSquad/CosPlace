
import torch
import logging
import torchvision
from torch import nn
import copy
from torch.autograd import Function
from model.layers import Flatten, L2Norm, GeM
import os
from model.autoencoder import Autoencoder

CHANNELS_NUM_IN_LAST_CONV = {
    "resnet18": 512,
    "resnet18_gldv2": 512,
    "resnet18_places": 512,
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
        x = GradientReversalFunction.apply(x)
        return x

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
    def __init__(self, backbone, fc_output_dim, domain_adaptation = False, backbone_path = None, aada=False):
        super().__init__()
        self.backbone, features_dim, _ = get_backbone(backbone, backbone_path)
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
        self.autoencoder = Autoencoder(features_dim) if aada == True else None
        self.backbone_grad_layer_3 = []
        self.backbone_grad_layer_4 = []
        self.aggregation_grad = []
        
        
    def save_bb_grad(self):
        self.backbone_grad_layer_3 = []
        self.backbone_grad_layer_4 = []
        self.aggregation_grad = []


        for name, child in self.backbone.named_children():
            if name == "6":  # Freeze layers before conv_3
                for params in child.parameters():
                    self.backbone_grad_layer_3.append(params.grad.clone())
            if name == "7":  # Freeze layers before conv_3
                for params in child.parameters():
                    self.backbone_grad_layer_4.append(params.grad.clone())
        for name,child in self.aggregation.named_children():
            if name == "1":
                for params in child.parameters():
                    self.aggregation_grad.append(params.grad.clone())
            if name == "3":
                for params in child.parameters():
                    self.aggregation_grad.append(params.grad.clone())


        
                
    
    def load_bb_grad(self):
        for name, child in self.backbone.named_children():
            if name == "6":  # Freeze layers before conv_3
                for idx,params in enumerate(child.parameters()):
                    params.grad = self.backbone_grad_layer_3[idx]
            if name == "7":  # Freeze layers before conv_3
                for idx,params in enumerate(child.parameters()):
                    params.grad = self.backbone_grad_layer_4[idx]

        idx = 0
        for name,child in self.aggregation.named_children():
            if name == "1":
                for params in child.parameters():
                    
                    params.grad = self.aggregation_grad[idx]
                    idx = idx + 1
            if name == "3":
                for params in child.parameters():
                    params.grad = self.aggregation_grad[idx]
                    idx = idx + 1

        

                    

            

    
    def forward(self, x, grl=False, aada=False, aada_linear = True, targets = None):
        features = self.backbone(x)
        if grl==True:
            # perform adaptation round
            # logits output dim is num_domains
            x =  self.discriminator(features)
            return x
        elif aada==True:
            features = self.aggregation(features)
            # perform adaptation round
            # logits output dim is num_domains
            if aada_linear:

                #features = torch.nn.functional.adaptive_avg_pool2d(features, (1,1))
                #features = features.view(features.shape[0], -1)
                features_sources = features[targets==0]
                features_targets = features[targets==1]
            else:
                features_sources = features[targets==0, :, :, :]
                features_targets = features[targets==1, :, :, :]
            
            ae_output_sources = self.autoencoder(features_sources)
            ae_output_targets = self.autoencoder(features_targets)
            return features_sources, features_targets, ae_output_sources, ae_output_targets
        return self.aggregation(features)


def get_backbone(backbone_name, backbone_path = None):
    if backbone_name.startswith("resnet"):
        if backbone_name == "resnet18":
            backbone = torchvision.models.resnet18(pretrained=True,)
        elif backbone_name == "resnet50":
            backbone = torchvision.models.resnet50(pretrained=True)
        elif backbone_name == "resnet101":
            backbone = torchvision.models.resnet101(pretrained=True)
        elif backbone_name == "resnet152":
            backbone = torchvision.models.resnet152(pretrained=True)
        elif backbone_name == "resnet18_places":
            if backbone_path is None:
                raise ValueError("You must specify the path to the pretrained model")
            backbone = torchvision.models.resnet18(num_classes = 365)
            file_path = backbone_path
            state_dict = torch.load(file_path, map_location=torch.device('cpu'))
            backbone.load_state_dict(state_dict)
        elif backbone_name == "resnet18_gldv2":
            if backbone_path is None:
                raise ValueError("You must specify the path to the pretrained model")
            backbone = torchvision.models.resnet18(num_classes = 512)
            file_path = backbone_path
            state_dict = torch.load(file_path, map_location=torch.device('cpu'))
            backbone.load_state_dict(state_dict)
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
