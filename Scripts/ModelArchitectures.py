import torch
from efficientnet_pytorch import EfficientNet as en
from torch import nn
from torchvision.models import *


def Efficientnet_B0_(n_classes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = en.from_pretrained("efficientnet-b0")
    in_features = model._fc.in_features
    model._fc = nn.Linear(in_features, n_classes)

    # freeze all layers ( do not update gradients)
    for param in model.parameters():
        param.requires_grad = False

    # unfreeze last layer (update gradients)
    for param in model._fc.parameters():
        param.requires_grad = True

    model = torch.nn.DataParallel(model)
    model.to(device)
    return model

def Efficientnet_B7_(n_classes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = en.from_pretrained("efficientnet-b7")
    in_features = model._fc.in_features
    model._fc = nn.Linear(in_features, n_classes)

    # freeze all layers ( do not update gradients)
    for param in model.parameters():
        param.requires_grad = False

    # unfreeze last layer (update gradients)
    for param in model._fc.parameters():
        param.requires_grad = True

    model = torch.nn.DataParallel(model)
    model.to(device)
    return model

def VGG_():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = vgg19(weights=None).to(device)
    model.fc = nn.Linear(1000, 2)
    model = torch.nn.DataParallel(model)
    model.to(device)
    return model



def Efficientnet_B4_():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = en.from_pretrained("efficientnet-b4")
    in_features = model._fc.in_features
    model._fc = nn.Linear(in_features, 4)
    model = torch.nn.DataParallel(model)
    model.to(device)
    return model


def Efficientnet_B5_():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = en.from_pretrained("efficientnet-b5")
    in_features = model._fc.in_features
    model._fc = nn.Linear(in_features, 6)
    model = torch.nn.DataParallel(model)
    model.to(device)
    return model

def Other():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return densenet169(num_classes=2, weights=None).to(device)