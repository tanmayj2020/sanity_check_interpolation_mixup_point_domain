import torch.nn as nn
import torchvision.models as backbone_
import torch
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VGG_Network(nn.Module):
    def __init__(self, hp):
        super(VGG_Network, self).__init__()
        self.features = backbone_.vgg16(pretrained=True).features

        self.features._modules["0"] = nn.Conv2d(hp.channels, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.pool_method = nn.AdaptiveMaxPool2d(1)

        if hp.dataset_name == "TUBerlin":
            num_class = 250
        else:
            num_class = 125
        self.classifier = nn.Linear(512, num_class)

    def forward(self, x):
        x = self.features(x)
        feats = self.pool_method(x)
        features = torch.flatten(feats, 1)
        out = self.classifier(features)
        return out

class Resnet_Network(nn.Module):
    def __init__(self, hp):
        super(Resnet_Network, self).__init__()
        backbone = backbone_.resnet50(pretrained=True)  # resnet50, resnet18, resnet34

        self.features = nn.Sequential()
        for name, module in backbone.named_children():
            if name not in ["avgpool", "fc"]:
                self.features.add_module(name, module)

        # self.features.conv1 = nn.Conv2d(hp.channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.pool_method = nn.AdaptiveMaxPool2d(1)  # as default

        if hp.dataset_name == "TUBerlin":
            num_class = 250
        else:
            num_class = 125

        self.classifier = nn.Linear(2048, num_class)

    def forward(self, x):
        x = self.features(x)
        x = self.pool_method(x)
        features = torch.flatten(x, 1)
        out = self.classifier(features)
        return nn.Sigmoid()(out)
        