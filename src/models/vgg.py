import torch.nn as nn
from torchvision import models


def get_vgg():
    vgg16 = models.vgg16(pretrained=True)

    class NewVGG16(nn.Module):
        def __init__(self):
            super().__init__()
            self.vgg16 = vgg16
            # for param in self.vgg16.features.parameters():
            #     param.requires_grad = False
            self.fc = nn.Sequential(nn.Linear(4096, 100),
                                    nn.Linear(100, 3))

        def forward(self, x):
            x = self.vgg16(x)
            x = self.fc(x)
            return x

    vgg16.classifier = nn.Sequential(*list(vgg16.classifier.children()))[:-1]

    return NewVGG16()
