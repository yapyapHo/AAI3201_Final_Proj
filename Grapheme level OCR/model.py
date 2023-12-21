from torchvision import models
import torch.nn as nn

class OCR(nn.Module):
    def __init__(self, device):
        super(OCR, self).__init__()

        res34 = models.resnet34(pretrained=True).to(device)
        self.base = nn.Sequential(*list(res34.children())[:-2])

        # 마지막 fc layer 수정
        setattr(self, "fc%d" % 0, nn.Linear(512*8*8, 19))
        setattr(self, "fc%d" % 1, nn.Linear(512*8*8, 21))
        setattr(self, "fc%d" % 2, nn.Linear(512*8*8, 28))

    def forward(self, x):
        x = self.base(x)
        x = x.view(x.size(0), -1)
        output = []
        for i in range(3):
            output.append(getattr(self, "fc%d"%i)(x))

        return output