import torch.nn as nn
import torchvision


def get_deeplabv3_resnet50(num_classes=1, pretrained=True):
    ''' gets pretrained model. '''
    return torchvision.models.segmentation.deeplabv3_resnet50(pretrained=pretrained, progress=True, num_classes=num_classes)


# dropout not used.
class CNN(nn.Module):
    ''' 
    classic CNN model.
    reference: github - 'Korean-OCR-Model-Design-based-on-Keras-CNN'
    input data: 32*32 normalized np array (according to reference)
    '''
    def __init__(self):
        super(CNN, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3, stride=1, padding=0, bias=True)
        self.relu1 = nn.ReLU() # 3*32*32 -> 128*30*30
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2) # 128*15*15
        # self.dropout1 = nn.Dropout2d()
        
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=0, bias=True)
        self.relu2 = nn.ReLU() # 128*15*15 -> 256*13*13
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2) # 256*6*6
        # self.dropout2 = nn.Dropout2d()
        
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=0, bias=True)
        self.relu3 = nn.ReLU() # 256*6*6 -> 512*4*4
        # self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2) 512*1*1
        # self.dropout3 = nn.Dropout2d()

        self.fc1 = nn.Linear(8192, 2350) # (input_features, output_features) 

    def forward(self, x):
        x = self.conv1(x) # [batch size, feature, width, height] = [64, channel, X, Y]
        x = self.relu1(x)
        x = self.maxpool1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        
        x = self.conv3(x)
        x = self.relu3(x)
        x = x.view(x.size(0), -1) # flatten
        
        x = self.fc1(x)

        return x
