import torch
from torch import nn
from torch.nn import LeakyReLU, Dropout

from monai.networks.nets.vit import ViT

from LN_malignancy_GNN.pytorch.resnet import * 
from LN_malignancy_GNN.pytorch.sgc_cnn import SGC_CNN 
from LN_malignancy_GNN.pytorch.densenet import DenseNet3d 
from LN_malignancy_GNN.pytorch.resnet_spottune import SpotTune



def resnet18(n_classes=1, in_channels=3, dropout=0.0, blocks=BasicBlock):
    return ResNet(blocks, [2,2,2,2], in_channels=in_channels, n_classes=n_classes, dropout=dropout)

def resnet34(n_classes=1, in_channels=3, dropout=0.0, blocks=BasicBlock):
    return ResNet(blocks, [3,4,6,3], in_channels=in_channels, n_classes=n_classes, dropout=dropout)

def resnet50(n_classes=1, in_channels=3, dropout=0.0, blocks=Bottleneck):
    return ResNet(blocks, [3,4,6,3], in_channels=in_channels, n_classes=n_classes, dropout=dropout)

def resnet101(n_classes=1, in_channels=3, dropout=0.0, blocks=Bottleneck):
    return ResNet(blocks, [3,4,23,3], in_channels=in_channels, n_classes=n_classes, dropout=dropout)

def resnet152(n_classes=1, in_channels=3, dropout=0.0, blocks=Bottleneck):
    return ResNet(blocks, [3,8,36,3], in_channels=in_channels, n_classes=n_classes, dropout=dropout)
def resnet200(n_classes=1, in_channels=3, dropout=0.0, blocks=Bottleneck):
    return ResNet(blocks, [3,24,36,3], in_channels=in_channels, n_classes=n_classes, dropout=dropout)

def densenet3d(n_classes=1, in_channels=64, dropout = 0.0):
    return DenseNet3d(n_classes=n_classes, in_channels=in_channels, dropout_p=dropout)

def svm_preds(n_classes=1, in_channel=64, dropout= 0.0):
    return

def spottune18(n_classes=1, in_channels=3, dropout=0.0, blocks=BasicBlock):
    return SpotTune(main='main18', in_channels=in_channels, num_classes=n_classes, dropout=dropout)

def spottune34(n_classes=1, in_channels=3, dropout=0.0, blocks=BasicBlock):
    return SpotTune(main='main34', in_channels=in_channels, num_classes=n_classes, dropout=dropout)

def spottune50(n_classes=1, in_channels=3, dropout=0.0, blocks=Bottleneck):
    return SpotTune(main='main50', in_channels=in_channels, num_classes=n_classes, dropout=dropout)

def spottune101(n_classes=1, in_channels=3, dropout=0.0, blocks=Bottleneck):
    return SpotTune(main='main101', in_channels=in_channels, num_classes=n_classes, dropout=dropout)

def spottune152(n_classes=1, in_channels=3, dropout=0.0, blocks=Bottleneck):
    return SpotTune(main='main152', in_channels=in_channels, num_classes=n_classes, dropout=dropout)

def spottune200(n_classes=1, in_channels=3, dropout=0.0, blocks=Bottleneck):
    return SpotTune(main='main200', in_channels=in_channels, num_classes=n_classes, dropout=dropout)


def vit(n_classes=1, in_channels=3, dropout=0.0):
    return ViT(
        in_channels=in_channels,
        img_size = (50,50,20),
        patch_size = 16,
        hidden_size = 768,
        mlp_dim = 3072,
        num_layers = 10,
        num_heads = 12,
        pos_embed = "conv",
        classification = True,
        num_classes = n_classes,
        dropout_rate = dropout,
        spatial_dims = 3,
        )

class LNCNN(nn.Module):
    def __init__(self, n_classes, in_channels, dropout):
        super(LNCNN, self).__init__()

        self.cn1 = nn.Conv3d(in_channels, 64, kernel_size=(5,5,5), stride=(1,1,1), padding='valid', bias=False)
        self.cn2 = nn.Conv3d(64, 64, kernel_size=(3,3,3), stride=(1,1,1), padding='valid', bias=False)
        self.cn3 = nn.Conv3d(64, 64, kernel_size=(3,3,3), stride=(1,1,1), padding='valid', bias=False)
        self.cn4 = nn.Conv3d(64, 64, kernel_size=(3,3,3), stride=(1,1,1), padding='valid', bias=False)
        self.cn5 = nn.Conv3d(64, 64, kernel_size=(3,3,3), stride=(1,1,1), padding='valid', bias=False)
        self.maxpool1 = nn.MaxPool3d(kernel_size=(2,2,2), stride=2)
        #self.cn6 = nn.Conv3d(64, 64, kernel_size=(3,3,3), stride=(1,1,1), padding='valid', bias=False)
        self.cn6 = nn.Conv3d(64, 64, kernel_size=(3,3,3), stride=(1,1,1), padding='same', bias=False)
        #self.cn7 = nn.Conv3d(64, 64, kernel_size=(2,2,2), stride=(1,1,1), padding='valid', bias=False)
        self.cn7 = nn.Conv3d(64, 64, kernel_size=(2,2,2), stride=(1,1,1), padding='same', bias=False)
        self.cn8 = nn.Conv3d(64, 64, kernel_size=(3,3,3), stride=(1,1,1), padding='same',bias=False)
        self.maxpool2 = nn.MaxPool3d(kernel_size=(3,3,3), stride=2)
        self.cn9 = nn.Conv3d(64, 64, kernel_size=(3,3,3), stride=(1,1,1), padding='same', bias=False)
        self.cn10 = nn.Conv3d(64, 64, kernel_size=(3,3,1), stride=(1,1,1), padding='same', bias=False)
        self.cn11 = nn.Conv3d(64, 64, kernel_size=(3,3,1), stride=(1,1,1), padding='same', bias=False)
        self.cn12 = nn.Conv3d(64, 32, kernel_size=(3,3,1), stride=(1,1,1), padding='same', bias=False)

        self.bn1 = nn.BatchNorm3d(64)
        self.bn2 = nn.BatchNorm3d(64)
        self.bn3 = nn.BatchNorm3d(64)
        self.bn4 = nn.BatchNorm3d(64)
        self.bn5 = nn.BatchNorm3d(64)
        self.bn6 = nn.BatchNorm3d(64)
        self.bn7 = nn.BatchNorm3d(64)
        self.bn8 = nn.BatchNorm3d(64)
        self.bn9 = nn.BatchNorm3d(64)
        self.bn10 = nn.BatchNorm3d(64)
        self.bn11 = nn.BatchNorm3d(64)
        self.bn12 = nn.BatchNorm3d(32)

        
        self.dropout = nn.Dropout(dropout)
        self.dropout05 = nn.Dropout(0.05)
        self.flatten = nn.Flatten()
        #self.avgpool = nn.AdaptiveAvgPool3d((1,1,1))
        #self.linear = nn.Linear(27104, n_classes)
        self.linear = nn.Linear(2592, n_classes)
        #self.classify = nn.Linear(256, n_classes)
        self.relu = nn.ReLU()


    def forward(self, x):
        x = self.cn1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout05(x)
        x = self.cn2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout05(x)
        x = self.cn3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.dropout05(x)
        x = self.cn4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.dropout05(x)
        x = self.cn5(x)
        x = self.bn5(x)
        x = self.relu(x)
        x = self.dropout05(x)
        x = self.maxpool1(x)
        x = self.cn6(x)
        x = self.bn6(x)
        x = self.relu(x)
        x = self.dropout05(x)
        x = self.cn7(x)
        x = self.bn7(x)
        x = self.relu(x)
        x = self.dropout05(x)
        x = self.cn8(x)
        x = self.bn8(x)
        x = self.relu(x)
        x = self.dropout05(x)
        x = self.maxpool2(x)
        x = self.cn9(x)
        x = self.bn9(x)
        x = self.relu(x)
        x = self.dropout05(x)
        x = self.cn10(x)
        x = self.bn10(x)
        x = self.relu(x)
        x = self.dropout05(x)
        x = self.cn11(x)
        x = self.bn11(x)
        x = self.relu(x)
        x = self.dropout05(x)
        x = self.cn12(x)
        x = self.bn12(x)
        x = self.relu(x)
        x = self.dropout05(x)
        #x = self.avgpool(x)
        x = self.flatten(x)
         
        x = self.linear(x)
        x = self.relu(x)
        x = self.dropout(x)

        #x = self.classify(x)

        return x.squeeze()
