import torch.nn as nn
import math
import copy
import torch.utils.model_zoo as model_zoo
import torch


class MultiwayResnet_AllTrained(nn.Module):
    def __init__(self):
        super(MultiwayResnet_AllTrained, self).__init__()

        resnet_bin0 = torch.load('/home/hdl2/Desktop/SonoFetalImage/metrics/binary_resnets/bin0_Sequential.pkl')
        resnet_bin1 = torch.load('/home/hdl2/Desktop/SonoFetalImage/metrics/binary_resnets/bin1_Sequential.pkl')
        resnet_bin2 = torch.load('/home/hdl2/Desktop/SonoFetalImage/metrics/binary_resnets/bin2_Sequential.pkl')
        resnet_bin3 = torch.load('/home/hdl2/Desktop/SonoFetalImage/metrics/binary_resnets/bin3_Sequential.pkl')
        resnet_bin4 = torch.load('/home/hdl2/Desktop/SonoFetalImage/metrics/binary_resnets/bin4_Sequential.pkl')
        resnet_bin5 = torch.load('/home/hdl2/Desktop/SonoFetalImage/metrics/binary_resnets/bin5_Sequential.pkl')
        resnet_bin6 = torch.load('/home/hdl2/Desktop/SonoFetalImage/metrics/binary_resnets/bin6_Sequential.pkl')
        resnet_bin7 = torch.load('/home/hdl2/Desktop/SonoFetalImage/metrics/binary_resnets/bin7_Sequential.pkl')
        resnet_bin8 = torch.load('/home/hdl2/Desktop/SonoFetalImage/metrics/binary_resnets/bin8_Sequential.pkl')
        resnet_bin9 = torch.load('/home/hdl2/Desktop/SonoFetalImage/metrics/binary_resnets/bin9_Sequential.pkl')

        bin_resnets = [resnet_bin0, resnet_bin1, resnet_bin2, resnet_bin3, resnet_bin4,
                       resnet_bin5, resnet_bin6, resnet_bin7, resnet_bin8, resnet_bin9]

        self.ways = nn.ModuleList(bin_resnets)

    def forward(self, x):
        xs = [self.ways[i](x) for i in range(len(self.ways))]
        out = torch.cat(xs, 1)
        out = nn.Sigmoid()(out)

        return out


class MultiwayResnet_FcTrained(nn.Module):
    def __init__(self):
        super(MultiwayResnet_FcTrained, self).__init__()

        resnet_bin0    = torch.load('/home/hdl2/Desktop/SonoFetalImage/metrics/multiway_resnet_bestways/bin0/gpu0/bin0_ResNet.pkl')
        resnet_bin0_fc = torch.load('/home/hdl2/Desktop/SonoFetalImage/metrics/multiway_resnet_bestways/bin0/gpu0/bin0_ResNet.pkl').fc
        resnet_bin1_fc = torch.load('/home/hdl2/Desktop/SonoFetalImage/metrics/multiway_resnet_bestways/bin1/gpu0/bin1_ResNet.pkl').fc
        resnet_bin2_fc = torch.load('/home/hdl2/Desktop/SonoFetalImage/metrics/multiway_resnet_bestways/bin2/gpu0/bin2_ResNet.pkl').fc
        resnet_bin3_fc = torch.load('/home/hdl2/Desktop/SonoFetalImage/metrics/multiway_resnet_bestways/bin3/gpu0/bin3_ResNet.pkl').fc
        resnet_bin4_fc = torch.load('/home/hdl2/Desktop/SonoFetalImage/metrics/multiway_resnet_bestways/bin4/gpu0/bin4_ResNet.pkl').fc
        resnet_bin5_fc = torch.load('/home/hdl2/Desktop/SonoFetalImage/metrics/multiway_resnet_bestways/bin5/gpu0/bin5_ResNet.pkl').fc
        resnet_bin6_fc = torch.load('/home/hdl2/Desktop/SonoFetalImage/metrics/multiway_resnet_bestways/bin6/gpu0/bin6_ResNet.pkl').fc
        resnet_bin7_fc = torch.load('/home/hdl2/Desktop/SonoFetalImage/metrics/multiway_resnet_bestways/bin7/gpu0/bin7_ResNet.pkl').fc
        resnet_bin8_fc = torch.load('/home/hdl2/Desktop/SonoFetalImage/metrics/multiway_resnet_bestways/bin8/gpu0/bin8_ResNet.pkl').fc
        resnet_bin9_fc = torch.load('/home/hdl2/Desktop/SonoFetalImage/metrics/multiway_resnet_bestways/bin9/gpu0/bin9_ResNet.pkl').fc
        precisions = torch.load('precision_01')

        bin_resnets = [resnet_bin0_fc, resnet_bin1_fc, resnet_bin2_fc, resnet_bin3_fc, resnet_bin4_fc,
                       resnet_bin5_fc, resnet_bin6_fc, resnet_bin7_fc, resnet_bin8_fc, resnet_bin9_fc]

        self.resnet_bin0 = resnet_bin0
        self.fc_ways = nn.ModuleList(bin_resnets)

    def forward(self, x):
        x = self.resnet_bin0.conv1(x)
        x = self.resnet_bin0.bn1(x)
        x = self.resnet_bin0.relu(x)
        x = self.resnet_bin0.maxpool(x)

        x = self.resnet_bin0.layer1(x)
        x = self.resnet_bin0.layer2(x)
        x = self.resnet_bin0.layer3(x)
        x = self.resnet_bin0.layer4(x)

        x = self.resnet_bin0.avgpool(x)
        fc_input = x.view(x.size(0), -1)

        xs = [self.fc_ways[i](fc_input) for i in range(len(self.fc_ways))]
        out = torch.cat(xs, 1)
        # out = nn.Softmax()(out)
        # out = nn.Sigmoid()(out)

        return out