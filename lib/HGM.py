import torch
import torch.nn as nn
import torch.nn.functional as F


class HGM(nn.Module):
    def __init__(self, in_channels):
        super(HGM, self).__init__()

        # 高频门控生成器
        self.hf_gate = nn.Sequential(
            nn.Conv2d(1, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid()
        )

        # 特征融合层
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels * 3, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, high_information, x, pred=None):
        # 1. 高频特征尺寸对齐
        high_information = F.interpolate(high_information, size=(x.size()[2], x.size()[3]), mode='bilinear',
                                         align_corners=True)

        # 2. 生成高频门控信号并注入
        gate = self.hf_gate(high_information)
        gated_x = x * gate

        # 3. 反向注意力机制
        if pred is not None:
            pred = torch.sigmoid(pred)
            if pred.size()[2] != x.size()[2]:
                pred = F.interpolate(pred, size=(x.size()[2], x.size()[3]), mode='bilinear', align_corners=True)
            background_x = x * (1 - pred)
        else:
            background_x = x

        # 4. 多源融合
        fusion_feature = torch.cat([x, gated_x, background_x], dim=1)
        fusion_feature = self.fusion_conv(fusion_feature)

        return fusion_feature
