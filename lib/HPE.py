import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import rgb_to_grayscale
from lib.pvtv2 import pvt_v2_b2
from lib.HGM import HGM
from lib.SCR import SCR

class Conv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class Up(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Sequential(
            Conv(in_channels, in_channels // 4),
            Conv(in_channels // 4, out_channels)

        )
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x1, x2):
        x = torch.cat([x2, x1], dim=1)
        out = self.conv(x)
        return self.up(out)

class Out(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(Out, self).__init__()
        self.conv1 = Conv(in_channels, in_channels // 4, kernel_size=kernel_size,stride=stride, padding=padding)

        self.conv2 = nn.Conv2d(in_channels // 4, out_channels, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x



def gauss_kernel(channels=1):
    kernel = torch.tensor([[1., 4., 6., 4., 1], [4., 16., 24., 16., 4.],
                           [6., 24., 36., 24., 6.], [4., 16., 24., 16., 4.],
                           [1., 4., 6., 4., 1.]]) / 256.
    return kernel.repeat(channels, 1, 1, 1)


def downsample(x): return x[:, :, ::2, ::2]


def conv_gauss(img, kernel):
    kernel = kernel.to(img.device)
    img = F.pad(img, (2, 2, 2, 2), mode='reflect')
    return F.conv2d(img, kernel, groups=img.shape[1])


def upsample(x, channels):
    cc = torch.cat([x, torch.zeros(x.shape[0], x.shape[1], x.shape[2], x.shape[3], device=x.device)], dim=3)
    cc = cc.view(x.shape[0], x.shape[1], x.shape[2] * 2, x.shape[3]).permute(0, 1, 3, 2)
    cc = torch.cat([cc, torch.zeros(x.shape[0], x.shape[1], x.shape[3], x.shape[2] * 2, device=x.device)], dim=3)
    cc = cc.view(x.shape[0], x.shape[1], x.shape[3] * 2, x.shape[2] * 2).permute(0, 1, 3, 2)
    return conv_gauss(cc, 4 * gauss_kernel(channels))


def make_laplace_pyramid(img, level, channels):
    current = img
    pyr = []
    for _ in range(level):
        filtered = conv_gauss(current, gauss_kernel(channels))
        down = downsample(filtered)
        up = upsample(down, channels)
        if up.shape[2] != current.shape[2] or up.shape[3] != current.shape[3]:
            up = F.interpolate(up, size=(current.shape[2], current.shape[3]))
        diff = current - up
        pyr.append(diff)
        current = down
    pyr.append(current)
    return pyr


# ===================================================================

class DynamicFrequencyModulation(nn.Module):
    def __init__(self, in_channels=1):
        super(DynamicFrequencyModulation, self).__init__()
        self.branch1 = nn.Sequential(nn.Conv2d(in_channels, in_channels, 1), nn.ReLU(inplace=True),
                                     nn.Conv2d(in_channels, in_channels, 1))
        self.branch2 = nn.Sequential(nn.Conv2d(in_channels, in_channels, 1), nn.ReLU(inplace=True),
                                     nn.Conv2d(in_channels, in_channels, 1))
        self.fusion = nn.Conv2d(in_channels * 2, in_channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.fusion(torch.cat([self.branch1(x), self.branch2(x)], dim=1)))


class HPENet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(HPENet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.conv = nn.Sequential(nn.Conv2d(1, 3, 1), nn.BatchNorm2d(3), nn.ReLU(inplace=True))

        self.backbone = pvt_v2_b2()
        path = '/home/weimengru/code/pvt_v2_b2.pth'
        try:
            save_model = torch.load(path, map_location='cpu')
            model_dict = self.backbone.state_dict()
            state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
            model_dict.update(state_dict)
            self.backbone.load_state_dict(model_dict)
        except Exception as e:
            print(f"Warning: PVT weights not loaded. Error: {e}")

        self.dms = DynamicFrequencyModulation(in_channels=1)   #DMS

        self.CBR5 = nn.Sequential(nn.Conv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(inplace=True))
        self.CBR4 = nn.Sequential(nn.Conv2d(320, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.CBR3 = nn.Sequential(nn.Conv2d(128, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.CBR2 = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        # HGM (4次)
        self.hgm5 = HGM(512)
        self.hgm4 = HGM(256)
        self.hgm3 = HGM(128)
        self.hgm2 = HGM(64)


        self.scr4 = SCR(256, deep_channels=512)
        self.scr3 = SCR(128, deep_channels=256)
        self.scr2 = SCR(64, deep_channels=128)

        self.up5 = nn.Sequential(Conv(512, 512), nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
        self.up4 = Up(768, 256)  #
        self.up3 = Up(384, 128)  #
        self.up2 = Up(192, 64)  #

        self.out5 = Out(512, n_classes)
        self.out4 = Out(256, n_classes)
        self.out3 = Out(128, n_classes)
        self.out2 = Out(64, n_classes)

    def forward(self, x):
        # 1. HPE 提取多尺度高频信息
        grayscale_img = rgb_to_grayscale(x)
        hf_pyr = make_laplace_pyramid(grayscale_img, 6, 1)
        hf2, hf3, hf4, hf5 = hf_pyr[2], hf_pyr[3], hf_pyr[4], hf_pyr[5]

        # 2. DMS 动态噪声抑制
        beta = self.dms(grayscale_img)
        beta2 = F.interpolate(beta, size=(hf2.size(2), hf2.size(3)), mode='bilinear', align_corners=True)
        beta3 = F.interpolate(beta, size=(hf3.size(2), hf3.size(3)), mode='bilinear', align_corners=True)
        beta4 = F.interpolate(beta, size=(hf4.size(2), hf4.size(3)), mode='bilinear', align_corners=True)
        beta5 = F.interpolate(beta, size=(hf5.size(2), hf5.size(3)), mode='bilinear', align_corners=True)

        hf2, hf3, hf4, hf5 = hf2 * beta2, hf3 * beta3, hf4 * beta4, hf5 * beta5


        if x.size()[1] == 1:
            x = self.conv(x)

        # 3. Backbone 提取主干特征
        f2, f3, f4, f5 = self.backbone(x)

        f_c5 = self.CBR5(f5)
        f_c4 = self.CBR4(f4)
        f_c3 = self.CBR3(f3)
        f_c2 = self.CBR2(f2)

        # ---------------- Level 5 (1/32) ----------------
        # HGM 没有上一层预测，传 None
        hgm5_out = self.hgm5(hf5, f_c5, pred=None)
        f_d5 = self.up5(hgm5_out)
        out5 = self.out5(f_d5)

        # ---------------- Level 4 (1/16) ----------------
        hgm4_out = self.hgm4(hf4, f_c4, out5)
        mcfr_out4 = self.mcfr4(hgm4_out, hgm5_out)
        f_d4 = self.up4(f_d5, mcfr_out4)
        out4 = self.out4(f_d4)

        # ---------------- Level 3 (1/8) ----------------
        hgm3_out = self.hgm3(hf3, f_c3, out4)
        mcfr_out3 = self.mcfr3(hgm3_out, mcfr_out4)
        f_d3 = self.up3(f_d4, mcfr_out3)
        out3 = self.out3(f_d3)

        # ---------------- Level 2 (1/4) ----------------
        hgm2_out = self.hgm2(hf2, f_c2, out3)
        mcfr_out2 = self.mcfr2(hgm2_out, mcfr_out3)
        f_d2 = self.up2(f_d3, mcfr_out2)
        out2 = self.out2(f_d2)

        return out2, out3, out4, out5


class HPEModel(nn.Module):
    def __init__(self, n_channels=3, n_classes=1):
        super(HPEModel, self).__init__()
        self.net = HPENet(n_channels, n_classes)

    def forward(self, images):
        return self.net(images)