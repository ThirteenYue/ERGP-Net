from typing import Union, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize._tstutils import f1


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        padding = kernel_size // 2 if dilation == 1 else dilation
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


# 下采样
class DownConvBNReLU(ConvBNReLU):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dilation: int = 1, flag: bool = True):
        super().__init__(in_ch, out_ch, kernel_size, dilation)
        self.down_flag = flag

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.down_flag:
            x = F.max_pool2d(x, kernel_size=2, stride=2, ceil_mode=True)

        return self.relu(self.bn(self.conv(x)))


# 全局关注模块
class GP(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3):
        super(GP, self).__init__()

        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_ch, 1, kernel_size=1, padding=0),  # b,1,16,16
            nn.BatchNorm2d(1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, stride=2))  # b,1,8,8

        self.GAVG = nn.Sequential(
            nn.AdaptiveAvgPool2d(8),  # b,c,1,1
            nn.Softmax(), )

        self.conv3x3 = nn.Conv2d(in_ch, out_ch, kernel_size, padding=1, stride=1, bias=False)

    def forward(self, x):
        # x torch.Size([8, 768, 16, 16])

        fm = self.GAVG(x)  # b,c=768,1,1

        fg = self.conv1x1(x)  # torch.Size([8, 1, 8, 8])

        ou = self.conv3x3(x)  # torch.Size([b, c=768, 16, 16])

        #####################################
        # ！！！没有自动广播！！！
        # fm 自动广播 (b,768,1,1) -> (b,768,8,8)
        # fg 自动广播 (b,1,8,8) -> (b,768,8,8)
        # torch.matmul(fg,fm) -> (b,768,8,8)
        gm = torch.matmul(fg, fm)
        # gm = torch.mm(fg, fm)
        #####################################

        # gm 2倍上采样 -> (b,768,16,16)
        Up = torch.nn.UpsamplingNearest2d(scale_factor=2)
        gm1 = Up(gm)

        # gm + ou
        #
        return ou + gm1


# 跳跃空间注意力模块
class SA(nn.Module):
    def __init__(self, in_ch):
        super(SA, self).__init__()
        self.SA_w = nn.AdaptiveAvgPool2d((None, 1))
        self.SA_h = nn.AdaptiveAvgPool2d((None, 1))
        self.SA_c = nn.AdaptiveAvgPool2d((None, in_ch))

        self.conv1 = nn.Conv2d(1, in_ch, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.size()
        Reshape_ch = self.SA_w(x).permute(0, 3, 1, 2).reshape(B, 1, C * H)
        Tch = Reshape_ch.permute(0, 2, 1)
        Reshape_cw = self.SA_h(x.permute(0, 1, 3, 2)).permute(0, 3, 1, 2).reshape(B, 1, C * W)
        MM = F.softmax(torch.bmm(Tch, Reshape_cw),dim=1)  # (B, C*H, C*W)
        M1 = MM.reshape(B, C * C, H, W)
        M2 = self.SA_c(M1.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        return M2 + x


# 串联通道注意力
class CA(nn.Module):
    def __init__(self, in_ch):
        super(CA, self).__init__()
        self.c1 = nn.AdaptiveAvgPool2d(1)

    def _slow_forward(self, x):
        B, C, H, W = input.size()
        c1 = x.reshape(B, C, H * W)
        Tc1 = c1.permute(0, 2, 1)  # (B, H*W, C)
        Sf = F.softmax(torch.mm(c1, Tc1))  # (B, C, C)

        Si = F.sigmoid(self.c1(x))

        ca = torch.mm(torch.mm(Si, c1), Sf)

        return ca


# 嵌入残差卷积块
class ERB(nn.Module):
    def __init__(self, in_ch: int, mid_ch: int, out_ch: int, ):
        super().__init__()
        self.Upear = nn.UpsamplingNearest2d(scale_factor=2)
        self.ConvBNReLU = ConvBNReLU
        self.DownConvBNReLU = DownConvBNReLU(mid_ch // 2, mid_ch // 2)
        self.conv_in = ConvBNReLU(in_ch, mid_ch)
        self.conv1 = nn.Conv2d(mid_ch * 2, mid_ch // 2, kernel_size=1, stride=1, padding=0)

        self.f1 = self.ConvBNReLU(mid_ch, mid_ch // 2)

        self.f2 = self.ConvBNReLU(mid_ch // 2, mid_ch // 2)
        self.f3 = self.ConvBNReLU(mid_ch // 2, mid_ch // 2)
        self.f4 = self.ConvBNReLU(mid_ch // 2, mid_ch // 2)
        self.f5 = self.ConvBNReLU(mid_ch // 2, mid_ch // 2)
        self.f6 = self.ConvBNReLU(mid_ch // 2, mid_ch // 2)

        self.t4 = self.ConvBNReLU(mid_ch, mid_ch // 2)
        self.t3 = self.ConvBNReLU(mid_ch, mid_ch // 2)
        self.t2 = self.ConvBNReLU(mid_ch, mid_ch // 2)
        self.t1 = self.ConvBNReLU(mid_ch, mid_ch // 2)
        self.t0 = self.ConvBNReLU(mid_ch, out_ch // 2)

    def forward(self, x):
        x0 = self.conv_in(x)
        x1 = self.f1(x0)
        x10 = self.DownConvBNReLU(x1)
        x2 = self.f2(x10)
        x20 = self.DownConvBNReLU(x2)
        x21 = self.DownConvBNReLU(x20)  # d2
        x22 = self.DownConvBNReLU(x21)
        x3 = self.f3(x20)
        x30 = self.DownConvBNReLU(x3)  # d4
        x4 = self.f4(x30)
        x40 = self.DownConvBNReLU(x4)
        x5 = self.f5(x40)
        x6 = self.f6(x5)

        # 高维特征融合
        Hf = self.conv1(torch.cat([x22, x40, x5, x6], dim=1))

        u4 = self.Upear(Hf)
        u2 = self.Upear(u4)

        y4 = self.t4(torch.cat([Hf, x6], dim=1))
        y40 = self.Upear(y4)
        y3 = self.t3(torch.cat([u4, y40], dim=1))
        y30 = self.Upear(y3)
        y2 = self.t2(torch.cat([u2, y30], dim=1))
        y20 = self.Upear(y2)
        y21 = torch.cat([y20, x2], dim=1)
        y1 = self.t1(y21)
        y10 = self.Upear(y1)
        y11 = torch.cat([y10, x1], dim=1)
        out = self.t0(y11)

        return torch.cat([x, out], dim=1)


# 底层卷积
class E5D4(nn.Module):
    def __init__(self, in_ch: int, mid_ch: int, out_ch: int):
        super().__init__()
        self.conv_in = ConvBNReLU(in_ch, mid_ch)
        self.f1 = ConvBNReLU(mid_ch, mid_ch, dilation=2)
        self.f2 = ConvBNReLU(mid_ch, mid_ch, dilation=4)

        self.t2 = ConvBNReLU(mid_ch, mid_ch, dilation=4)
        self.t1 = ConvBNReLU(mid_ch * 2, mid_ch, dilation=2)
        self.out = ConvBNReLU(mid_ch * 2, out_ch)

    def forward(self, x):
        x1 = self.conv_in(x)
        x2 = self.f1(x1)
        x3 = self.f2(x2)

        y3 = self.t2(x3)

        y21 = torch.cat((x2, y3), dim=1)
        y2 = self.t1(y21)
        y11 = torch.cat((x1, y2), dim=1)
        y1 = self.out(y11)
        out = y1

        return out


class Conv1x1(nn.Module):
    def __init__(self, in_chs, out_chs):
        super().__init__()
        self.conv = nn.Conv2d(in_chs, out_chs, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        return self.conv(x)


class ERGP(nn.Module):
    def __init__(self, in_ch, n_class, bilinear):
        super(ERGP, self).__init__()
        self.in_ch = in_ch
        self.n_class = n_class
        self.bilinear = bilinear
        self.Upear = nn.UpsamplingNearest2d(scale_factor=2)
        self.Up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.Do = nn.MaxPool2d(2, stride=2)

        self.in_conv = ConvBNReLU(in_ch, 32)
        self.out_conv = ConvBNReLU(3, n_class)
        self.E1 = ERB(32, 64, 64)
        self.E2 = ERB(64, 64, 128)
        self.E3 = ERB(128, 64, 256)
        self.E4 = ERB(256, 64, 512)
        self.E5D4_1 = E5D4(512, 64, 512)  #####

        self.SA1 = SA(64)
        self.SA2 = SA(128)
        self.SA3 = SA(256)

        self.gpm = GP(1024, 1024)
        self.gpm = GP(512, 512)

        self.E5D4_2 = E5D4(1024, 64, 512)  #####
        self.D3 = ERB(512, 64, 256)
        self.D2 = ERB(256, 64, 128)
        self.D1 = ERB(128, 64, 64)

        self.CA3 = CA(512)
        self.CA2 = CA(256)
        self.CA1 = CA(128)

        # 1x1降dimention
        self.up3to512 = Conv1x1(1280, 512)
        self.up2to256 = Conv1x1(1152, 256)
        self.up1to128 = Conv1x1(1088, 128)

        # predictor head
        self.up3_out = Conv1x1(640, 1)
        self.up2_out = Conv1x1(320, 1)
        self.up1_out = Conv1x1(160, 1)

    def forward(self, x):
        xc1 = self.in_conv(x)
        xe1 = self.E1(xc1)
        x1 = self.Do(xe1)
        sa1 = self.SA1(x1)
        xe2 = self.E2(x1)
        x2 = self.Do(xe2)
        sa2 = self.SA2(x2)
        xe3 = self.E3(x2)
        x3 = self.Do(xe3)
        sa3 = self.SA3(x3)
        xe4 = self.E4(x3)
        x4 = self.Do(xe4)
        xe5 = self.E5D4_1(x4)

        gp = self.gpm(xe5)
        gp3 = torch.nn.functional.interpolate(gp, scale_factor=2, mode='bilinear', align_corners=None)
        gp2 = torch.nn.functional.interpolate(gp, scale_factor=4, mode='bilinear', align_corners=None)
        gp1 = torch.nn.functional.interpolate(gp, scale_factor=8, mode='bilinear', align_corners=None)

        d0 = torch.cat((gp, xe5), dim=1)
        d = self.E5D4_2(d0)

        d = F.interpolate(d, scale_factor=2, mode='bilinear', align_corners=True)
        up3 = torch.cat((sa3, gp3, d), dim=1)
        up3 = self.up3to512(up3)
        up3 = self.D3(up3)  # b,640, 32, 32

        d = F.interpolate(d, scale_factor=2, mode='bilinear', align_corners=True)
        up2 = torch.cat((sa2, gp2, d), dim=1)
        up2 = self.up2to256(up2)
        up2 = self.D2(up2)

        d = F.interpolate(d, scale_factor=2, mode='bilinear', align_corners=True)
        up1 = torch.cat((sa1, gp1, d), dim=1)
        up1 = self.up1to128(up1)
        up1 = self.D1(up1)

        out_up3 = F.interpolate(self.up3_out(up3),scale_factor=8)
        out_up2 = F.interpolate(self.up2_out(up2),scale_factor=4)
        out_up1 = F.interpolate(self.up1_out(up1),scale_factor=2)

        output = self.out_conv(torch.cat((out_up3, out_up2, out_up1), dim=1))

        return output


# model = ERGP(in_ch=3, n_class=2, bilinear=True, )
# image = torch.randn(1, 3, 256, 256)
# model(image)

