import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

# Helper blocks
class CAB(nn.Module):
    def __init__(self, channels, r=16):
        super().__init__()
        channels = int(max(1, channels))
        r = int(max(1, r))
        hidden = max(1, channels // r)
        self.fc1 = nn.Conv2d(channels, hidden, 1, bias=True)
        self.fc2 = nn.Conv2d(hidden, channels, 1, bias=True)
    def forward(self, x):
        s = F.adaptive_avg_pool2d(x, 1)
        s = F.silu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s

class SAB(nn.Module):
    def __init__(self, in_ch, reduce=4, k=7):
        super().__init__()
        in_ch = int(max(1, in_ch))
        mid = max(8, in_ch // reduce)
        self.reduce = nn.Conv2d(in_ch, mid, 1, bias=False)
        self.conv = nn.Conv2d(mid, 1, k, padding=k//2, bias=False)
    def forward(self, x):
        m = self.reduce(x)
        m = self.conv(m)
        m = torch.sigmoid(m)
        return x * m

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, d=1, bn=True, act=True):
        super().__init__()
        in_ch = int(max(1, in_ch))
        out_ch = int(max(1, out_ch))
        self.dw = nn.Conv2d(in_ch, in_ch, k, stride=s, padding=p, dilation=d, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch) if bn else nn.Identity()
        self.act = nn.SiLU() if act else nn.Identity()
    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class DilatedDSCBundle(nn.Module):
    def __init__(self, ch, out_ch=None, dilations=(1,2,3)):
        super().__init__()
        ch = int(max(1, ch))
        out_ch = int(out_ch or ch)
        self.branches = nn.ModuleList([DepthwiseSeparableConv(ch, out_ch, k=3, p=d, d=d) for d in dilations])
        self.mix = nn.Conv2d(len(dilations) * out_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU()
    def forward(self, x):
        feats = [b(x) for b in self.branches]
        x = torch.cat(feats, dim=1)
        x = self.mix(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class StageEnhancer(nn.Module):
    def __init__(self, ch, out_ch=None):
        super().__init__()
        ch = int(max(1, ch))
        out_ch = int(out_ch or ch)
        self.conv1 = DepthwiseSeparableConv(ch, out_ch)
        self.dilated = DilatedDSCBundle(out_ch, out_ch)
        self.cab = CAB(out_ch)
        self.sab = SAB(out_ch)
        self.proj = nn.Identity() if ch == out_ch else nn.Conv2d(ch, out_ch, 1, bias=False)
    def forward(self, x):
        idt = self.proj(x)
        x = self.conv1(x)
        x = self.dilated(x)
        x = self.cab(x)
        x = self.sab(x)
        return x + idt

class LateralReduce(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        in_ch = int(max(1, in_ch))
        out_ch = int(max(1, out_ch))
        self.conv = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU()
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class FuseMix(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        in_ch = int(max(1, in_ch))
        out_ch = int(max(1, out_ch))
        self.mix = DepthwiseSeparableConv(in_ch, out_ch)
    def forward(self, x):
        return self.mix(x)

class MultiLevelMiTB3Fusion(nn.Module):
    def __init__(self, fusion_ch=320, pretrained=False, debug=False):
        super().__init__()
        self.debug = debug
        self.backbone = smp.encoders.get_encoder(
            "mit_b3",
            in_channels=3,
            depth=5,
            weights="imagenet" if pretrained else None
        )
        enc_chs_all = list(self.backbone.out_channels)
        enc_chs_nonzero = [int(c) for c in enc_chs_all if (c is not None and int(c) > 0)]
        picked = enc_chs_nonzero[-4:]
        c2, c3, c4, c5 = picked
        self._enc_chs = (c2, c3, c4, c5)
        self.enh2 = StageEnhancer(c2, c2)
        self.enh3 = StageEnhancer(c3, c3)
        self.enh4 = StageEnhancer(c4, c4)
        self.enh5 = StageEnhancer(c5, c5)
        self.lat2 = LateralReduce(c2, fusion_ch // 4)
        self.lat3 = LateralReduce(c3, fusion_ch // 2)
        self.lat4 = LateralReduce(c4, fusion_ch)
        self.lat5 = LateralReduce(c5, fusion_ch)
        self.mix4 = FuseMix(fusion_ch + fusion_ch, fusion_ch)
        self.mix3 = FuseMix(fusion_ch // 2 + fusion_ch, fusion_ch // 2)
        self.mix2 = FuseMix(fusion_ch // 4 + fusion_ch // 2, fusion_ch // 4)
        self.ref4 = StageEnhancer(fusion_ch, fusion_ch)
        self.ref3 = StageEnhancer(fusion_ch // 2, fusion_ch // 2)
        self.ref2 = StageEnhancer(fusion_ch // 4, fusion_ch // 4)
        self.unify4 = nn.Conv2d(fusion_ch, fusion_ch // 2, 1, bias=False)
        self.unify3 = nn.Conv2d(fusion_ch // 2, fusion_ch // 2, 1, bias=False)
        self.unify2 = nn.Conv2d(fusion_ch // 4, fusion_ch // 2, 1, bias=False)
        cat_ch = 3 * (fusion_ch // 2)
        self.seg_head = nn.Sequential(
            DepthwiseSeparableConv(cat_ch, fusion_ch // 2),
            nn.Conv2d(fusion_ch // 2, 1, kernel_size=1)
        )

    def forward(self, x):
        feats = self.backbone(x)
        valid_feats = [f for f in feats if f is not None and f.shape[1] > 0]
        c2_feat, c3_feat, c4_feat, c5_feat = valid_feats[-4], valid_feats[-3], valid_feats[-2], valid_feats[-1]
        c2 = self.enh2(c2_feat)
        c3 = self.enh3(c3_feat)
        c4 = self.enh4(c4_feat)
        c5 = self.enh5(c5_feat)
        l2 = self.lat2(c2)
        l3 = self.lat3(c3)
        l4 = self.lat4(c4)
        l5 = self.lat5(c5)
        p5 = l5
        up5 = F.interpolate(p5, size=l4.shape[-2:], mode='bilinear', align_corners=False)
        p4 = self.mix4(torch.cat([l4, up5], dim=1))
        up4 = F.interpolate(p4, size=l3.shape[-2:], mode='bilinear', align_corners=False)
        p3 = self.mix3(torch.cat([l3, up4], dim=1))
        up3 = F.interpolate(p3, size=l2.shape[-2:], mode='bilinear', align_corners=False)
        p2 = self.mix2(torch.cat([l2, up3], dim=1))
        p4 = self.ref4(p4)
        p3 = self.ref3(p3)
        p2 = self.ref2(p2)
        u4 = self.unify4(p4)
        u3 = self.unify3(p3)
        u2 = self.unify2(p2)
        tgt_hw = u4.shape[-2:]
        u3r = F.interpolate(u3, size=tgt_hw, mode='bilinear', align_corners=False)
        u2r = F.interpolate(u2, size=tgt_hw, mode='bilinear', align_corners=False)
        cat = torch.cat([u4, u3r, u2r], dim=1)
        seg = self.seg_head(cat)
        seg = F.interpolate(seg, size=x.shape[2:], mode='bilinear', align_corners=False)
        return seg
