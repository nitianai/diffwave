import torch
import torch.nn as nn
import torch.nn.functional as F

# 自定义的LayerScale层
class _LayerScale(nn.Module):
    def __init__(self, init_value=1e-5, num_channels=96):
        super(_LayerScale, self).__init__()
        self.scale = nn.Parameter(init_value * torch.ones((1, num_channels, 1)))

    def forward(self, x):
        return x * self.scale

# 自定义的DConv层
class _DConv(nn.Module):
    def __init__(self, out_channels):
        super(_DConv, self).__init__()
        inter_channels = out_channels // 4
        final_channels = out_channels * 2
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(out_channels, inter_channels, kernel_size=3, stride=1, padding=1),
                nn.GroupNorm(1, inter_channels, eps=1e-05, affine=True),
                nn.GELU(approximate='none'),
                nn.Conv1d(inter_channels, final_channels, kernel_size=1, stride=1),
                nn.GroupNorm(1, final_channels, eps=1e-05, affine=True),
                nn.GLU(dim=1),
                _LayerScale(num_channels=final_channels // 2)
            ),
            nn.Sequential(
                nn.Conv1d(out_channels, inter_channels, kernel_size=3, stride=1, padding=2, dilation=2),
                nn.GroupNorm(1, inter_channels, eps=1e-05, affine=True),
                nn.GELU(approximate='none'),
                nn.Conv1d(inter_channels, final_channels, kernel_size=1, stride=1),
                nn.GroupNorm(1, final_channels, eps=1e-05, affine=True),
                nn.GLU(dim=1),
                _LayerScale(num_channels=final_channels // 2)
            )
        ])

    def forward(self, x):
        for layer in self.layers:
            x = x + layer(x)
        return x

# 主模型结构HEncLayer
class HEncLayer(nn.Module):
    def __init__(self, in_channels=2, out_channels=48, Activer=nn.Identity()):
        super().__init__()
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(8, 8), stride=(2, 2), padding=(3, 3))
        self.norm1 = Activer
        self.rewrite = nn.Conv2d(out_channels, out_channels * 2, kernel_size=(1, 1), stride=(1, 1))
        self.norm2 = Activer
        self.dconv = _DConv(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm1(x)
        x = self.rewrite(x)
        x = self.norm2(x)
        x = F.glu(x, dim=1)
        B, C, Fq, T = x.shape
        # # 转换数据形状以匹配Conv1d的输入要求
        # x = x.permute(0, 2, 1, 3).reshape(-1, C, T)
        # x = self.dconv(x)
        # x = x.view(B, Fq, C, T).permute(0, 2, 1, 3)

        return x

class StyleEncLayer(nn.Module):
    def __init__(self, in_channels=2, out_channels=48, Activer=nn.Identity()):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(8, 8), stride=(2, 2), padding=(3, 3))
        self.norm1 = Activer
        self.rewrite = nn.Conv2d(out_channels, out_channels * 2, kernel_size=(1, 1), stride=(1, 1))
        self.norm2 = Activer

    def forward(self, x):
        x = self.conv(x)
        x = self.norm1(x)
        x = self.rewrite(x)
        x = self.norm2(x)
        x = F.glu(x, dim=1)
        B, C, Fq, T = x.shape

        return x

class HDecLayer(nn.Module):
    def __init__(self, in_channels, out_channels, Activer=nn.Identity()):
        super().__init__()
        self.conv_t = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=(8, 8), stride=(2, 2), padding=(3, 3))
        self.norm2 = Activer
        self.rewrite = nn.Conv2d(in_channels, in_channels * 2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.norm1 = Activer

    def forward(self, x):
        y = self.rewrite(x)
        y = self.norm1(y)
        y = F.glu(y, dim=1)
        z = self.conv_t(y)
        z = self.norm2(z)
        return z

if __name__ == '__main__':

    # 实例化模型并打印
    model = StyleEncLayer(in_channels=2, out_channels=48)
    x = torch.randn(10, 2, 320, 100)
    y = model(x)

    print(model)
