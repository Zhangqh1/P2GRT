import torch
import torch.nn as nn
from flash_attn.flash_attn_interface import flash_attn_func
from torch.ao.nn.quantized import ReLU6
from .StefanEnhancement import StefanThermalAtten

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Apply convolution and activation without batch normalization."""
        return self.act(self.conv(x))


class CrossAttn(nn.Module):
    """
    Area-attention模块, 需要安装flash attention.

    Attributes:
        dim (int):隐藏通道数;
        num_heads (int):注意力机制被划分到的头的数量;
        area (int, 可选):特征图划分的区域数量.默认值为1.

    Methods:
        前向传播: 对输入张量进行前向处理，并在执行区域注意力机制后输出一个张量.

    Examples:
        >>> import torch
        >>> from ultralytics.nn.modules import AAttn
        >>> model = AAttn(dim=64, num_heads=2, area=4)
        >>> x = torch.randn(2, 64, 128, 128)
        >>> output = model(x)
        >>> print(output.shape)

    Notes:
        recommend that dim//num_heads be a multiple of 32 or 64.

    """

    def __init__(self, dim, num_heads, area=4):
        """Initializes the area-attention module, a simple yet efficient attention module for YOLO."""
        super().__init__()
        self.area = area

        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        all_head_dim = head_dim * self.num_heads

        self.x_q = Conv(dim, all_head_dim, 1, act=False)
        self.x_k = Conv(dim, all_head_dim, 1, act=False)
        self.x_v = Conv(dim, all_head_dim, 1, act=False)
        self.x_proj = Conv(all_head_dim, dim, 1, act=False)

        self.y_q = Conv(dim, all_head_dim, 1, act=False)
        self.y_k = Conv(dim, all_head_dim, 1, act=False)
        self.y_v = Conv(dim, all_head_dim, 1, act=False)
        self.y_proj = Conv(all_head_dim, dim, 1, act=False)

        self.x_pe = Conv(all_head_dim, dim, 5, 1, 2, g=dim, act=False)
        self.y_pe = Conv(all_head_dim, dim, 5, 1, 2, g=dim, act=False)

        self.conv = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU()
        )

        self.ir_enhancement = StefanThermalAtten(all_head_dim)

    def forward(self, x, y):
        # x is IR, y is VI
        """Processes the input tensor 'x' through the area-attention"""
        B, C, H, W = x.shape
        N = H * W

        x_k = self.x_k(x).flatten(2).transpose(1, 2)
        x_v = self.x_v(x)
        x_v = self.ir_enhancement(x_v)
        x_pp = self.x_pe(x_v)
        x_v = x_v.flatten(2).transpose(1, 2)

        y_q = self.y_q(y).flatten(2).transpose(1, 2)

        if self.area > 1:

            x_k = x_k.reshape(B * self.area, N // self.area, C)
            x_v = x_v.reshape(B * self.area, N // self.area, C)
            y_q = y_q.reshape(B * self.area, N // self.area, C)
            B, N, _ = y_q.shape


        x_k = x_k.view(B, N, self.num_heads, self.head_dim)
        x_v = x_v.view(B, N, self.num_heads, self.head_dim)
        y_q = y_q.view(B, N, self.num_heads, self.head_dim)

        y = flash_attn_func(
            y_q.contiguous().half(),
            x_k.contiguous().half(),
            x_v.contiguous().half()
        ).to(y_q.dtype)

        y = x_v - y

        if self.area > 1:
            y = y.reshape(B // self.area, N * self.area, C)
            B, N, _ = y.shape

        y = y.reshape(B, H, W, C).permute(0, 3, 1, 2)

        y = self.y_proj(y + x_pp)

        return y
