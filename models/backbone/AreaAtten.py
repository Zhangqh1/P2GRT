import torch
import torch.nn as nn
from flash_attn.flash_attn_interface import flash_attn_func

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k] # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k] # auto-pad
    return p

class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU() # default activation

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
    
class AAttn(nn.Module):
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

    def __init__(self, dim, num_heads, area, flatten_mode):
        """Initializes the area-attention module, a simple yet efficient attention module for YOLO."""
        super().__init__()
        self.area = area
        self.flatten_mode = flatten_mode

        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        all_head_dim = head_dim * self.num_heads

        self.qk = Conv(dim, all_head_dim * 2, 1, act=False)
        self.v = Conv(dim, all_head_dim, 1, act=False)
        self.proj = Conv(all_head_dim, dim, 1, act=False)

        self.pe = Conv(all_head_dim, dim, 5, 1, 2, g=dim, act=False)


    def forward(self, x):
        """Processes the input tensor 'x' through the area-attention"""
        B, C, H, W = x.shape
        N = H * W

        if self.flatten_mode == 'C':
            qk = self.qk(x).permute(0,1,3,2).flatten(2).transpose(1, 2)
            v = self.v(x).permute(0,1,3,2)
            pp = self.pe(v).permute(0,1,3,2)
            v = v.flatten(2).transpose(1, 2)

        if self.flatten_mode == 'R':
            qk = self.qk(x).flatten(2).transpose(1, 2)
            v = self.v(x)
            pp = self.pe(v)
            v = v.flatten(2).transpose(1, 2)

        if self.area > 1:
            qk = qk.reshape(B * self.area, N // self.area, C * 2)
            v = v.reshape(B * self.area, N // self.area, C)
            B, N, _ = qk.shape
        q, k = qk.split([C, C], dim=2)
        q = q.view(B, N, self.num_heads, self.head_dim)
        k = k.view(B, N, self.num_heads, self.head_dim)
        v = v.view(B, N, self.num_heads, self.head_dim)

        x = flash_attn_func(
            q.contiguous().half(),
            k.contiguous().half(),
            v.contiguous().half()
        ).to(q.dtype)

        if self.area > 1:
            x = x.reshape(B // self.area, N * self.area, C)
            B, N, _ = x.shape

        if self.flatten_mode == 'C':
            x = x.reshape(B, W, H, C).permute(0, 3, 1, 2).permute(0, 1, 3, 2)
        else:
            x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)

        return self.proj(x + pp)


class ABlock(nn.Module):
    """
    ABlock实现具有有效特征提取的区域注意力块.这个类封装了应用多头注意力的功能,并将特征图划分为多个区域以及前馈神经网络层.

    属性:
    dim (int):隐藏通道数;
    num_heads (int):注意力机制被划分到的头的数量;
    mlp_ratio(浮动,可选):MLP膨胀率(或MLP隐藏维度比).默认值为1.2;
    area (int,可选):特征图划分的区域数量.默认值为1.
    方法:
    前向传播:通过块执行前向传递,应用区域注意力和前馈层.

    Examples:
        Create a ABlock and perform a forward pass
        >>> model = ABlock(dim=64, num_heads=2, mlp_ratio=1.2, area=4)
        >>> x = torch.randn(2, 64, 128, 128)
        >>> output = model(x)
        >>> print(output.shape)

    Notes: 
        建议 dim//num_heads 的值是32或64的倍数.
    """

    def __init__(self, dim, num_heads, mlp_ratio, area, flatten_mode):
        """Initializes the ABlock with area-attention and feed-forward layers for faster feature extraction."""
        super().__init__()

        self.attn = AAttn(dim, num_heads=num_heads, area=area, flatten_mode=flatten_mode)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(Conv(dim, mlp_hidden_dim, 1), Conv(mlp_hidden_dim, dim, 1, act=False))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Initialize weights using a truncated normal distribution."""
        if isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """Executes a forward pass through ABlock, applying area-attention and feed-forward layers to the input tensor."""
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x

class MyABlock(nn.Module):
    def __init__(self, dim, num_heads, area=8, mlp_ratio=1.2):
        super().__init__()
        self.ABlock1 = ABlock(dim, num_heads, mlp_ratio, area, flatten_mode='R')
        self.ABlock2 = ABlock(dim, num_heads, mlp_ratio, area, flatten_mode='C')

    def forward(self, x):
        x1 = self.ABlock1(x)
        x2 = self.ABlock2(x1)
        return x2
    
if __name__ == "__main__":
    # 模块参数
    batch_size = 1 # 批大小
    channels = 32 # 输入特征通道数
    height = 256 # 图像高度
    width = 256 # 图像宽度

    model = ABlock(dim=channels, num_heads=2, mlp_ratio=1.2, area=4)
    print(model)

    # 生成随机输入张量 (batch_size, channels, height, width)
    x = torch.randn(batch_size, channels, height, width)

    # 打印输入张量的形状
    print("Input shape:", x.shape)

    # 前向传播计算输出
    output = model(x)

    # 打印输出张量的形状
    print("Output shape:", output.shape)