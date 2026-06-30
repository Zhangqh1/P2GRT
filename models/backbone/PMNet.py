import torch
import torch.nn as nn
import torch.nn.functional as F


class GradientCalculator(nn.Module):
    """计算图像梯度，模拟|∇u|，支持多通道输入"""

    def __init__(self, in_channels):
        super(GradientCalculator, self).__init__()
        self.in_channels = in_channels

        # 定义水平和垂直方向的Sobel算子，为每个输入通道创建一组卷积
        self.conv_x = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False)
        self.conv_y = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False)

        # Sobel算子参数初始化
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)

        # 为每个输入通道复制相同的Sobel核
        sobel_x = sobel_x.repeat(in_channels, 1, 1, 1)
        sobel_y = sobel_y.repeat(in_channels, 1, 1, 1)

        self.conv_x.weight = nn.Parameter(sobel_x)
        self.conv_y.weight = nn.Parameter(sobel_y)

        # 权重参数不参与训练
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        # 计算x和y方向梯度 (batch_size, in_channels, H, W)
        grad_x = self.conv_x(x)
        grad_y = self.conv_y(x)

        # 计算每个通道的梯度模长，然后在通道维度取平均得到单通道梯度强度
        grad_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)  # 加小值防止除零
        grad_mag = torch.mean(grad_mag, dim=1, keepdim=True)  # (batch_size, 1, H, W)

        return grad_mag


class DiffusionCoefficient(nn.Module):
    """学习扩散系数函数c(·)，支持多通道处理"""

    def __init__(self, channels = 16, num_directions = 4):
        super(DiffusionCoefficient, self).__init__()
        # 通过卷积网络学习扩散系数，输入为1通道的梯度模长
        self.conv1 = nn.Conv2d(1, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(channels, num_directions, kernel_size=3, padding=1)
        self.relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()  # 确保输出在0-1之间，模拟扩散系数特性

    def forward(self, grad_mag):
        # 学习梯度到扩散系数的映射
        x = self.relu(self.conv1(grad_mag))
        x = self.relu(self.conv2(x))
        # 输出扩散系数c，值越大表示扩散越强
        c = 2 * self.sigmoid(self.conv3(x))
        return c


class AdaptiveDiffusionBlock(nn.Module):
    """自适应扩散块，模拟PM算法的扩散过程，支持多通道"""

    def __init__(self, in_channels, channels=16):
        super(AdaptiveDiffusionBlock, self).__init__()
        self.in_channels = in_channels

        # 定义不同方向的扩散卷积核（模拟各向异性扩散）
        # 使用分组卷积确保每个通道独立处理
        self.diffusion_kernels = nn.ModuleList([
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False)
            for _ in range(4)
        ])

        # 初始化卷积核为四个方向的扩散算子
        base_kernels = [
            torch.tensor([[0, 1, 0], [0, -1, 0], [0, 0, 0]], dtype=torch.float32),  # 上
            torch.tensor([[0, 0, 0], [0, -1, 0], [0, 1, 0]], dtype=torch.float32),  # 下
            torch.tensor([[0, 0, 0], [1, -1, 0], [0, 0, 0]], dtype=torch.float32),  # 左
            torch.tensor([[0, 0, 0], [0, -1, 1], [0, 0, 0]], dtype=torch.float32)  # 右
        ]

        # 为每个输入通道复制相同的核
        for i, kernel in enumerate(base_kernels):
            kernel = kernel.repeat(in_channels, 1, 1, 1)
            self.diffusion_kernels[i].weight = nn.Parameter(kernel)
            # self.diffusion_kernels[i].weight.requires_grad = False

        # 梯度计算器和扩散系数学习器
        self.gradient_calculator = GradientCalculator(in_channels)
        self.diffusion_coeff = DiffusionCoefficient(channels, num_directions=4)

        # 注意力机制用于融合不同方向的扩散结果
        self.fuse_conv = nn.Sequential(nn.Conv2d(in_channels * 4, in_channels, kernel_size=1),
                                       nn.LeakyReLU())

    def forward(self, x):
        # x: (batch_size, in_channels, H, W)

        # 计算梯度模长 (batch_size, 1, H, W)
        grad_mag = self.gradient_calculator(x)

        # 计算扩散系数 (batch_size, 1, H, W)
        c = self.diffusion_coeff(grad_mag)

        # 不同方向的扩散
        diffusions = []
        for i,kernel in enumerate(self.diffusion_kernels):
            d = kernel(x)
            d = d * c[:, i:i+1, :, :]
            diffusions.append(d)

        # 拼接四个方向结果
        fused = torch.cat(diffusions, dim=1)  # (B, 4*C, H, W)

        # 卷积融合
        fused = self.fuse_conv(fused)  # (B, C, H, W)

        # 残差连接
        return x + fused


class PMNet(nn.Module):
    """完整的PM算法模拟网络，支持指定输入输出通道数"""

    def __init__(self, in_channels, out_channels, num_blocks=2, channels=16):
        super(PMNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.channel_tune = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        # 多个扩散块，模拟多步演化
        self.diffusion_blocks = nn.Sequential(
            *[AdaptiveDiffusionBlock(out_channels, channels) for _ in range(num_blocks)]
        )

        # 输出卷积，将内部通道映射到目标输出通道
        # self.activation = nn.ReLU()
        # self.output_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        # x: (batch_size, in_channels, H, W)

        # 多步扩散过程
        x_diffused = self.channel_tune(x)
        x_diffused = self.diffusion_blocks(x_diffused)

        return x_diffused


