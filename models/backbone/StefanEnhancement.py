import torch
import torch.nn as nn

class LearnablePowerScale(nn.Module):
    """可学习幂次方缩放层（改进版）"""

    def __init__(self, init_exponent=4.0, init_scale=1.0):
        super().__init__()
        self.log_exponent = nn.Parameter(torch.log(torch.tensor(init_exponent)))  # 对数参数化
        self.scale = nn.Parameter(torch.tensor(init_scale))

    def forward(self, x):
        exponent = torch.exp(self.log_exponent)  # 保证指数为正
        return x ** exponent * self.scale

class StefanThermalAtten(nn.Module):
    """增强版热力学注意力机制（优化版本）"""

    def __init__(self, in_channels, T_scale=1.0):
        super().__init__()

        # 温度估计网络参数
        self.temp_estimator = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(in_channels // 2, in_channels // 4, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(in_channels // 4, 1, 1),
            nn.Sigmoid(),
            LearnablePowerScale(init_exponent=4.0, init_scale=T_scale)
        )

        # 发射率网络参数（动态分组）
        self.emis_net = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1),
            nn.GroupNorm(4, in_channels // 2),
            nn.GELU(),
            nn.Conv2d(in_channels // 2, in_channels // 4, 3, padding=1),
            nn.GroupNorm(4, in_channels // 4),
            nn.GELU(),
            nn.Conv2d(in_channels // 4, 1, 1),
            nn.Hardtanh(min_val=0.01, max_val=0.99)
        )

        # 物理特征融合模块
        self.phys_fusion = nn.Sequential(
            nn.Conv2d(2, 32, 5, padding=2),
            nn.Tanh(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x_v):

        # 温度分布估计
        T_map = self.temp_estimator(x_v)  # [B,1,H,W]

        # 发射率估计
        ε_map = self.emis_net(x_v)  # [B,1,H,W]

        # 物理特征融合
        phys_feat = torch.cat([T_map, ε_map], dim=1)
        att_phys = self.phys_fusion(phys_feat)

        output = x_v * (1 + att_phys)

        # 稳定残差增强
        return output

