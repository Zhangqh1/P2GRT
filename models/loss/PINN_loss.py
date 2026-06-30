import torch
import torch.nn as nn
import torch.nn.functional as F

class PINNLoss(nn.Module):
    def __init__(self):
        super(PINNLoss, self).__init__()

        # Sobel 卷积核
        sobel_x_weights = torch.tensor(
            [[-1, 0, 1],
             [-2, 0, 2],
             [-1, 0, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3)

        sobel_y_weights = torch.tensor(
            [[-1, -2, -1],
             [0, 0, 0],
             [1, 2, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3)

        # 注册为 buffer，而不是 Parameter（不参与梯度更新）
        self.register_buffer("sobel_x", sobel_x_weights)
        self.register_buffer("sobel_y", sobel_y_weights)

    def forward(self, x):
        pde_loss = self.loss_pde(x)
        bc_loss = self.loss_bc(x)

        return pde_loss, bc_loss

    def sobel_filter(self, img):
        """支持多通道的 Sobel 过滤"""
        C = img.shape[1]
        sobel_x = self.sobel_x.repeat(C, 1, 1, 1)
        sobel_y = self.sobel_y.repeat(C, 1, 1, 1)
        grad_x = F.conv2d(img, sobel_x, padding=1, groups=C)
        grad_y = F.conv2d(img, sobel_y, padding=1, groups=C)
        return grad_x, grad_y

    def loss_pde(self, x):
        x.requires_grad_(True)
        grad_x, grad_y = self.sobel_filter(x)
        grad_norm = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)
        K = 0.1
        c = torch.exp(-(grad_norm / K) ** 2)
        c_grad_x = c * grad_x
        c_grad_y = c * grad_y

        # 定义一阶差分卷积核（水平和垂直方向）
        diff_x_kernel = torch.tensor([[0, 0, 0],
                                      [0, -1, 1],
                                      [0, 0, 0]], dtype=torch.float32).view(1, 1, 3, 3)  # 水平差分
        diff_y_kernel = torch.tensor([[0, 0, 0],
                                      [0, -1, 0],
                                      [0, 1, 0]], dtype=torch.float32).view(1, 1, 3, 3)  # 垂直差分

        # 适配多通道
        C = x.shape[1]
        diff_x_kernel = diff_x_kernel.repeat(C, 1, 1, 1).to(x.device)
        diff_y_kernel = diff_y_kernel.repeat(C, 1, 1, 1).to(x.device)

        # 卷积计算二阶导数（自动padding保持形状）
        div_c_grad_x = F.conv2d(c_grad_x, diff_x_kernel, padding=1, groups=C)
        div_c_grad_y = F.conv2d(c_grad_y, diff_y_kernel, padding=1, groups=C)

        # 散度计算（形状已匹配，无需填充）
        div_term = div_c_grad_x + div_c_grad_y
        loss_pde = torch.mean(div_term ** 2)
        return loss_pde

    def loss_bc(self, x):
        # 1. 计算梯度和扩散系数
        grad_x, grad_y = self.sobel_filter(x)
        grad_norm = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)
        K = 0.3
        c = torch.exp(-(grad_norm / K) ** 2)

        norm_grad = (grad_norm - grad_norm.min()) / (grad_norm.max() - grad_norm.min() + 1e-8)

        # with torch.no_grad():
        #     print(norm_grad.min().item(), norm_grad.max().item(), norm_grad.mean().item())

        # 2. 识别物体边缘区域（梯度较大的区域）
        edge_mask = (norm_grad > 0.05).float()  # 阈值可调整

        # 3. 在边缘区域约束灰度通量（c·∇I）为0
        flux_x = c * grad_x
        flux_y = c * grad_y
        loss_bc = torch.mean(edge_mask * (flux_x ** 2 + flux_y ** 2))  # 仅惩罚边缘区域的通量

        return loss_bc