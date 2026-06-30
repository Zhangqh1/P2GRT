import time
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from data.data_preprocess import data_preprocess
from models.loss import loss_vif, PINN_loss
from models.backbone import FusionUNet

bar_format = '{l_bar}{bar}| {n_fmt}/{total_fmt} '

# prepare training dataset, please config your dataset path
root_MSRS_train_dir = '/data/zhangqh/DataSet/new_MSRS/IR'
root_LLVIP_train_dir = '/data/zhangqh/DataSet/new_LLVIP/IR'
# transform = transforms.Compose([transforms.Resize((256, 256)), transforms.Grayscale(), transforms.ToTensor()])
MSRS_train = data_preprocess(root_MSRS_train_dir)
LLVIP_train = data_preprocess(root_LLVIP_train_dir)

TraningDatset = MSRS_train + LLVIP_train
train_dataloader = DataLoader(TraningDatset, batch_size=16, drop_last=True, shuffle=True, num_workers=8)

# Create STFNet network model
model = FusionUNet.Net()
device = torch.device("cuda:{}".format(1) if torch.cuda.is_available() else "cpu")
model = model.to(device)

# checkpoint = torch.load("./checkpoints/P2GRTFuse_9.pth")
# model.load_state_dict(checkpoint['net'])

# define loss
compute_loss = loss_vif.fusion_loss_vif()
pinnloss = PINN_loss.PINNLoss().to(device)


# 初始化时分离参数组
ir_params = []  # 红外特征提取层参数
other_params = []  # 其他层参数
target_modules = ["ir_extract_fea1", "ir_extract_fea2", "ir_extract_fea3", "ir_extract_fea4"]
for name, param in model.named_parameters():
    if any(module in name for module in target_modules):
        ir_params.append(param)
    else:
        other_params.append(param)

# define optimizer
learning_rate = 0.0001
# 优化器使用参数组（可分别设置学习率，这里统一）
optimizer = torch.optim.Adam([
    {'params': ir_params, 'lr': learning_rate},
    {'params': other_params, 'lr': learning_rate}
])
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)

# training epoch
epoch = 10

for i in range(1, epoch+1):
    start_time = time.time()
    total_train_step = 0
    print("training on {} epoch starting".format(i))
    model.train()

    for idx, data in enumerate(tqdm(train_dataloader, bar_format=bar_format)):

        img_ir, img_vis = data
        img_ir = img_ir.to(device)
        img_vis = img_vis.to(device)
        fusion_out = model(img_ir, img_vis)

        # training fusion network
        # loss function
        loss_gradient, loss_l1, loss_SSIM = compute_loss(img_ir, img_vis, fusion_out)
        pde_loss, bc_loss = pinnloss(fusion_out)
        loss_gradient, loss_l1, loss_SSIM, pde_loss, bc_loss = 20 * loss_gradient, 40 * loss_l1, 3 * loss_SSIM, 200 * pde_loss, 100 * bc_loss
        total_loss = loss_gradient + loss_l1 + loss_SSIM + pde_loss + bc_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # print training information
        total_train_step = total_train_step + 1
        if total_train_step % 10 == 0:
            print("training with {} epoch and {} times and total loss is {}, "
                  "loss_gradient is {}, loss_l1 is {}, loss_SSIM is {}, pdeloss is {}, bcloss is {}".
                  format(i, total_train_step, round(total_loss.item(), 3), round(loss_gradient.item(), 3),
                         round(loss_l1.item(), 3), round(loss_SSIM.item(), 3), round(pde_loss.item(), 3), round(bc_loss.item(), 3)))



    end_time = time.time()  # 记录周期结束的时间
    time_consume = int(end_time - start_time) // 60
    print(f"Epoch {i + 1} finished in {time_consume} minute")
    scheduler.step()
    # save checkpoint
    checkpoints = {
        "net": model.state_dict()
    }
    torch.save(checkpoints, "./checkpoints/P2GRTFuse_{}.pth".format(i))
