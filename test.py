import os
import torch
import time
from PIL import Image
import torchvision.transforms as T
from torchvision.utils import save_image
from models.backbone import FusionUNet
import statistics
from thop import profile

# method name
method = 'P2GRTFuse'
window_size = 16
device = torch.device("cuda:{}".format(1) if torch.cuda.is_available() else "cpu")

# test dataset
# dataset = ['TNO_test', 'MSRS_test', 'RoadScene_test', 'M3FD_test']
# results_dir = ['FusedImg_TNO', 'FusedImg_MSRS', 'FusedImg_RoadScene', 'FusedImg_M3FD']
dataset = ['M3FD_test']
results_dir = ['FusedImg_M3FD1']

for x,ds in enumerate(dataset):
    # source image path
    root_path = '/data/zhangqh/DataSet/Test_Dataset/' + ds
    # load all images
    img_list = os.listdir(os.path.join(root_path, 'ir'))

    for i in range(10, 11):
        global start_time
        # path for save fused image
        fused_path = './Results/' + results_dir[x] + '/' + method + '_' + str(i) + '/'
        # load model
        model = FusionUNet.Net()
        model_path = './checkpoints/P2GRTFuse_' + str(i) + '.pth'
        checkpoint = torch.load(model_path,weights_only=True)
        model.load_state_dict(checkpoint['net'])

        # set for test
        model.to(device)
        model.eval()

        # load all images
        fuse_time = []
        # total_flops = 0.
        for img in img_list:

            img_ir_path = os.path.join(root_path, 'ir', img)
            img_vis_path = img_ir_path.replace('ir/', 'vi/')

            # read ir and vis images
            img_ir = Image.open(img_ir_path)
            img_vis = Image.open(img_vis_path)
            ori_size = img_ir.size

            # transform
            transform = T.Compose([T.Grayscale(), T.ToTensor()])
            img_ir = transform(img_ir)
            img_vis = transform(img_vis)

            img_ir = img_ir.view(1, 1, ori_size[1],ori_size[0]).to(device)
            img_vis = img_vis.view(1, 1, ori_size[1],ori_size[0]).to(device)
            # test
            with torch.no_grad():
                _, _, h_old, w_old = img_ir.size()
                if h_old % window_size != 0:
                    h_pad = (h_old // window_size + 1) * window_size - h_old
                else:
                    h_pad = 0
                if w_old % window_size != 0:
                    w_pad = (w_old // window_size + 1) * window_size - w_old
                else:
                    w_pad = 0
                img_ir = torch.cat([img_ir, torch.flip(img_ir, [2])], 2)[:, :, :h_old + h_pad, :]
                img_ir = torch.cat([img_ir, torch.flip(img_ir, [3])], 3)[:, :, :, :w_old + w_pad]
                img_vis = torch.cat([img_vis, torch.flip(img_vis, [2])], 2)[:, :, :h_old + h_pad, :]
                img_vis = torch.cat([img_vis, torch.flip(img_vis, [3])], 3)[:, :, :, :w_old + w_pad]
                start_time = time.time()
                out = model(img_ir, img_vis)
                end_time = time.time()
                fuse_time.append(end_time - start_time)

                flops, params = profile(model, inputs=(img_ir, img_vis))
                # total_flops = total_flops + flops

                out = out[..., :h_old, :w_old]
                max = out.max()
                min = out.min()
                out = (out - min)/(max - min)
                out = out.view(1, ori_size[1], ori_size[0])
                fusion_img = out
                if not os.path.exists(fused_path):
                    os.makedirs(fused_path)
                save_image(fusion_img, fused_path + img)

        mean = statistics.mean(fuse_time[1:])
        print(f'fuse avg time : {mean:.4f}')
        # print(f'totoal_flops: {total_flops}')
        print(f'params: {params}')

