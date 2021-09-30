import os
from os import path as osp
import sys

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
sys.path.append(osp.join(sys.path[0], '../'))
sys.path.append(osp.join(sys.path[0], '../../'))
# sys.path.append(osp.join(sys.path[0], '../../../'))
import time
import torch
import cv2
import torch.nn as nn
import shutil
import numpy as np
from PIL import Image
import torch.nn.functional as F

from src.myideas.swin_transformer.segmentors.encoder_decoder import EncoderDecoder as Swin




def adjust_learning_rate(optimizer, cur_iter, max_iters, lr_pow, set_lr, warmup_steps):

    warm_lr = 1e-6/4
    if cur_iter < warmup_steps:
        linear_step = set_lr - warm_lr
        lr = warm_lr + linear_step * (cur_iter / warmup_steps)
    else:
        scale_running_lr = ((1. - float(cur_iter) / max_iters) ** lr_pow)
        lr = set_lr * scale_running_lr

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr

def train(args, get_dataloader_func=get_pix2pix_maps_dataloader):

    model_saver = ModelSaver(save_path=args.save,
                             name_list=['swin_transformer'])

    sw = SummaryWriter(args.tensorboard_path)



    swin_transformer = Swin('swin_base_patch4_window7_224.pth')
    print(swin_transformer)

    if args.gpu:

        swin_transformer=swin_transformer.cuda()
    model_saver.load('swin_transformer', swin_transformer)


    #get optim
    nodecay_params = []
    other_params = []
    for pname, p in swin_transformer.named_parameters():
        if 'absolute_pos_embed' in pname or 'relative_position_bias_table' in pname or 'norm' in pname:
            # print(pname)
            nodecay_params += [p]
        else:
            other_params += [p]
    optim_p = [{'params': other_params},
              {'params': nodecay_params, 'weight_decay': 0. }]

    #add to optim
    swin_optimizer = torch.optim.AdamW(optim_p,lr=0.00006/4,betas=(0.9, 0.999),weight_decay=0.01)


    device = get_device(args)



    data_loader = get_dataloader_func(args, train=True, flag=2)
    datasize = len(data_loader)
    max_iters = datasize * args.epochs


    total_steps = 0
    for epoch in range(epoch_now, args.epochs):

        data_loader = get_dataloader_func(args, train=True, flag=2)


        data_loader = tqdm(data_loader)

        for step, sample in enumerate(data_loader):
            total_steps = total_steps + 1

            imgs_seg = sample['A_seg'].to(device)  # (shape: (batch_size, 3, img_h, img_w))
            label_imgs = sample['seg'].type(torch.LongTensor).to(device)  # (shape: (batch_size, img_h, img_w))
            label_imgs_unsq = torch.unsqueeze(label_imgs, 1)




            swin_lr = adjust_learning_rate(swin_optimizer, cur_iter, max_iters, lr_pow=1, set_lr=0.00006 / 4,
                                           warmup_steps=1500)

            swinloss, outputs, feature_map = swin_transformer.forward_train(imgs_seg, label_imgs_unsq)

            seg_loss = swinloss['decode.loss_seg'] + swinloss['aux.loss_seg']

 


            swin_optimizer.zero_grad()

            seg_loss.backward()

            swin_optimizer.step()


        


if __name__ == '__main__':
    args = config()

    # args.label_nc = 5

    from src.pix2pixHD.myutils import seed_torch
    print(f'\nset seed as {args.seed}!\n')
    seed_torch(args.seed)

    train(args)

pass
