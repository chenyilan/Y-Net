# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:       main_beam Ynet+das
   Project Name:    beamform
   Author :         Hengrong LAN
   Date:            2019/1/10
   Device:          GTX1080Ti
-------------------------------------------------
   Change Activity:
                   2019/1/10:
-------------------------------------------------
"""

from networks.model_ynet import YNet
from utils.visualizer import Visualizer
from skimage.measure import compare_ssim, compare_psnr
from utils.mydataset import ReconDataset
from utils.trick import *
import os
import time
import random
import torch
import torchvision
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import numpy as np
import scipy.stats as st

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset_pathr = './data/20181219/'
learning_rate=0.005
batch_size = 64
test_batch = 32
start_epoch=0
loadcp = False


losses = AverageMeter()
batch_time = AverageMeter()
data_time = AverageMeter()
train_ssim_meter = AverageMeter()
train_psnr_meter = AverageMeter()
train_ssim_top20 = AverageMeter(num_top=20)
train_psnr_top20 = AverageMeter(num_top=20)
test_ssim_meter = AverageMeter()
test_psnr_meter = AverageMeter()
test_ssim_top10 = AverageMeter(num_top=10)
test_psnr_top10 = AverageMeter(num_top=10)

losses_list_avg=[]
losses_list_val=[]


curr_lr = learning_rate

vis = Visualizer(env='beam_ynet_3cat')


#source activate pytorch

train_dataset = ReconDataset(dataset_pathr,train=True, das=True)
test_dataset = ReconDataset(dataset_pathr,train=False, das=True)

train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True)
test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=test_batch,
        shuffle=True)


# Model
model = YNet(in_channels=1,merge_mode='concat')
model = nn.DataParallel(model)
model = model.to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

if loadcp:
   checkpoint = torch.load('reconstruction_Unet_2200.ckpt')
   model.load_state_dict(checkpoint['state_dict'])
   start_epoch=checkpoint['epoch']-1
   curr_lr = checkpoint['curr_lr']
   optimizer.load_state_dict(checkpoint['optimizer'])



# For updating learning rate
def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
cudnn.benchmark = True


total_step = len(train_loader)
test_total_step = len(test_loader)

epoch = start_epoch
print("start")
print('train_data :{}'.format(train_dataset.__len__()))
print('test_data :{}'.format(test_dataset.__len__()))
end = time.time()

# Train
while True:
        for batch_idx, (rawdata ,reimage,bfimg) in enumerate(train_loader):
                
                rawdata = rawdata.to(device)
                reimage = reimage.to(device)
                bfimg = bfimg.to(device)
                bfimg = F.upsample(bfimg, (128, 128), mode='bilinear')
                reimage = F.upsample(reimage, (128, 128), mode='bilinear')

                outputs = model(rawdata,bfimg)
                loss = criterion(outputs, reimage)
                losses.update(loss.item(), rawdata.size(0))

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                ssim = compare_ssim(np.array(reimage.detach().squeeze()), np.array(outputs.detach().squeeze()))
                train_ssim_meter.update(ssim)
                psnr = compare_psnr(np.array(reimage.detach().squeeze()), np.array(outputs.detach().squeeze()), data_range=255)
                train_psnr_meter.update(psnr)
                
                # visualization and evaluation
                if (batch_idx +1) %8 ==0:
                   out_image= outputs.detach()
                   bf_img= bfimg.detach()
                   vis_index= random.randint(0,reimage.size(0)-1)
                   vis.img(name='ground truth',img_=10*reimage[vis_index])
                   vis.img(name='beamform', img_=30*bf_img[vis_index])
                   vis.img(name='reconstruction', img_=10*out_image[vis_index])

                batch_time.update(time.time() - end)
                end = time.time()

                if (batch_idx + 1) % 20 == 0:
                        print('Epoch [{}], Start [{}], Step [{}/{}], Loss: {:.4f},Time [{batch_time.val:.3f}({batch_time.avg:.3f})]'
                              .format(epoch + 1, start_epoch, batch_idx + 1, total_step, loss.item(),batch_time=batch_time))

        vis.plot_multi_win(
            dict(
                loss_val=losses.val,
                loss_avg=losses.avg)) 
        losses_list_avg.append(losses.val)
        losses_list_val.append(losses.avg)
        
        vis.plot_multi_win(dict(
                                train_ssim=train_ssim_meter.avg, 
                                train_psnr=train_psnr_meter.avg))  
        mean, deviation, best = train_ssim_top20.top_update_calc(train_ssim_meter.avg)
        vis.plot_single_win(dict(mean=mean, deviation=deviation, best=best), win='train_ssim_')
        
        mean, deviation, best = train_psnr_top20.top_update_calc(train_psnr_meter.avg)
        vis.plot_single_win(dict(mean=mean, deviation=deviation, best=best), win='train_psnr_')
        
        # Validata
        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
               for batch_idx, (rawdata ,reimage,bfimg) in enumerate(test_loader):
                   rawdata = rawdata.to(device)
                   reimage = reimage.to(device)
                   bfimg = bfimg.to(device)
                   bfimg = F.upsample(bfimg, (128, 128), mode='bilinear')
                   outputs = model(rawdata,bfimg)  
                   ssim = compare_ssim(np.array(reimage.squeeze()), np.array(outputs.squeeze()))
                   test_ssim_meter.update(ssim)
                   psnr = compare_psnr(np.array(reimage.squeeze()), np.array(outputs.squeeze()), data_range=255)
                   test_psnr_meter.update(psnr) 
                   if (batch_idx +1) % 10 ==0:
                      out_image= outputs.detach()   
                      bf_img= bfimg.detach()                   
                      vis_index= random.randint(0,reimage.size(0)-1)
                      vis.img(name='Test: ground truth',img_=10*reimage[vis_index])
                      vis.img(name='Test: beamform', img_=30*bf_img[vis_index])
                      vis.img(name='Test: reconstruction', img_=10*out_image[vis_index]) 
                      
               
               vis.plot_multi_win(dict(
                                      test_ssim=test_ssim_meter.avg, 
                                      test_psnr=test_psnr_meter.avg))  
               mean, deviation, best = test_ssim_top10.top_update_calc(test_ssim_meter.avg)
               vis.plot_single_win(dict(mean=mean, deviation=deviation, best=best), win='test_ssim_')
        
               mean, deviation, best = test_psnr_top10.top_update_calc(test_psnr_meter.avg)
               vis.plot_single_win(dict(mean=mean, deviation=deviation, best=best), win='test_psnr_')
         
        # Decay learning rate
        if (epoch + 1) % 50 == 0:
                curr_lr /= 5
                update_lr(optimizer, curr_lr)

        if (epoch+1) % 100 ==0:
                torch.save({'epoch': epoch + 1,
                            'state_dict':model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'curr_lr': curr_lr,
                            'loss_avg':losses_list_avg,
                            'loss_val':losses_list_val
                           },
                          './checkpoint/bfrec_ynet_3cat_das_{}.ckpt'
                           .format(epoch + 1))
                print('Save ckpt successfully!')
        epoch=epoch+1

