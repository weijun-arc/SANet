import os
import sys
import cv2
import datetime
import argparse
import numpy as np
import ctypes
libgcc_s = ctypes.CDLL('libgcc_s.so.1')
sys.dont_write_bytecode = True
sys.path.insert(0, '../')

import apex
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import Model
import albumentations as A
from albumentations.pytorch import ToTensorV2

class Data(Dataset):
    def __init__(self, args):
        self.args      = args
        self.samples   = [name for name in os.listdir(args.datapath+'/image') if name[0]!="."]
        self.transform = A.Compose([
            A.Normalize(),
            A.Resize(352, 352),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            ToTensorV2()
        ])
        self.color1, self.color2 = [], []
        for name in self.samples:
            if name[:-4].isdigit():
                self.color1.append(name)
            else:
                self.color2.append(name)

    def __getitem__(self, idx):
        name  = self.samples[idx]
        image = cv2.imread(self.args.datapath+'/image/'+name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

        name2  = self.color1[idx%len(self.color1)] if np.random.rand()<0.7 else self.color2[idx%len(self.color2)]
        image2 = cv2.imread(self.args.datapath+'/image/'+name2)
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2LAB)

        mean , std  = image.mean(axis=(0,1), keepdims=True), image.std(axis=(0,1), keepdims=True)
        mean2, std2 = image2.mean(axis=(0,1), keepdims=True), image2.std(axis=(0,1), keepdims=True)
        image = np.uint8((image-mean)/std*std2+mean2)
        image = cv2.cvtColor(image, cv2.COLOR_LAB2RGB)
        mask  = cv2.imread(self.args.datapath+'/mask/'+name, cv2.IMREAD_GRAYSCALE)/255.0
        pair  = self.transform(image=image, mask=mask)
        return pair['image'], pair['mask']

    def __len__(self):
        return len(self.samples)

def bce_dice(pred, mask):
    ce_loss   = F.binary_cross_entropy_with_logits(pred, mask)
    pred      = torch.sigmoid(pred)
    inter     = (pred*mask).sum(dim=(1,2))
    union     = pred.sum(dim=(1,2))+mask.sum(dim=(1,2))
    dice_loss = 1-(2*inter/(union+1)).mean()
    return ce_loss, dice_loss


class Train(object):
    def __init__(self, Data, Model, args):
        ## dataset
        self.args      = args 
        self.data      = Data(args)
        self.loader    = DataLoader(self.data, batch_size=args.batch_size, pin_memory=True, shuffle=True, num_workers=args.num_workers)
        ## model
        self.model     = Model(args)
        self.model.train(True)
        self.model.cuda()
        ## parameter
        base, head = [], []
        for name, param in self.model.named_parameters():
            if 'bkbone' in name:
                base.append(param)
            else:
                head.append(param)
        self.optimizer = torch.optim.SGD([{'params':base, 'lr':0.1*args.lr}, {'params':head, 'lr':args.lr}], momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)
        self.model, self.optimizer = apex.amp.initialize(self.model, self.optimizer, opt_level='O2')
        self.logger    = SummaryWriter(args.savepath)

    def train(self):
        global_step = 0
        for epoch in range(self.args.epoch):
            if epoch+1 in [64, 96]:
                self.optimizer.param_groups[0]['lr'] *= 0.5
                self.optimizer.param_groups[1]['lr'] *= 0.5

            for image, mask in self.loader:
                image, mask = image.cuda().float(), mask.cuda().float()
                rand  = np.random.choice([256, 288, 320, 352], p=[0.1, 0.2, 0.3, 0.4])
                image = F.interpolate(image, size=(rand, rand), mode='bilinear')
                mask  = F.interpolate(mask.unsqueeze(1),  size=(rand, rand), mode='nearest').squeeze(1)

                pred = self.model(image)
                pred = F.interpolate(pred, size=mask.shape[1:], mode='bilinear', align_corners=True)[:,0,:,:]
                loss_ce, loss_dice = bce_dice(pred, mask)

                self.optimizer.zero_grad()
                with apex.amp.scale_loss(loss_ce+loss_dice, self.optimizer) as scale_loss:
                    scale_loss.backward()
                self.optimizer.step()

                ## log
                global_step += 1
                self.logger.add_scalar('lr'  , self.optimizer.param_groups[0]['lr'], global_step=global_step)
                self.logger.add_scalars('loss', {'ce':loss_ce.item(), 'dice':loss_dice.item()}, global_step=global_step)
                if global_step % 10 == 0:
                    print('%s | step:%d/%d/%d | lr=%.6f | ce=%.6f | dice=%.6f'%(datetime.datetime.now(), global_step, epoch+1, self.args.epoch, self.optimizer.param_groups[0]['lr'], loss_ce.item(), loss_dice.item()))
            if (epoch+1)%8==0:
                torch.save(self.model.state_dict(), self.args.savepath+'/model-'+str(epoch+1))


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath'    ,default='../data/train')
    parser.add_argument('--savepath'    ,default='./out')
    parser.add_argument('--mode'        ,default='train')
    parser.add_argument('--lr'          ,default=0.4)
    parser.add_argument('--epoch'       ,default=128)
    parser.add_argument('--batch_size'  ,default=64)
    parser.add_argument('--weight_decay',default=5e-4)
    parser.add_argument('--momentum'    ,default=0.9)
    parser.add_argument('--nesterov'    ,default=True)
    parser.add_argument('--num_workers' ,default=8)
    parser.add_argument('--snapshot'    ,default=None)
    
    args = parser.parse_args()
    t    = Train(Data, Model, args)
    t.train()


