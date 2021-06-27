import os
import sys
import cv2
import argparse
import numpy as np
import ctypes
import matplotlib.pyplot as plt
libgcc_s = ctypes.CDLL('libgcc_s.so.1')
sys.dont_write_bytecode = True
sys.path.insert(0, '../')

import apex
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from model import Model
import albumentations as A
from albumentations.pytorch import ToTensorV2

class Data(Dataset):
    def __init__(self, args):
        self.args      = args
        self.samples   = [name for name in os.listdir(args.datapath+'/image') if name[0]!="."]
        self.transform = A.Compose([
            A.Resize(352, 352),
            A.Normalize(),
            ToTensorV2()
        ])

    def __getitem__(self, idx):
        name   = self.samples[idx]
        image  = cv2.imread(self.args.datapath+'/image/'+name)
        image  = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        origin = image
        H,W,C  = image.shape
        mask   = cv2.imread(self.args.datapath+'/mask/'+name, cv2.IMREAD_GRAYSCALE)/255.0
        pair   = self.transform(image=image, mask=mask)
        return pair['image'], pair['mask'], (H, W), name, origin

    def __len__(self):
        return len(self.samples)

class Test(object):
    def __init__(self, Data, Model, args):
        ## dataset
        self.args      = args 
        self.data      = Data(args)
        self.loader    = DataLoader(self.data, batch_size=1, pin_memory=True, shuffle=True, num_workers=args.num_workers)
        ## model
        self.model     = Model(args)
        self.model.train(False)
        self.model.cuda()

    def save_prediction(self):
        print(self.args.datapath.split('/')[-1])
        with torch.no_grad():
            for image, mask, shape, name, origin in self.loader:
                image = image.cuda().float()
                pred  = self.model(image)
                pred  = F.interpolate(pred, size=shape, mode='bilinear', align_corners=True)[0, 0]
                pred[torch.where(pred>0)] /= (pred>0).float().mean()
                pred[torch.where(pred<0)] /= (pred<0).float().mean()
                pred  = torch.sigmoid(pred).cpu().numpy()*255
                if not os.path.exists(self.args.predpath):
                    os.makedirs(self.args.predpath)
                cv2.imwrite(self.args.predpath+'/'+name[0], np.round(pred))


if __name__=='__main__':
    for name in ['CVC-300', 'CVC-ClinicDB', 'CVC-ColonDB', 'ETIS-LaribPolypDB', 'Kvasir']:
        parser = argparse.ArgumentParser()
        parser.add_argument('--datapath'    ,default='../data/test/'+name)
        parser.add_argument('--predpath'    ,default='../eval/prediction/SANet/'+name)
        parser.add_argument('--mode'        ,default='test')
        parser.add_argument('--num_workers' ,default=4)
        parser.add_argument('--snapshot'    ,default='./out/model-128')
        args = parser.parse_args()

        t = Test(Data, Model, args)
        t.save_prediction()