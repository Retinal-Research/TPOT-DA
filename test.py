import argparse
import os, pdb
import torch, cv2
from torch.autograd import Variable
import numpy as np
import torch.nn as nn
import time, math, glob
# import scipy.io as sio
from PIL import Image
from Helper.ssim import calculate_ssim_folder,calculate_ssim
from torchvision.utils import save_image
from dataloader.EyeQ_sample import TestSet
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from model.model_LC import _NetG,_NetD
## only for cyclegan
#from model_cycle.Model_cycle import _NetG,_NetD


parser = argparse.ArgumentParser(description="OTEGAN test")
parser.add_argument("--checkpoints", default="", type=str, help="the checkpoints path to the generator pth")
parser.add_argument("--save", default="", type=str, help="the dir path to store generation result")
parser.add_argument("--save_dir", default="", type = str, help = 'the directory to store statistic experiments metrics')
parser.add_argument("--metrics_name", default="", type =str, help = 'the name of file that store the ssim and psnr metrics')
parser.add_argument("--gpus", default="0", type=str, help="gpu ids")
parser.add_argument("--new_image_size", type = int, default = 256, help = 'the new image size')
### modified parser parameters
parser.add_argument("--root",default = '', type = str, help = 'root path to directory')
parser.add_argument("--csv_test", default = '', type = str, help = 'the path to csv_test file')

opt = parser.parse_args()

def PSNR(original, compressed): 
    mse = np.mean((original - compressed) ** 2) 
    if(mse == 0):  # MSE is zero means no noise is present in the signal . 
                  # Therefore PSNR have no importance.   ###the pixel range should be 0 to 255
        return 100
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse)) 
    return psnr 


test_dataset = TestSet(opt)
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, drop_last=False)

os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpus)
#cuda = True#opt.cuda


if not os.path.exists(opt.save):
    os.makedirs(opt.save,exist_ok=True)
    print(f'save the generation results here')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load(opt.checkpoints, map_location=torch.device('cuda'))
# print(checkpoint.keys())
model = _NetG()
## not load from best_ssim
#model.load_state_dict(checkpoint["model"], strict = True)
##load from best ssim
model.load_state_dict(checkpoint, strict=True) 
model.to(device)
#####
with torch.no_grad():
    model.eval()
    psnr = 0
    ssim = 0
    for index, batch in enumerate(test_loader):
        input_A = batch['A'].to(device)
        target_B = batch['B'].to(device)

        fake_B = model(input_A)
        #print(f'type:{type(fake_B)}, shape:{fake_B.shape}')
        fake_B = torch.clamp(fake_B,min=0.0,max=1.0)
        psnr += PSNR(target_B.squeeze().cpu().numpy() * 255.0, fake_B.squeeze().cpu().numpy() * 255.0)
        # print(f'{(target_B.squeeze().cpu().numpy() * 255.0).shape},{type(target_B.squeeze().cpu().numpy() * 255.0)}, {np.max(target_B.squeeze().cpu().numpy() * 255.0)}, {np.min(target_B.squeeze().cpu().numpy() * 255.0)}')
        # print(f'{(fake_B.squeeze().cpu().numpy() * 255.0).shape},{type(fake_B.squeeze().cpu().numpy() * 255.0)},{np.max(fake_B.squeeze().cpu().numpy() * 255.0)},{np.min(fake_B.squeeze().cpu().numpy() * 255.0)}')
        # print(calculate_ssim(target_B.squeeze().cpu().numpy() * 255.0, fake_B.squeeze().cpu().numpy() * 255.0))
        ssim += calculate_ssim(target_B.squeeze().permute((1,2,0)).cpu().numpy() * 255.0, fake_B.squeeze().permute((1,2,0)).cpu().numpy() * 255.0)
        fake_name = batch['B_paths'][0].split('/')[-1]
        save_image(fake_B.data, os.path.join(opt.save,fake_name))
    
    ###calculate psnr
    Final_psnr = psnr / len(test_loader)
    ##calculate ssim
    Final_ssim = ssim / len(test_loader)
    ###
    if not os.path.exists(opt.save_dir):
        os.makedirs(opt.save_dir, exist_ok=True)

    with open(os.path.join(opt.save_dir, opt.metrics_name),'a') as f:
        f.write(f'SSIM value:{Final_ssim}'+'\n')
        f.write(f'PSNR value:{Final_psnr}'+'\n')

    print(f'SSIM and PSNR metrics have been calculated')


    

 

