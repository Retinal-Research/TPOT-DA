import argparse, os, glob
from pickle import TRUE
import torch,pdb
import math, random, time
import numpy as np
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from model.model_LC import _NetG,_NetD,_NetD_512
from torchvision.utils import save_image
import torch.utils.model_zoo as model_zoo
from random import randint, seed
import random
import cv2
from dataloader.EyeQ_sample import EyeQ_Dataset, UnpairedDataSet,ValidationSet
import torchvision.transforms as T
from pytorch_msssim import ssim,ms_ssim
from topoloss import Topo_after_scale
from segmentation.EyeQ_inference import inference_mask, mask_save
from segmentation.con_net import MainNet


def PSNR(pred, gt, shave_border=0):
    height, width = pred.shape[:2]
    pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
    gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
    imdff = pred - gt
    rmse = math.sqrt((imdff ** 2).mean())
    if rmse == 0:
        return 100  
    return 20 * math.log10(1.0 / rmse)

# Training settings
parser = argparse.ArgumentParser(description="PyTorch SRResNet") 
parser.add_argument("--batchSize", type=int, default=16, help="training batch size")
parser.add_argument("--nEpochs", type=int, default=200, help="number of epochs to train for")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning Rate. Default=1e-4")
parser.add_argument("--step", type=int, default=100, help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=500")
parser.add_argument("--cuda", default=True, help="Use cuda?")
parser.add_argument("--resume", default="", type=str, help="Path to resume model (default: none")
parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--threads", type=int, default=0, help="Number of threads for data loader to use, (default: 1)")
parser.add_argument("--pretrained", default="SottGan/Experiment/exp10/checkpoint/model_denoise_200_45.pth", type=str, help="Path to pretrained model (default: none)")
parser.add_argument("--noise_sigma", default=70, type=int, help="standard deviation of the Gaussian noise (default: 50)")
parser.add_argument("--gpus", default="0", type=str, help="gpu ids (default: 0)")
parser.add_argument("--root", default="", type=str, help = 'the directory to parent folder')
###
parser.add_argument("--save_frequency", default = 10, type = int, help= 'the frequency the store the model weight ')
parser.add_argument("--print_frequency", type = int, default = 10, help = 'the frequency to print the loss value')
parser.add_argument("--save_image_frequency", type = int, default = 10, help = 'the frequency to store the generation images and masks')
###
parser.add_argument("--save_dir", type = str, default = "", help = 'the directory to save images and checkpoints in process')
parser.add_argument("--threshold",type = float, default = 0.5, help = 'the threshold to store the segmentation mask feature')
### changes for topo regularization
parser.add_argument("--topo_start_epoch", type = int, default=100, help = ' Where to start to use the topo regularization' )
parser.add_argument("--weight_topo", type = float, default=0.5, help = 'the weight for the topo_loss')
parser.add_argument("--patch_size", type = int, default = 65, help = ' the patch_size for topological loss')
parser.add_argument("--load_seg_path", type = str, default ='', help = 'the default path the segmentation pretrained weight')

parser.add_argument("--weight_ot",type = float, default = 10.0, help = 'weight for the ot_ssim constraint')
parser.add_argument("--weight_idt", type = float, default = 10.0, help = 'weight for the identity constraint')

### additional parameters for EyeQ_training
parser.add_argument("--csv_good",type = str, default='',help = 'the absolute path to csv file for good image')
parser.add_argument("--csv_bad",type = str, default='',help = 'the absolute path to csv file for bad image')
parser.add_argument("--new_image_size", type = int, default = 256, help = 'the new image size')

### define the parameters for validation
parser.add_argument("--csv_val", type = str, default = "", help = 'the csv file to validation set')
parser.add_argument("--batchSize_val", type = int, default = 1, help = 'the batchsize for validation set')
parser.add_argument("--best_ssim", type = float, default = 0.0, help = ' the best ssim value used to update, need to change in continue_training')


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return total_num, trainable_num

def main():
    global opt, model, netContent
    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    cuda = opt
    opt.seed = random.randint(1, 10000)
    print("Random Seed: ", opt.seed)
    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)

    cudnn.benchmark = True

    print("===> Building model") ### define the model used
    model = _NetG()
    discr = _NetD()
 
    print("===> Setting GPU")
    if cuda:
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
        #  dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            model = nn.DataParallel(model)
            discr = nn.DataParallel(discr)

        model.to(device=device)
        discr.to(device=device)

    # optionally resume from a checkpoint
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            opt.start_epoch = checkpoint["epoch"] + 1
            model.load_state_dict(checkpoint["model"])
            discr.load_state_dict(checkpoint["discr"])
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    # optionally copy weights from a checkpoint
    if opt.pretrained:
        if os.path.isfile(opt.pretrained):
            print("=> loading model '{}'".format(opt.pretrained))
            weights = torch.load(opt.pretrained)
            model.load_state_dict(weights['model'],strict=True)
            discr.load_state_dict(weights['discr'],strict=True)
        else:
            print("=> no model found at '{}'".format(opt.pretrained))

    print("===> Training")
    OT_CONSTRAN =[]
    GLOSS=[]
    GI = []
    Psnr = []
    TOPO = []
    print('===> Create DataLoader')
    train_dataset = UnpairedDataSet(opt)
    train_loader = DataLoader(dataset= train_dataset, num_workers= opt.threads, batch_size= opt.batchSize, shuffle= True)
    val_dataset = ValidationSet(opt)
    val_loader = DataLoader(dataset = val_dataset, num_workers= opt.threads, batch_size = opt.batchSize_val, shuffle=False, drop_last=False)
    best_ssim = opt.best_ssim
    print(f'Train and val dataset has been created based on data in :{opt.root}')


    print(f'===> direcotry setting')
    if not os.path.exists(opt.save_dir):
        os.makedirs(opt.save_dir,exist_ok=True)
        print(f'creating saving dir at:{opt.save_dir}')
    ####
    print(f'===> Creating pre-train segmentation network')
    if os.path.exists(opt.load_seg_path):
        generator = MainNet()
        generator.load_state_dict(torch.load(opt.load_seg_path), strict=True)
        print(f'Generator has been loaded from {opt.load_seg_path} and moved to {device}')
    else:
        raise ValueError(f'{opt.load_seg_path} does not exist')
    ####
    print(f'===> Topo module Creating')
    Topo_calculation = Topo_after_scale()
    
    ####
    print("===> Setting Optimizer")
    #G_optimizer = optim.RMSprop(model.parameters(), lr=opt.lr/2)
    # Include both model and topo_module parameters in G_optimizer
    G_optimizer = optim.RMSprop(list(model.parameters()) + list(Topo_calculation.parameters()), lr=opt.lr/2)
    D_optimizer = optim.RMSprop(discr.parameters(), lr=opt.lr)

    ####
    for epoch in range(opt.start_epoch, opt.nEpochs + 1):
        ot = 0
        Gloss=0
        Gidentity = 0
        topo_loss = 0
        if epoch <= opt.topo_start_epoch:
            a,b,c,d = train(train_loader, G_optimizer,D_optimizer,model, discr, epoch, opt.save_dir, generator, Topo_calculation)
            ot += a
            Gloss += b
            Gidentity +=c
        else:
            a,b,c,d,e = train(train_loader, G_optimizer,D_optimizer,model, discr, epoch, opt.save_dir, generator,Topo_calculation)
            ot += a
            Gloss += b
            Gidentity += c
            topo_loss += e

        OT_CONSTRAN.append(format(ot))
        GLOSS.append(format(Gloss))
        GI.append(format(Gidentity))
        Psnr.append(format(d))
        TOPO.append(format(topo_loss))
        if epoch % opt.save_frequency ==0:
            save_checkpoint(model, discr, epoch, opt.save_dir)

        ## inference on validation set and renew the best_ssim
        new_ssim = eval(val_loader,model,best_ssim,epoch,opt.save_dir)
        best_ssim = new_ssim

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10"""
    lr = opt.lr * (0.1 ** (epoch // opt.step))
    return lr 

def train(training_data_loader, G_optimizer, D_optimizer, model, discr, epoch, image_save_path, seg_generator, topo_module):

    lr = adjust_learning_rate(D_optimizer, epoch-1)
    oT = []
    Gloss=[]
    Dloss = []
    Psnr = []
    Gidentity = []
    Topo = []
    for param_group in G_optimizer.param_groups:
        param_group["lr"] = lr/2
    for param_group in D_optimizer.param_groups:
        param_group["lr"] = lr

    print("Epoch={}, lr={}".format(epoch, D_optimizer.param_groups[0]["lr"]))
    #model.train()
    #discr.train()

    for iteration, batch in enumerate(training_data_loader, 1):
        
        target = Variable(batch['B'])
        raw = Variable(batch['A'])

        if opt.cuda:
            target = target.cuda()
            raw = raw.cuda()
            input = raw


        # train discriminator D
        discr.zero_grad()

        D_result = discr(target).squeeze()
        D_real_loss = -D_result.mean()

        G_result = model(input)
        D_result = discr(G_result.data).squeeze()

        D_fake_loss = D_result.mean()

        D_train_loss = D_real_loss + D_fake_loss
        Dloss.append(D_train_loss.data)

        D_train_loss.backward()
        D_optimizer.step()

        #gradient penalty
        discr.zero_grad()
        alpha = torch.rand(target.size(0), 1, 1, 1)
        alpha1 = alpha.cuda().expand_as(target)
        interpolated1 = Variable(alpha1 * target.data + (1 - alpha1) * G_result.data, requires_grad=True)
        
        out = discr(interpolated1).squeeze()

        grad = torch.autograd.grad(outputs=out,
                                   inputs=interpolated1,
                                   grad_outputs=torch.ones(out.size()).cuda(),
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        grad = grad.view(grad.size(0), -1)
        grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
        d_loss_gp = torch.mean((grad_l2norm - 1) ** 2)

        # Backward + Optimize
        gp_loss = 10 * d_loss_gp

        gp_loss.backward()
        D_optimizer.step()

        # train generator G
        discr.zero_grad()
        model.zero_grad()

        G_result = model(input)
        D_result = discr(G_result).squeeze()

        ot_loss = 1 - ms_ssim(G_result,input,data_range=1.0,size_average=True)
        oT.append(ot_loss.data)
        
        new_targe = model(target)
        
        G_identity = 1 - ms_ssim(new_targe,target,data_range=1.0,size_average=True)
        Gidentity.append(G_identity.data)

        if epoch <= opt.topo_start_epoch:
        ### 10 for weight_ot and 15 for weight_idt
            G_train_loss = - D_result.mean() + opt.weight_ot * ot_loss + opt.weight_idt * G_identity 
            Gloss.append(G_train_loss)
            G_train_loss.backward()
            G_optimizer.step()

            pp=PSNR(input,G_result)
            Psnr.append(pp)
        else:
            #print(f"Min_input: {torch.min(input[0,:,:,:])}, Max_input: {torch.max(input[0,:,:,:])},shape:{input.shape}")
            G_result_mask, input_mask = inference_mask(G_result,input,seg_generator)
            #binary_mask = (input_mask > 0.5).float()  ## create the ground_truth mask
            #binary_mask = binary_mask.to(G_result_mask.dtype)
            #print(f"Min: {torch.min(input_mask[0,:,:,:])}, Max: {torch.max(input_mask[0,:,:,:])}")
            #print(f'G_result_mask.shape:{G_result_mask.shape},binary_mask.shape:{binary_mask.shape}') ##(B,1,256,256)
            ### define function to check topo_mask

            #topo_loss_batch = topo_module(G_result_mask, binary_mask, opt.patch_size)
            topo_loss_batch = topo_module(G_result_mask, input_mask, opt.patch_size)
            Topo.append(topo_loss_batch.item())
            ### save mask based on frequency
            if epoch % opt.save_image_frequency ==0:
                if not os.path.exists(os.path.join(image_save_path,'mask')):
                    os.makedirs(os.path.join(image_save_path,'mask'),exist_ok= True)
                    print(f"Creating mask saving folder at: {os.path.join(image_save_path, 'mask')}")
                    ## saving
                mask_save(G_result_mask,input_mask,os.path.join(image_save_path,'mask'),opt.threshold,epoch)
            ###
            G_train_loss = - D_result.mean() + opt.weight_ot * ot_loss + opt.weight_idt * G_identity  + opt.weight_topo * topo_loss_batch
            Gloss.append(G_train_loss)
            G_train_loss.backward()
            G_optimizer.step()

            pp=PSNR(input,G_result)
            Psnr.append(pp)


        if iteration % opt.print_frequency == 0 and epoch <= opt.topo_start_epoch:
            print("===> Epoch[{}]({}/{}): Loss_G: {:.5}, Loss_ot: {:.5}, loss_identity: {:.5}".format(epoch, iteration, len(training_data_loader), G_train_loss.data, ot_loss.data,G_identity.data))
        elif iteration % opt.print_frequency ==0 and epoch > opt.topo_start_epoch:
            print("===> Epoch[{}]({}/{}): Loss_G: {:.5}, Loss_ot: {:.5}, loss_identity: {:.5}, loss_topo:{:.5}".format(epoch, iteration, len(training_data_loader), G_train_loss.data, ot_loss.data,G_identity.data,topo_loss_batch.item()))
    ### save images at the end of every epoch
    if not os.path.exists(os.path.join(image_save_path,'image')):
        os.makedirs(os.path.join(image_save_path,'image'),exist_ok= True)
        print(f"Creating image saving folder at: {os.path.join(image_save_path, 'image')}")

    if epoch % opt.save_image_frequency ==0:
        save_image(G_result.data, os.path.join(os.path.join(image_save_path,'image'), f'{epoch}_output.png' ))
        save_image(input.data, os.path.join(os.path.join(image_save_path,'image'), f'{epoch}_input.png' ))
        save_image(target.data, os.path.join(os.path.join(image_save_path,'image'), f'{epoch}_gt.png' ))

    if epoch <= opt.topo_start_epoch:
        return torch.mean(torch.FloatTensor(oT)),torch.mean(torch.FloatTensor(Gloss)),torch.mean(torch.FloatTensor(Gidentity)),torch.mean(torch.FloatTensor(Psnr))
    if epoch > opt.topo_start_epoch:
        return torch.mean(torch.FloatTensor(oT)),torch.mean(torch.FloatTensor(Gloss)),torch.mean(torch.FloatTensor(Gidentity)),torch.mean(torch.FloatTensor(Psnr)), torch.mean(torch.FloatTensor(Topo))
    
#### checkpoint_saving functions
def save_checkpoint(model, discr, epoch, save_dir_path):
    # Create checkpoint directory if it doesn't exist
    path_check = os.path.join(save_dir_path, 'checkpoints')
    if not os.path.exists(path_check):
        os.makedirs(path_check, exist_ok=True)
        print(f'Creating checkpoint saving folder at: {path_check}')
    
    # Define model output path with formatted epoch number
    model_out_path = os.path.join(path_check, f"model_denoise_epoch_{epoch}.pth")
    
    # Prepare the state to save both the model and the discriminator
    state = {
        "model": model.state_dict(),
        "discr": discr.state_dict(),
        "epoch": epoch  # Save the epoch for resuming training later
    }
    
    # Save the state dictionary
    torch.save(state, model_out_path)
    print(f"Epoch {epoch}: Checkpoint saved to {model_out_path}")


def save_best_ssim(model, save_dir_path):
    path_check = os.path.join(save_dir_path, 'checkpoints')
    if not os.path.exists(path_check):
        os.makedirs(path_check, exist_ok=True)
        print(f'Creating SSIM checkpoints saving folder at: {path_check}')
    
    model_out_path = os.path.join(path_check, "best_SSIM.pth")
    torch.save(model.state_dict(), model_out_path)
    print(f'Model saved at: {model_out_path}')

def eval(val_dataloader, model, best_ssim, epoch, save_dir_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    SSIM_epoch = 0
    with torch.no_grad():
        for i, batch in enumerate(val_dataloader):
            input_A = batch['A'].to(device)
            target_B = batch['B'].to(device)
            
            # Generate the output from the model
            generate_B = model(input_A)
            im_output = torch.clamp(generate_B, min=0.0, max=1.0)
            
            # Compute SSIM for this batch
            SSIM_epoch += ssim(im_output, target_B, data_range=1.0, nonnegative_ssim=True)
        
        # Calculate the average SSIM for the epoch
        best_epoch_ssim = SSIM_epoch / len(val_dataloader)
        print(f'Epoch {epoch}: Average validation SSIM: {best_epoch_ssim:.4f}')
        
        # Save model if SSIM improves
        if best_epoch_ssim > best_ssim:
            best_ssim = best_epoch_ssim
            save_best_ssim(model, save_dir_path)
            print(f'Saving best model at Epoch {epoch} with SSIM: {best_ssim:.4f}')
    
    model.train()
    return best_ssim





if __name__ == "__main__":
    main()
