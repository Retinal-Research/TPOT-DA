import argparse
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from torchvision.utils import save_image
from torchvision import transforms as T
from pytorch_msssim import ssim, ms_ssim

from model.model_LC import _NetG
from model.DA_wrapper import GeneratorWithFullAdapter,VGGFeatureExtractor
from dataloader.EyeQ_sample import DA_Dataset, ValidationSet
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import torch.nn.functional as F
def perceptual_loss(pred, target, vgg_extractor):
    vgg_extractor = vgg_extractor.to(pred.device)

    pred_feats = vgg_extractor(pred)
    target_feats = vgg_extractor(target)

    loss = 0
    for pf, tf in zip(pred_feats, target_feats):
        loss += F.mse_loss(pf, tf)
    return loss

vgg_input_transform = T.Normalize(
    mean=[0.485, 0.456, 0.406], 
    std=[0.229, 0.224, 0.225]
)

# Utility function for PSNR
def PSNR(original, compressed):
    mse = ((original - compressed) ** 2).mean()
    if mse == 0:
        return 100
    return 20 * np.log10(255.0 / np.sqrt(mse))

# Save regular checkpoint
def save_checkpoint(model, epoch, save_dir_path):
    os.makedirs(os.path.join(save_dir_path, 'checkpoints'), exist_ok=True)
    path = os.path.join(save_dir_path, 'checkpoints', f"model_denoise_epoch_{epoch}.pth")
    torch.save({"model": model.state_dict(), "epoch": epoch}, path)
    print(f"Checkpoint saved at {path}")

# Save best SSIM model
def save_best_ssim(model, save_dir_path, ssim_val, psnr_val):
    os.makedirs(os.path.join(save_dir_path, 'checkpoints'), exist_ok=True)
    path = os.path.join(save_dir_path, 'checkpoints', f"best_SSIM_{ssim_val:.4f}_PSNR_{psnr_val:.4f}.pth")
    torch.save(model.state_dict(), path)
    print(f"Best model saved at {path}")

# Validation evaluation
@torch.no_grad()
def eval(val_loader, model, best_ssim, epoch, save_dir_path):
    model.eval()
    device = next(model.parameters()).device
    total_ssim, total_psnr = 0, 0

    for batch in val_loader:
        input_A = batch['A'].to(device)
        target_B = batch['B'].to(device)
        output = model(input_A).clamp(0, 1)

        total_ssim += ssim(output, target_B, data_range=1.0, nonnegative_ssim=True).item()
        total_psnr += PSNR(target_B.squeeze().cpu().numpy() * 255.0, output.squeeze().cpu().numpy() * 255.0)

    avg_ssim = total_ssim / len(val_loader)
    avg_psnr = total_psnr / len(val_loader)

    print(f"Epoch {epoch}: Avg SSIM = {avg_ssim:.4f}, Avg PSNR = {avg_psnr:.2f}")

    if avg_ssim > best_ssim:
        save_best_ssim(model, save_dir_path, avg_ssim, avg_psnr)
        print(f"New best model at epoch {epoch} (SSIM: {avg_ssim:.4f})")
        return avg_ssim

    return best_ssim

# Main training function
def main(opt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(opt.seed)
    cudnn.benchmark = True

    # Load pretrained generator and wrap with adapter
    base_model = _NetG().to(device)
    base_model.load_state_dict(torch.load(opt.checkpoints, map_location=device), strict=True)
    model = GeneratorWithFullAdapter(base_model, opt.n_feat, opt.scale_unetfeats).to(device)

    # Data
    train_loader = DataLoader(DA_Dataset(opt), batch_size=opt.batchSize, shuffle=True, num_workers=opt.threads)
    val_loader = DataLoader(ValidationSet(opt), batch_size=opt.batchSize_val, shuffle=False, num_workers=opt.threads)

    # Optimizer + scheduler
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=opt.nEpochs, eta_min=1e-6)
    mse_loss = nn.MSELoss()

    best_ssim = opt.best_ssim
    vgg_extractor = VGGFeatureExtractor()
    print("Starting training...")

    for epoch in range(opt.start_epoch, opt.nEpochs + 1):
        model.train()
        total_mse, total_ssim, total_perceptual = 0, 0, 0

        for i, batch in enumerate(train_loader):
            input = batch['A'].to(device)
            target = batch['B'].to(device)

            optimizer.zero_grad()
            output = model(input)
            loss_mse = mse_loss(output, target)
            loss_ssim = 1 - ms_ssim(output, target, data_range=1.0)
            output_vgg = vgg_input_transform(output.clamp(0, 1))
            target_vgg = vgg_input_transform(target.clamp(0, 1))
            loss_perc = perceptual_loss(output_vgg, target_vgg, vgg_extractor)
            loss = opt.lambda_mse * loss_mse + opt.lambda_ssim * loss_ssim + opt.lamda_vgg * loss_perc
            loss.backward()
            optimizer.step()

            total_mse += loss_mse.item()
            total_ssim += loss_ssim.item()
            total_perceptual += loss_perc.item()

            if i % opt.print_frequency == 0:
                print(f"Epoch {epoch} [{i}/{len(train_loader)}] - MSE: {loss_mse.item():.4f}, SSIM: {loss_ssim.item():.4f}, Perceutal:{loss_perc.item():.4f}, Total: {loss.item():.4f}")

        scheduler.step()

        # if epoch % opt.save_frequency == 0:
        #     save_checkpoint(model, epoch, opt.save_dir)

        best_ssim = eval(val_loader, base_model, best_ssim, epoch, opt.save_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--batchSize", type=int, default=16)
    parser.add_argument("--nEpochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--cuda", type=bool, default=True)
    parser.add_argument("--checkpoints", type=str, required=True)
    parser.add_argument("--n_feat", type=int, default=80)
    parser.add_argument("--scale_unetfeats", type = int, default = 48)
    parser.add_argument("--csv_good", type=str)
    parser.add_argument("--csv_bad", type=str)
    parser.add_argument("--new_image_size", type=int, default=256)
    parser.add_argument("--root", type=str)
    parser.add_argument("--csv_val", type=str)
    parser.add_argument("--batchSize_val", type=int, default=1)
    parser.add_argument("--best_ssim", type=float, default=0.0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--save_frequency", type=int, default=10)
    parser.add_argument("--print_frequency", type=int, default=10)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--start_epoch", type=int, default=1)
    parser.add_argument("--lambda_mse", type=float, default=0.5)
    parser.add_argument("--lambda_ssim", type=float, default=0.5)
    parser.add_argument("--lamda_vgg", type = float, default = 1)
    parser.add_argument("--seed", type=int, default=random.randint(1, 10000))
    args = parser.parse_args()

    main(args)
