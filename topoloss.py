import time
import numpy as np
import matplotlib.pyplot as plt
import gudhi as gd
# from pylab import *
import torch
from PIL import Image
import torch.nn as nn
import math



def compute_dgm_force(lh_dgm, gt_dgm, pers_thresh=0.03, pers_thresh_perfect=0.99, do_return_perfect=False):
    """
    Compute the persistent diagram of the image

    Args:
        lh_dgm: likelihood persistent diagram.
        gt_dgm: ground truth persistent diagram.
        pers_thresh: Persistent threshold, which also called dynamic value, which measure the difference.
        between the local maximum critical point value with its neighouboring minimum critical point value.
        The value smaller than the persistent threshold should be filtered. Default: 0.03
        pers_thresh_perfect: The distance difference between two critical points that can be considered as
        correct match. Default: 0.99
        do_return_perfect: Return the persistent point or not from the matching. Default: False

    Returns:
        force_list: The matching between the likelihood and ground truth persistent diagram
        idx_holes_to_fix: The index of persistent points that requires to fix in the following training process
        idx_holes_to_remove: The index of persistent points that require to remove for the following training
        process

    """
    #print(f'lh_dgm shape:{lh_dgm.shape}')
    lh_pers = abs(lh_dgm[:, 1] - lh_dgm[:, 0])
    if (gt_dgm.shape[0] == 0):
        gt_pers = None;
        gt_n_holes = 0;
    else:
        gt_pers = gt_dgm[:, 1] - gt_dgm[:, 0]
        gt_n_holes = gt_pers.size  # number of holes in gt

    if (gt_pers is None or gt_n_holes == 0):
        idx_holes_to_fix = list();
        idx_holes_to_remove = list(set(range(lh_pers.size)))
        idx_holes_perfect = list();
    else:
        # check to ensure that all gt dots have persistence 1
        tmp = gt_pers > pers_thresh_perfect

        # get "perfect holes" - holes which do not need to be fixed, i.e., find top
        # lh_n_holes_perfect indices
        # check to ensure that at least one dot has persistence 1; it is the hole
        # formed by the padded boundary
        # if no hole is ~1 (ie >.999) then just take all holes with max values
        tmp = lh_pers > pers_thresh_perfect  # old: assert tmp.sum() >= 1
        lh_pers_sorted_indices = np.argsort(lh_pers)[::-1]
        if np.sum(tmp) >= 1:
            lh_n_holes_perfect = tmp.sum()
            idx_holes_perfect = lh_pers_sorted_indices[:lh_n_holes_perfect];
        else:
            idx_holes_perfect = list();

        # find top gt_n_holes indices
        idx_holes_to_fix_or_perfect = lh_pers_sorted_indices[:gt_n_holes];

        # the difference is holes to be fixed to perfect
        idx_holes_to_fix = list(
            set(idx_holes_to_fix_or_perfect) - set(idx_holes_perfect))

        # remaining holes are all to be removed
        idx_holes_to_remove = lh_pers_sorted_indices[gt_n_holes:];

    # only select the ones whose persistence is large enough
    # set a threshold to remove meaningless persistence dots
    pers_thd = pers_thresh
    idx_valid = np.where(lh_pers > pers_thd)[0]
    idx_holes_to_remove = list(
        set(idx_holes_to_remove).intersection(set(idx_valid)))

    force_list = np.zeros(lh_dgm.shape)
    
    # push each hole-to-fix to (0,1)
    force_list[idx_holes_to_fix, 0] = 0 - lh_dgm[idx_holes_to_fix, 0]
    force_list[idx_holes_to_fix, 1] = 1 - lh_dgm[idx_holes_to_fix, 1]

    # push each hole-to-remove to (0,1)
    force_list[idx_holes_to_remove, 0] = lh_pers[idx_holes_to_remove] / \
                                         math.sqrt(2.0)
    force_list[idx_holes_to_remove, 1] = -lh_pers[idx_holes_to_remove] / \
                                         math.sqrt(2.0)

    if (do_return_perfect):
        return force_list, idx_holes_to_fix, idx_holes_to_remove, idx_holes_perfect

    return force_list, idx_holes_to_fix, idx_holes_to_remove

def getCriticalPoints(likelihood):
    """
    Compute the critical points of the image (Value range from 0 -> 1)

    Args:
        likelihood: Likelihood image from the output of the neural networks

    Returns:
        pd_lh:  persistence diagram.
        bcp_lh: Birth critical points.
        dcp_lh: Death critical points.
        Bool:   Skip the process if number of matching pairs is zero.

    """
    lh = 1 - likelihood
    lh_vector = np.asarray(lh).flatten()
    ###
    #print(f'lh shape:{lh.shape}')
    expected_size = lh.shape[0] * lh.shape[1]
    if len(lh_vector) != expected_size:
        raise ValueError(f"Mismatch in the number of top-dimensional cells. "
                         f"Expected {expected_size}, got {len(lh_vector)}")

    lh_cubic = gd.CubicalComplex(
        dimensions=[lh.shape[0], lh.shape[1]],
        top_dimensional_cells=lh_vector
    )

    Diag_lh = lh_cubic.persistence(homology_coeff_field=2, min_persistence=0)
    pairs_lh = lh_cubic.cofaces_of_persistence_pairs()

    # If the paris is 0, return False to skip
    if (len(pairs_lh[0])==0): return 0, 0, 0, False

    # return persistence diagram, birth/death critical points
    pd_lh = np.array([[lh_vector[pairs_lh[0][0][i][0]], lh_vector[pairs_lh[0][0][i][1]]] for i in range(len(pairs_lh[0][0]))])
    # if pd_lh.ndim == 1:  # If it's 1D (only one pair), reshape to 2D
    #     pd_lh = pd_lh.reshape(1, -1)
    bcp_lh = np.array([[pairs_lh[0][0][i][0]//lh.shape[1], pairs_lh[0][0][i][0]%lh.shape[1]] for i in range(len(pairs_lh[0][0]))])
    dcp_lh = np.array([[pairs_lh[0][0][i][1]//lh.shape[1], pairs_lh[0][0][i][1]%lh.shape[1]] for i in range(len(pairs_lh[0][0]))])

    return pd_lh, bcp_lh, dcp_lh, True

def getTopoLoss(likelihood_tensor, gt_tensor, topo_size=100):
    """
    Calculate the topology loss of the predicted image and ground truth image 
    Warning: To make sure the topology loss is able to back-propagation, likelihood 
    tensor requires to clone before detach from GPUs. In the end, you can hook the
    likelihood tensor to GPUs device.

    Args:
        likelihood_tensor:   The likelihood pytorch tensor (H,W).
        gt_tensor        :   The groundtruth of pytorch tensor (H,W).
        topo_size        :   The size of the patch is used. Default: 100

    Returns:
        loss_topo        :   The topology loss value (tensor)

    """
    ## since original output is already pass sigmoid,here directly use likelihood_tensor
    #likelihood = torch.sigmoid(likelihood_tensor).clone()
    likelihood = likelihood_tensor.clone()
    gt = gt_tensor.clone()

    likelihood = torch.squeeze(likelihood).cpu().detach().numpy()
    gt = torch.squeeze(gt).cpu().detach().numpy()

    topo_cp_weight_map = np.zeros(likelihood.shape)
    topo_cp_ref_map = np.zeros(likelihood.shape)

    for y in range(0, likelihood.shape[0], topo_size):
        for x in range(0, likelihood.shape[1], topo_size):

            lh_patch = likelihood[y:min(y + topo_size, likelihood.shape[0]),
                         x:min(x + topo_size, likelihood.shape[1])]
            gt_patch = gt[y:min(y + topo_size, gt.shape[0]),
                         x:min(x + topo_size, gt.shape[1])]

            if(np.min(lh_patch) == 1 or np.max(lh_patch) == 0): continue
            if(np.min(gt_patch) == 1 or np.max(gt_patch) == 0): continue
            ##
            #print(f'lh_patch type:{type(lh_patch)},shape:{lh_patch.shape}') #lh_patch type:<class 'np.ndarray'>,shape:(3, 65, 256)
            # Get the critical points of predictions and ground truth
            pd_lh, bcp_lh, dcp_lh, pairs_lh_pa = getCriticalPoints(lh_patch)
            pd_gt, bcp_gt, dcp_gt, pairs_lh_gt = getCriticalPoints(gt_patch)

            # If the pairs not exist, continue for the next loop and if the pd_lh shape is not equal to pd_gt, continue for the next loop
            if not(pairs_lh_pa): continue
            if not(pairs_lh_gt): continue
            if not(len(pd_lh.shape) == 2): continue
            ###
            # print(f'pd_lh.shape:{pd_lh.shape}')
            # print(f'pd_gt.shape:{pd_gt.shape}')

            force_list, idx_holes_to_fix, idx_holes_to_remove = compute_dgm_force(pd_lh, pd_gt, pers_thresh=0.03)

            if (len(idx_holes_to_fix) > 0 or len(idx_holes_to_remove) > 0):
                for hole_indx in idx_holes_to_fix:
                    if (int(bcp_lh[hole_indx][0]) >= 0 and int(bcp_lh[hole_indx][0]) < likelihood.shape[0] and int(
                            bcp_lh[hole_indx][1]) >= 0 and int(bcp_lh[hole_indx][1]) < likelihood.shape[1]):
                        topo_cp_weight_map[y + int(bcp_lh[hole_indx][0]), x + int(
                            bcp_lh[hole_indx][1])] = 1  # push birth to 0 i.e. min birth prob or likelihood
                        topo_cp_ref_map[y + int(bcp_lh[hole_indx][0]), x + int(bcp_lh[hole_indx][1])] = 0
                    if (int(dcp_lh[hole_indx][0]) >= 0 and int(dcp_lh[hole_indx][0]) < likelihood.shape[
                        0] and int(dcp_lh[hole_indx][1]) >= 0 and int(dcp_lh[hole_indx][1]) <
                            likelihood.shape[1]):
                        topo_cp_weight_map[y + int(dcp_lh[hole_indx][0]), x + int(
                            dcp_lh[hole_indx][1])] = 1  # push death to 1 i.e. max death prob or likelihood
                        topo_cp_ref_map[y + int(dcp_lh[hole_indx][0]), x + int(dcp_lh[hole_indx][1])] = 1
                for hole_indx in idx_holes_to_remove:
                    if (int(bcp_lh[hole_indx][0]) >= 0 and int(bcp_lh[hole_indx][0]) < likelihood.shape[
                        0] and int(bcp_lh[hole_indx][1]) >= 0 and int(bcp_lh[hole_indx][1]) <
                            likelihood.shape[1]):
                        topo_cp_weight_map[y + int(bcp_lh[hole_indx][0]), x + int(
                            bcp_lh[hole_indx][1])] = 1  # push birth to death  # push to diagonal
                        if (int(dcp_lh[hole_indx][0]) >= 0 and int(dcp_lh[hole_indx][0]) < likelihood.shape[
                            0] and int(dcp_lh[hole_indx][1]) >= 0 and int(dcp_lh[hole_indx][1]) <
                                likelihood.shape[1]):
                            topo_cp_ref_map[y + int(bcp_lh[hole_indx][0]), x + int(bcp_lh[hole_indx][1])] = \
                                lh_patch[int(dcp_lh[hole_indx][0]), int(dcp_lh[hole_indx][1])]
                        else:
                            topo_cp_ref_map[y + int(bcp_lh[hole_indx][0]), x + int(bcp_lh[hole_indx][1])] = 1
                    if (int(dcp_lh[hole_indx][0]) >= 0 and int(dcp_lh[hole_indx][0]) < likelihood.shape[
                        0] and int(dcp_lh[hole_indx][1]) >= 0 and int(dcp_lh[hole_indx][1]) <
                            likelihood.shape[1]):
                        topo_cp_weight_map[y + int(dcp_lh[hole_indx][0]), x + int(
                            dcp_lh[hole_indx][1])] = 1  # push death to birth # push to diagonal
                        if (int(bcp_lh[hole_indx][0]) >= 0 and int(bcp_lh[hole_indx][0]) < likelihood.shape[
                            0] and int(bcp_lh[hole_indx][1]) >= 0 and int(bcp_lh[hole_indx][1]) <
                                likelihood.shape[1]):
                            topo_cp_ref_map[y + int(dcp_lh[hole_indx][0]), x + int(dcp_lh[hole_indx][1])] = \
                                lh_patch[int(bcp_lh[hole_indx][0]), int(bcp_lh[hole_indx][1])]
                        else:
                            topo_cp_ref_map[y + int(dcp_lh[hole_indx][0]), x + int(dcp_lh[hole_indx][1])] = 0

    topo_cp_weight_map = torch.tensor(topo_cp_weight_map, dtype=torch.float).cuda()
    topo_cp_ref_map = torch.tensor(topo_cp_ref_map, dtype=torch.float).cuda()

    # Measuring the MSE loss between predicted critical points and reference critical points
    loss_topo = (((likelihood_tensor * topo_cp_weight_map) - topo_cp_ref_map) ** 2).sum()
    return loss_topo

# def getTopo_batch(likelihood_batch_tensor, gt_batch_tensor, topo_size=65):
#     if len(likelihood_batch_tensor.shape) != 4 or len(gt_batch_tensor.shape) != 4 or likelihood_batch_tensor.shape != gt_batch_tensor.shape:
#         raise ValueError(f'Output shape: {likelihood_batch_tensor.shape}, Ground truth shape: {gt_batch_tensor.shape}')
#     else:
#         B, C, H, W = likelihood_batch_tensor.shape  # C here should be 1
#         total_topo_loss = 0.0  # Initialize loss accumulator
        
#         for b_index in range(B):
#             likelihood_tensor = likelihood_batch_tensor[b_index, :, :, :].squeeze()
#             gt_tensor = gt_batch_tensor[b_index, :, :, :].squeeze()
            
#             # Compute topo loss for the current batch item and add to total
#             topo_loss = getTopoLoss(likelihood_tensor, gt_tensor, topo_size)
#             total_topo_loss += topo_loss
        
#         # Calculate the average loss over the batch
#         avg_topo_loss = total_topo_loss / B
        
#         # Ensure the result is on the correct device (GPU)
#         return avg_topo_loss.cuda() if likelihood_batch_tensor.is_cuda else avg_topo_loss
    
### create learnable scaler value to scale the proper topoloss
class Topo_after_scale(nn.Module):
    def __init__(self):
        super(Topo_after_scale, self).__init__()
        # Trainable scalar (initialized with some value, e.g., 0.0)
        self.logvar = nn.Parameter(torch.tensor(0.0, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")))
    
    def getTopo_batch(self, likelihood_batch_tensor, gt_batch_tensor, topo_size=65):
        # Check for input shape validity
        if len(likelihood_batch_tensor.shape) != 4 or len(gt_batch_tensor.shape) != 4 or likelihood_batch_tensor.shape != gt_batch_tensor.shape:
            raise ValueError(f'Output shape: {likelihood_batch_tensor.shape}, Ground truth shape: {gt_batch_tensor.shape}')
        
        B, C, H, W = likelihood_batch_tensor.shape  # C here should be 1
        total_topo_loss = 0.0  # Initialize loss accumulator

        # Iterate over the batch
        for b_index in range(B):
            likelihood_tensor = likelihood_batch_tensor[b_index, :, :, :].squeeze()
            gt_tensor = gt_batch_tensor[b_index, :, :, :].squeeze()

            # Compute topo loss for the current batch item and add to total
            #print(f'likelihood_tensor shape:{likelihood_tensor.shape}, gt_tensor shape:{gt_tensor.shape}')
            topo_loss = getTopoLoss(likelihood_tensor, gt_tensor, topo_size)
            total_topo_loss += topo_loss
        
        # Calculate the average loss over the batch
        avg_topo_loss = total_topo_loss / B

        # Ensure the result is on the correct device (GPU)
        if likelihood_batch_tensor.is_cuda:
            avg_topo_loss = avg_topo_loss.cuda()

        return avg_topo_loss

    def forward(self, likelihood_batch_tensor, gt_batch_tensor, topo_size=65):
        # Optionally clamp logvar to a safe range to prevent extreme values
        logvar_clamped = torch.clamp(self.logvar, min=-5, max=8)
        print(f"logvar_clamped: {logvar_clamped.item()}")
        # Compute the topological loss for the batch
        topo_loss = self.getTopo_batch(likelihood_batch_tensor, gt_batch_tensor, topo_size)
        # print(f'topo_loss:{topo_loss}')
        # print(f'logvar_clamped:{logvar_clamped}')


        # Scale the loss using logvar and regularization
        #scaled_output = torch.log(topo_loss + 1) / torch.exp(logvar_clamped) + 0.5 * logvar_clamped
        scaled_output = topo_loss / torch.exp(logvar_clamped) + 0.5 * logvar_clamped
        # print(f'scaled_topo_loss:{scaled_output}')

        return scaled_output

    

# def test_getTopoLoss(batch_size):
#     # Create a sample likelihood tensor and ground truth tensor
#     batch_size = batch_size
#     height, width = 256, 256
#     likelihood_tensor = torch.rand( batch_size,1,height, width).cuda()  # Simulate a batch of predicted tensors
#     gt_tensor = torch.rand( batch_size,1, height, width).cuda()          # Simulate a batch of ground truth tensors
    
#     # Call the topology loss function
#     topo_loss = getTopo_batch(likelihood_tensor, gt_tensor, topo_size=65)
    
#     # Print the result
#     print(f"Topology Loss: {topo_loss.item()}, {type(topo_loss)}, {topo_loss.device}")

# def check_mask_shape_pil(mask_path):
#     # Open the image (mask)
#     mask = Image.open(mask_path)
    
#     # Get the mode of the image (L for single channel, RGB for 3 channels)
#     print(f"Mask mode: {mask.mode}")
    
#     # Get the image size (width, height)
#     width, height = mask.size
    
#     # If it's grayscale (L), the channel is 1
#     channels = 1 if mask.mode == 'L' else len(mask.getbands())
    
#     # Display image shape as (height, width, channels)
#     print(f"Mask shape: {(height, width, channels)}")

if __name__ =='__main__':
    #test_getTopoLoss(5)
    path = '/home/local/ASURITE/xdong64/Desktop/ISBI_2025/test/Inference_result_testA/10001_right_001.jpeg'
    #check_mask_shape_pil(path)