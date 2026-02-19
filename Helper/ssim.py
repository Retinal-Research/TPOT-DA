import os
import numpy as np
import cv2

def calculate_ssim(img1, img2, border=0):
    '''Calculate SSIM between two images.
    img1, img2: Input images with pixel values in the range [0, 255].
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    
    # Ensure border does not exceed half of the image size
    h, w = img1.shape[:2]
    if border > 0:
        img1 = img1[border:h-border, border:w-border]
        img2 = img2[border:h-border, border:w-border]

    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = [ssim(img1[:, :, i], img2[:, :, i]) for i in range(3)]
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')

def ssim(img1, img2):
    '''Helper function to compute SSIM between two single-channel images.'''
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    # 11x11 Gaussian filter with sigma = 1.5
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean()

def calculate_ssim_folder(opt):
    dir_path = os.path.join(opt.root, 'good_quality')
    save_path = opt.save
    ssim_total = 0
    count = 0
    missing_files = []

    for image in os.listdir(dir_path):
        truth_img_path = os.path.join(dir_path, image)
        target_img_path = os.path.join(save_path, image)

        if os.path.exists(truth_img_path) and os.path.exists(target_img_path):
            truth = cv2.imread(truth_img_path)
            target = cv2.imread(target_img_path)

            # Ensure that both images are loaded correctly
            if truth is None or target is None:
                missing_files.append(image)
                continue
            print(f'truth shape:{truth.shape}, target shape:{target.shape}')
            ssim_total += calculate_ssim(truth, target)
            count += 1
        else:
            missing_files.append(image)

    # Log missing or problematic files
    if missing_files:
        with open(os.path.join(opt.save_dir, opt.metrics_name), 'a') as f:
            f.write(f'Missing or corrupt files: {", ".join(missing_files)}\n')

    if count == 0:
        raise ValueError("No valid images found for SSIM calculation.")

    return ssim_total / count


