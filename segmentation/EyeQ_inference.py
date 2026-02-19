import torch
import torch.nn as nn
import torch.fft as fft
import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory
parent_dir = os.path.dirname(script_dir)
# Add the parent directory to the Python path
sys.path.append(parent_dir)
import numpy as np
import torch.utils.data
from PIL import Image
from PIL import ImageOps
import re
from torchvision import transforms
import matplotlib.pyplot as plt
from segmentation.con_net import MainNet
from torchvision.utils import save_image

## help function
def numericalSort(value):
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def lstFiles(path):
    image_list=[]
    for image in os.listdir(path):
        if ".png" in image.lower():
            image_list.append(os.path.join(path, image))
        elif ".jpeg" in image.lower():
            image_list.append(os.path.join(path, image))
    image_list = sorted(image_list, key=numericalSort)
    return image_list

def Normalization(image):
    image = image.astype(np.float32)
    min_val = np.min(image)
    max_val = np.max(image)
    if max_val - min_val != 0:
        image = (image - min_val) / (max_val - min_val) * 255.0
    else:
        image = np.zeros_like(image)
    return image

# def save_segmentation_mask(sigmoid_output, threshold=0.5, save_path=''):
#     """
#     Save the segmentation mask from a sigmoid output based on a threshold.

#     Args:
#         sigmoid_output (torch.Tensor): The output tensor from the sigmoid layer, should be in the range [0, 1].
#         threshold (float): The threshold to convert the sigmoid output to a binary mask. Default is 0.5.
#         save_path (str): Path to save the generated mask as a PNG file.
#     """
#     # Apply the threshold to create a binary mask
#     binary_mask = (sigmoid_output > threshold).float()
    
#     # Save the mask as an image
#     save_image(binary_mask, save_path)
def save_segmentation_mask(sigmoid_output, threshold=0.5, save_path=''):
    """
    Save the segmentation mask from a sigmoid output based on a threshold.

    Args:
        sigmoid_output (torch.Tensor): The output tensor from the sigmoid layer, should be in the range [0, 1].
        threshold (float): The threshold to convert the sigmoid output to a binary mask. Default is 0.5.
        save_path (str): Path to save the generated mask as a PNG file.
    """
    ### normalize range to 0 and 1 to illustration
    min_val = sigmoid_output.min().item()
    max_val = sigmoid_output.max().item()
    if max_val > min_val:  # Ensure the denominator is not zero
        normalized_mask = (sigmoid_output - min_val) / (max_val - min_val)
    else:
        normalized_mask = sigmoid_output  # If min and max are equal, return the mask as is
    # Apply the threshold to create a binary mask
    binary_mask = (normalized_mask > threshold).float()
    
    # Convert the binary mask to a PIL Image (ensure it's single-channel)
    binary_mask_pil = Image.fromarray(binary_mask.squeeze().cpu().numpy().astype('uint8') * 255)  # Scale to [0, 255]
    
    # Save the mask as a grayscale image (L mode)
    binary_mask_pil.save(save_path, format='PNG')

####
class InferenceSet(torch.utils.data.Dataset):  

    def __init__(self, input_path,new_image_size): ## input_path is the path to father diect
        super(InferenceSet,self).__init__()
        self.dir_A = os.path.join(input_path, 'good_quality')  # create a path to validation set

        self.A_paths=lstFiles(self.dir_A) ## the list that store the absolute path for each images
        self.A_size = len(self.A_paths)
        self.size = new_image_size ## create it load training dataloader

    def read_image(self, path):  ## resize or padding original images based on image absote path
        target_size = (self.size,self.size)
        readed_image = Image.open(path)
        readed_image = readed_image.convert("RGB")
        original_size = readed_image.size
        if original_size[0] < target_size[0] or original_size[1] < target_size[1]:
        # Calculate padding
            delta_width = max(target_size[0] - original_size[0], 0)
            delta_height = max(target_size[1] - original_size[1], 0)
            padding = (delta_width // 2, delta_height // 2, delta_width - (delta_width // 2), delta_height - (delta_height // 2))
        # Pad the image
            image = ImageOps.expand(readed_image, padding)
        else:
        # Resize the image
            image = readed_image.resize(target_size, Image.LANCZOS)
        array_image = np.array(image).astype(np.float32)

        return array_image  ###(H,W,C)
    
    def __len__(self):
        return self.A_size
    
    def __getitem__(self, index):  ## need to return the A(LQ) with regard to related ground truth B

        A_path = self.A_paths[index % self.A_size]  ## find absolte A_path 
        A_img = self.read_image(A_path)
        A_img = Normalization(A_img) # set intensity 0 to 255  (h,w,c) array
        A_img = np.transpose(A_img, (2, 0, 1)) # ### need to modification (h,w,c)-> (c,h,w)

        return {'A': torch.from_numpy(A_img), 'A_paths': A_path}
    

###
def main_test(input_path, new_image_size,save_directory, load_path,threshold):
    ##first create dataset
    Inference_dataset=InferenceSet(input_path,new_image_size)
    Inference_dataloader=torch.utils.data.DataLoader(Inference_dataset,batch_size=1,shuffle=False,num_workers=1,drop_last=False)
    save_path = os.path.join(save_directory,f'Inference_result_testB')
    os.makedirs(save_path , exist_ok=True)
    ## then load generator and pretrained weight
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = MainNet()
    if os.path.exists(load_path):
        generator.load_state_dict(torch.load(load_path),strict=True)
        generator.to(device)
        print(f'SGL generator has been loaded from pre-trained and move to {device}')
    else:
        raise ValueError('load_path does not exist')
    ##
    with torch.no_grad():
        generator.eval()
        for i, data in enumerate(Inference_dataloader):
            input = data['A'].to(device)
            # print(f'type:{type(input)}')
            # print(f'shape:{input.shape}')
            name = data['A_paths'][0].split('/')[-1]#.split('.')[0]
            enhancement_map,likelihood_featuremap = generator(input)
            save_segmentation_mask(likelihood_featuremap,threshold,os.path.join(save_path,name))
        ##
def inference_mask(likelihood_tensor, groundtruth_tensor, generator):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = generator.to(device)
    
    # Check if the pre-trained model exists and load it
    # if os.path.exists(load_path):
    #     generator.load_state_dict(torch.load(load_path), strict=True)
    #     generator.to(device)
    #     print(f'Generator has been loaded from {load_path} and moved to {device}')
    # else:
    #     raise ValueError(f'{load_path} does not exist')
    
    ###
    with torch.no_grad():
        generator.eval()
        # Ensure the input tensors have the same shape
        assert likelihood_tensor.shape == groundtruth_tensor.shape, "Input tensors must have the same shape."
        
        # Generate synthetic masks
        _, synthetic_mask = generator(likelihood_tensor * 255.0) ## to ensure the pixel value range between 0 to 255
        _, truth_mask = generator(groundtruth_tensor * 255.0 )
        
        # Check that the output shape has the second dimension as 1 (for a single-channel mask)
        assert synthetic_mask.shape[1] == 1, "Synthetic mask must have a single channel."
        assert truth_mask.shape[1] == 1, "Ground truth mask must have a single channel."
        
        return synthetic_mask, truth_mask
    
def mask_save(likelihood_tensor, groundtruth_tensor,save_path,threshold,epoch):
    assert likelihood_tensor.shape == groundtruth_tensor.shape #(b,c,h,w)
    B = likelihood_tensor.shape[0]
    select_idx = np.random.randint(0, B)
    likelihood_tensor_mask = likelihood_tensor[select_idx,:,:,:].squeeze()
    #print(f'likelihood_tensor_mask:{likelihood_tensor_mask.shape}')  ##256 * 256
    groundtruth_tensor_mask = groundtruth_tensor[select_idx,:,:,:].squeeze()
    #print(f'groundtruth_tensor_mask:{groundtruth_tensor_mask.shape}')
    ##
    save_segmentation_mask(likelihood_tensor_mask,threshold,os.path.join(save_path,f'{epoch}_fakeB.png'))
    save_segmentation_mask(groundtruth_tensor_mask,threshold,os.path.join(save_path,f'{epoch}_inputA.png'))

if __name__=='__main__':
    input_path = '/home/local/ASURITE/xdong64/Desktop/ISBI_2025/test'
    new_image_size = 256
    save_drectory = '/home/local/ASURITE/xdong64/Desktop/ISBI_2025/test'
    load_path = '/home/local/ASURITE/xdong64/Desktop/ISBI_2025/SGL-Retinal-Vessel-Segmentation/pretrained/drive_k8.pth'
    threshold = 0.5
    main_test(input_path,new_image_size,save_drectory,load_path,threshold)


