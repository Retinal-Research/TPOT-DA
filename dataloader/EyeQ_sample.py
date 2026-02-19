import os 
import pandas as pd
import cv2
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
import random


class EyeQ_Dataset(Dataset):
    def __init__(self, root, file_dir, select_number, transform_HQ=None, transform_PQ = None):
        
        dataset = pd.read_csv(file_dir)

        #image_id = dataset.iloc[:,0].values

        image_dir = dataset.iloc[:,1].values

        image_quality = dataset.iloc[:,2].values
        
        image_Grading_label = dataset.iloc[:,3].values


        # select number of image    0->Hight Quality    1->usable   2->Poor quality
        index_HQ =  np.where(image_quality == 0)[0]
        #random_HQ_select = np.random.randint(low=0,high=int(len(index_HQ) - 1),size=select_number)
        #index_HQ_select = np.take(index_HQ,random_HQ_select)

        # Hight Quality 
        self.HQ_images = np.take(image_dir,index_HQ)
        self.HQ_DRgrading_labels = np.take(image_Grading_label,index_HQ)


        index_PQ = np.where(image_quality == 2)[0]
        random_PQ_select = np.random.randint(low=0,high=int(len(index_PQ) - 1),size=select_number)
        index_PQ_select = np.take(index_PQ,random_PQ_select)


        # Poor Quality 
        self.PQ_images = np.take(image_dir,index_PQ_select)
        self.PQ_DRgrading_labels = np.take(image_Grading_label,index_PQ_select)
        #labels = dataset.iloc[:, 1].values
        #image_ids = dataset.iloc[:, 0].values
        self.transform_HQ = transform_HQ
        self.transform_PQ = transform_PQ
        self.root = root
        self.select_number = select_number

    def __len__(self):
        return self.select_number


    def __getitem__(self, idx):

        # High Quality Image 
        # Generate the corresponding index of the label with hight quality image
        dr_label = self.PQ_DRgrading_labels[idx]
        #print(dr_label)
        all_HQ_index_with_same_label = np.where(self.HQ_DRgrading_labels == dr_label)[0]
        #print(all_HQ_index_with_same_label)
        random_HQ_index = np.random.randint(low=0,high=int(len(all_HQ_index_with_same_label) - 1),size=1)[0]
        generate_HQ_index = all_HQ_index_with_same_label[random_HQ_index]

        #print(random_HQ_index[0])
        hq_dr_label = self.HQ_DRgrading_labels[generate_HQ_index]

        HQ_file = os.path.splitext(self.HQ_images[generate_HQ_index])[0] + '.png'
        #print(hq_path)
        HQ_path = os.path.join(self.root,HQ_file)
        
        HQ_image = Image.open(HQ_path)

        #HQ_image = cv2.imread(HQ_path)
        #HQ_image = cv2.cvtColor(HQ_image, cv2.COLOR_BGR2RGB)
        
        if self.transform_HQ is not None:
            # transform_HQ = self.transform_HQ(image=HQ_image)
            # hq = transform_HQ["image"]
            hq = self.transform_HQ(HQ_image)

        #muti_label = load_image_label_from_xml(label)

            # mask = transformed["mask"]
        PQ_file = os.path.splitext(self.PQ_images[idx])[0] + '.png'
        PQ_path = os.path.join(self.root,PQ_file)
        PQ_image = Image.open(PQ_path)
        #PQ_image = cv2.imread(PQ_path)
        #PQ_image = cv2.cvtColor(PQ_image, cv2.COLOR_BGR2RGB)

        if self.transform_PQ is not None:
            # a 
            # transform_PQ = self.transform_PQ(image=PQ_image)
            # qp = transform_PQ["image"]
            pq = self.transform_PQ(PQ_image)


        # if self.augmentation is not None: 
        #     augmentation = self.augmentation(image=ori_image)
        #     aug = augmentation["image"]

        #     return image, aug,label
        # else:
        return pq,hq,dr_label,hq_dr_label
    
####
class UnpairedDataSet(Dataset):
    
    def __init__(self, opt):
                 
        super(UnpairedDataSet,self).__init__()
        ### modification for cvpr_baseline ###
        csv_file_low_quality  = pd.read_csv(opt.csv_bad, header=0)
        image_filenames_low = csv_file_low_quality.iloc[:, 0]
        csv_file_high_quality = pd.read_csv(opt.csv_good, header=0)
        image_filenames_high = csv_file_high_quality.iloc[:, 0]

        image_dir_A = image_filenames_low.apply(lambda x: os.path.join(opt.root, 'de_image', x))
        image_dir_B = image_filenames_high.apply(lambda x: os.path.join(opt.root, 'good_quality', x))

        self.A_paths = image_dir_A.tolist()
        self.B_paths = image_dir_B.tolist()

        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)

        self.size = opt.new_image_size
        # self.augmentation = opt.agumentation
        self.transform = T.Compose([
            T.Resize((self.size, self.size)),  # Resize to 256x256
            # T.RandomHorizontalFlip(),  # Apply random horizontal flip
            # T.RandomVerticalFlip(),    # Apply random vertical flip
            T.ToTensor(),  # Convert back to tensor (C, H, W)
        ])


    def read_image(self, path):
        # Open the image and convert to RGB (for JPEG support)
        readed_image = Image.open(path).convert("RGB")
        return readed_image 

    def __getitem__(self, index):

        A_path = self.A_paths[index % self.A_size] 

        B_path = self.B_paths[index % self.B_size]

        A_img = self.read_image(A_path)  ## return the PIL object
        B_img = self.read_image(B_path)

        A_img = self.transform(A_img)  # Convert (H, W, C) -> (C, H, W) Tensor with range (0,1)
        B_img = self.transform(B_img)

        #return {'A': torch.from_numpy(A_img), 'B': torch.from_numpy(B_img), 'A_paths': A_path, 'B_paths': B_path}
        return {'A': A_img, 'B': B_img, 'A_paths': A_path, 'B_paths': B_path}


    def __len__(self):
        return max(self.A_size, self.B_size)
###
## define the dataloader that focus on paired OOD dataset to fine fine ##
class DA_Dataset(Dataset):
    def __init__(self,opt):
        super(DA_Dataset,self).__init__()
        csv_file_low_quality  = pd.read_csv(opt.csv_bad, header=0)
        image_filenames_low = csv_file_low_quality.iloc[:, 0]
        csv_file_high_quality = pd.read_csv(opt.csv_good, header=0)
        image_filenames_high = csv_file_high_quality.iloc[:, 0]

        image_dir_A = image_filenames_low.apply(lambda x: os.path.join(opt.root, 'de_image', x))
        image_dir_B = image_filenames_high.apply(lambda x: os.path.join(opt.root, 'good_quality', x))

        self.A_paths = image_dir_A.tolist()
        self.B_paths = image_dir_B.tolist()

        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)

        self.size = opt.new_image_size
        # self.augmentation = opt.agumentation
        self.transform = T.Compose([
            T.Resize((self.size, self.size)),  # Resize to 256x256
            T.ToTensor(),  # Convert back to tensor (C, H, W)
        ])
        self.high_quality_path = opt.root


    def read_image(self, path):
        # Open the image and convert to RGB (for JPEG support)
        readed_image = Image.open(path).convert("RGB")
        return readed_image 

    def __getitem__(self, index):
        ### This is paired setting ###
        A_path = self.A_paths[index % self.A_size] 
        image_name = A_path.split('/')[-1] 
        B_path = os.path.join(self.high_quality_path, 'good_quality',image_name)

        A_img = self.read_image(A_path)  ## return the PIL object
        B_img = self.read_image(B_path)

        A_img = self.transform(A_img)  # Convert (H, W, C) -> (C, H, W) Tensor with range (0,1)
        B_img = self.transform(B_img)

        #return {'A': torch.from_numpy(A_img), 'B': torch.from_numpy(B_img), 'A_paths': A_path, 'B_paths': B_path}
        return {'A': A_img, 'B': B_img, 'A_paths': A_path, 'B_paths': B_path}


    def __len__(self):
        return max(self.A_size, self.B_size)

    
###
class ValidationSet(Dataset):  
    def __init__(self, opt):
        super(ValidationSet,self).__init__()

        csv_file_low_quality  = pd.read_csv(opt.csv_val, header=0)
        image_filenames_low = csv_file_low_quality.iloc[:, 0]
        csv_file_high_quality = pd.read_csv(opt.csv_val, header=0)
        image_filenames_high = csv_file_high_quality.iloc[:, 0]

        image_dir_A = image_filenames_low.apply(lambda x: os.path.join(opt.root, 'de_image', x))
        image_dir_B = image_filenames_high.apply(lambda x: os.path.join(opt.root, 'good_quality', x))

        self.A_paths = image_dir_A.tolist()
        self.B_paths = image_dir_B.tolist()

        #####
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        self.size = opt.new_image_size ## create it load training dataloader
        ##
        self.transform = T.Compose([
            T.Resize((self.size, self.size)),  # Resize to 256x256
            T.ToTensor(),  # Convert back to tensor (C, H, W) and normalize to (0,1)
        ])
        self.high_quality_path = opt.root
        self.validation_notexist = opt.save_dir

    def read_image(self, path):
        # Open the image and convert to RGB (for JPEG support)
        readed_image = Image.open(path).convert("RGB")
        return readed_image 
    
    ####
    def __len__(self):
        assert self.A_size == self.B_size
        return max(self.A_size, self.B_size)
    
    def __getitem__(self, index):  ## need to return the A(LQ) with regard to related ground truth B

        A_path = self.A_paths[index % self.A_size]  ## find absolte A_path 
        image_name = A_path.split('/')[-1] ##10001_right_001.jpeg  ## baseline : 10001.right.png
        B_path = os.path.join(self.high_quality_path, 'good_quality',image_name)
        target_path = B_path
        if target_path in self.B_paths:
        
            A_img = self.read_image(A_path)
            B_img = self.read_image(target_path)
            
            A_img = self.transform(A_img)  # Convert (H, W, C) -> (C, H, W) Tensor and normalize to -1 to 1
            B_img = self.transform(B_img)
        else:
            with open(os.path.join(self.validation_notexist,'validation_not_exist.txt'),'a') as f:
                f.write(f'{self.B_paths} does not contain image:{image_name}')
            raise ValueError(f'{self.B_paths} does not contain image:{image_name}')

        return {'A': A_img, 'B': B_img, 'A_paths': A_path, 'B_paths': B_path}

###
class TestSet(Dataset):
    def __init__(self, opt):
        super(TestSet,self).__init__()

        csv_file_low_quality  = pd.read_csv(opt.csv_test, header=0)
        image_filenames_low = csv_file_low_quality.iloc[:, 0]
        csv_file_high_quality = pd.read_csv(opt.csv_test, header=0)
        image_filenames_high = csv_file_high_quality.iloc[:, 0]

        image_dir_A = image_filenames_low.apply(lambda x: os.path.join(opt.root, 'de_image', x))
        image_dir_B = image_filenames_high.apply(lambda x: os.path.join(opt.root, 'good_quality', x))

        self.A_paths = image_dir_A.tolist()
        self.B_paths = image_dir_B.tolist()

        #####
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        self.size = opt.new_image_size ## create it load training dataloader
        ##
        self.transform = T.Compose([
            T.Resize((self.size, self.size)),  # Resize to 256x256
            T.ToTensor(),  # Convert back to tensor (C, H, W) and normalize to (0,1)
        ])
        self.high_quality_path = opt.root
        self.test_notexist = opt.save_dir

    def read_image(self, path):
        # Open the image and convert to RGB (for JPEG support)
        readed_image = Image.open(path).convert("RGB")
        return readed_image 
    
    def __len__(self):
        assert self.A_size == self.B_size
        return max(self.A_size, self.B_size)
    
    def __getitem__(self, index):  ## need to return the A(LQ) with regard to related ground truth B

        A_path = self.A_paths[index % self.A_size]  ## find absolte A_path 
        image_name = A_path.split('/')[-1] ##10001_right_001.jpeg  ## baseline : 10001.right.png
        B_path = os.path.join(self.high_quality_path, 'good_quality',image_name)
        target_path = B_path
        if target_path in self.B_paths:
        
            A_img = self.read_image(A_path)
            B_img = self.read_image(target_path)
            
            A_img = self.transform(A_img)  # Convert (H, W, C) -> (C, H, W) Tensor and normalize to 0 to 1
            B_img = self.transform(B_img)
        else:
            with open(os.path.join(self.test_notexist,'test_not_exist.txt'),'a') as f:
                f.write(f'{self.B_paths} does not contain image:{image_name}')
            raise ValueError(f'{self.B_paths} does not contain image:{image_name}')

        return {'A': A_img, 'B': B_img, 'A_paths': A_path, 'B_paths': B_path}
