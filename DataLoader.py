
import os
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import torch
import numpy as np
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import train_transforms, get_boxes_from_mask, init_point_sampling, test_transforms
import json
import random
from PIL import Image
import pdb

#--------------------------------


def resize_target_to_image(target, image):
    """
    将目标数组（target）调整至与图像（image）相同的第一维和第二维尺寸。
    使用双线性插值进行平滑缩放。
    
    参数:
    target -- 目标numpy数组，期望调整形状至与image的H,W匹配
    image -- 参考图像的numpy数组，用于获取目标尺寸
    
    返回:
    resized_target -- 调整尺寸后的目标数组
    """
    # 获取图像的高和宽
    img_height, img_width = image.shape[:2]
    
    # 确保目标是二维的，如果是单通道图像则添加一个通道维度
    if len(target.shape) == 2:
        target = np.expand_dims(target, axis=-1)
    
    # 使用OpenCV进行双线性插值的缩放操作
    resized_target = cv2.resize(target, (img_width, img_height), interpolation=cv2.INTER_LINEAR)
    
    # 如果原始target是二维的，去除添加的通道维度
    if len(target.shape) == 2:
        resized_target = resized_target[:, :, 0]
    
    return resized_target

# 示例用法
#target = np.zeros((320, 320))  # 这里用全零数组作为示例
#image = np.zeros((1920, 1080, 3))

#resized_target = resize_target_to_image(target, image)
#print(resized_target.shape)  # 应该输出 (1920, 1080)

#===================================================
def add_gaussian_noise(image, mean=0, var=0.1):
    sigma = var**0.5
    gaussian = np.random.normal(mean, sigma, image.shape)
    noisy_image = image + gaussian
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    return noisy_image

def add_salt_and_pepper_noise(image, amount=0.05, salt_vs_pepper=0.5):
    noisy_image = np.copy(image)
    num_salt = np.ceil(amount * image.size * salt_vs_pepper)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    noisy_image[tuple(coords)] = 255

    num_pepper = np.ceil(amount * image.size * (1. - salt_vs_pepper))
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    noisy_image[tuple(coords)] = 0
    return noisy_image

def add_occlusion(image, occlusion_size=(50, 50), position=(100, 100)):
    noisy_image = np.copy(image)
    x, y = position
    h, w = occlusion_size
    noisy_image[y:y+h, x:x+w] = 0
    return noisy_image

def apply_motion_blur(image, kernel_size=15):
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
    kernel /= kernel_size
    motion_blur_image = cv2.filter2D(image, -1, kernel)
    return motion_blur_image

def add_poisson_noise(image):
    vals = len(np.unique(image))
    vals = 2 ** np.ceil(np.log2(vals))
    noisy_image = np.random.poisson(image * vals) / float(vals)
    return np.clip(noisy_image, 0, 255).astype(np.uint8)

import matplotlib.pyplot as plt
from PIL import Image

# Assuming `image` is the noisy image output from `self.apply_noise(image)`
def display_image(image, title="Noisy Image"):
    # If the image is in a tensor format (PyTorch style), convert it to numpy
    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).numpy()  # Convert (C, H, W) to (H, W, C)

    # If the image is normalized (e.g., [-1, 1] or [0, 1]), rescale to [0, 255]
    # if image.min() < 0 or image.max() <= 1:
    #     image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)

    # Display the image using matplotlib
    plt.imshow(image)
    plt.axis('off')  # Turn off the axis for cleaner display
    plt.title(title)
    plt.show()
#===================================================



#--------------------------------
class TestingDataset(Dataset):
    
    def __init__(self, data_path, image_size=256, mode='test', requires_name=True, point_num=1, return_ori_mask=True, prompt_path=None, input_mask=False, noise_type=None, noise_params=None):
        """
        Initializes a TestingDataset object.
        Args:
            data_path (str): The path to the data.
            image_size (int, optional): The size of the image. Defaults to 256.
            mode (str, optional): The mode of the dataset. Defaults to 'test'.
            requires_name (bool, optional): Indicates whether the dataset requires image names. Defaults to True.
            point_num (int, optional): The number of points to retrieve. Defaults to 1.
            return_ori_mask (bool, optional): Indicates whether to return the original mask. Defaults to True.
            prompt_path (str, optional): The path to the prompt file. Defaults to None.
        """
        self.image_size = image_size
        self.return_ori_mask = return_ori_mask
        self.prompt_path = prompt_path
        self.prompt_list = {} if prompt_path is None else json.load(open(prompt_path, "r"))
        self.requires_name = requires_name
        self.point_num = point_num

        json_file = open(os.path.join(data_path, f'label2image_{mode}.json'), "r")
        dataset = json.load(json_file)
        
        #----------------------------------------------------------------------------------------------------
        self.input_mask = input_mask
        if self.input_mask:
            json_file_mask = open(os.path.join(data_path, f'input_masks.json'), "r")
            input_mask = json.load(json_file_mask)
            self.mask_paths = list(input_mask.keys())           
        #----------------------------------------------------------------------------------------------------


        self.image_paths = list(dataset.values())
        self.label_paths = list(dataset.keys())
      
        self.pixel_mean = [123.675, 116.28, 103.53]
        self.pixel_std = [58.395, 57.12, 57.375]
    #============================================================================
        self.noise_type = noise_type
        self.noise_params = noise_params if noise_params else {}
    #============================================================================
    
    #=====================================================================================================
    def apply_noise(self, img):
        img = np.array(img)
        if self.noise_type == 'gaussian':
            img = add_gaussian_noise(img, **self.noise_params)
        elif self.noise_type == 'salt_and_pepper':
            img = add_salt_and_pepper_noise(img, **self.noise_params)
        elif self.noise_type == 'occlusion':
            img = add_occlusion(img, **self.noise_params)
        elif self.noise_type == 'apply_motion_blur':
            img = apply_motion_blur(img,**self.noise_params)
        elif self.noise_type == 'add_poisson_noise':
            img = add_poisson_noise(img)
        return Image.fromarray(img)
    
    def save_noisy_image(self, file, img, filename):
        # 生成完整路径
        save_dir = os.path.join(r"F:\Segment\SAM-Med2D-main\append_\public/" f'{self.noise_type}_45')
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, filename.split("/")[-1])
        img.save(save_path)
        # print(f"Saved noisy image to {save_path}")
    #=====================================================================================================

    def __getitem__(self, index):
        """
        Retrieves and preprocesses an item from the dataset.
        Args:
            index (int): The index of the item to retrieve.
        Returns:
            dict: A dictionary containing the preprocessed image and associated information.
        """
        #db.set_trace()
        image_input = {}

            #image = cv2.imread(self.image_paths[index])
        image = Image.open(self.image_paths[index]).convert('RGB')
        #============================================================
        # 应用噪声
        if self.noise_type is not None:
            image = self.apply_noise(image)
            # self.save_noisy_image(self.image_paths[index], image, self.image_paths[index].split('/')[-1])
            # display_image(image, title="Noisy Image")
            # print("noise----------------------------------------------")
        #============================================================  lds


        image = np.array(image)

        image = (image - self.pixel_mean) / self.pixel_std


        mask_path = self.label_paths[index]
        #ori_np_mask = cv2.imread(mask_path, 0)
        ori_np_mask = Image.open(mask_path).convert('L')
        ori_np_mask = np.array(ori_np_mask)
        
        if ori_np_mask.max() == 255:
            ori_np_mask = ori_np_mask / 255
        if ori_np_mask.max() != 1:
            print(1)
        #assert np.array_equal(ori_np_mask, ori_np_mask.astype(bool)), f"Mask should only contain binary values 0 and 1. {self.label_paths[index]}"

        h, w = ori_np_mask.shape
        ori_mask = torch.tensor(ori_np_mask).unsqueeze(0)
        
        
        #-----------------------------------------------------------------------------------------
        #transforms = train_transforms(self.image_size, h, w)
        
        transforms = test_transforms(self.image_size, h, w)
        #-----------------------------------------------------------------------------------------
        
        
        
        #=----------------------------------------------------------------------------------------
        # image_cp = image
        # if ori_np_mask.shape != image.shape[:2]:
        #     image = cv2.resize(image, (768,768), interpolation=cv2.INTER_LINEAR)
        #     ori_np_mask = resize_target_to_image(ori_np_mask, image)
        #pdb.set_trace()
        #-----------------------------------------------------------------------------------------
        
        
        
        augments = transforms(image=image, mask=ori_np_mask)
        image, mask = augments['image'], augments['mask'].to(torch.int64)
        #pdb.set_trace()
        
    
        #-------------------------------------------------------------------------------------
        if self.input_mask:
            transforms_mask = train_transforms(192, h, w)
            mask_input = Image.open(self.mask_paths[index]).convert('L')
            assert  self.mask_paths[index].split('.')[0].split('/')[-1] == self.image_paths[index].split('.')[0].split('/')[-1]
            mask_input = np.array(mask_input)
            if mask_input.max() == 255:
                mask_input = mask_input / 255
            augments_mask = transforms_mask(image=image,mask=mask_input)
            mask_input = augments_mask['mask'].unsqueeze(0).float()
            #pdb.set_trace()
        #-------------------------------------------------------------------------------------
        
        
        
        

        if self.prompt_path is None:
            boxes = get_boxes_from_mask(mask, max_pixel = 0)
            try:
                point_coords, point_labels = init_point_sampling(mask, self.point_num)
            except Exception as e:
                print(f"ValueError encountered at index {self.image_paths[index]}, returning default values.=======================================================================================================================================================")
                point_coords, point_labels = None, None 
            
            
        else:
            prompt_key = mask_path.split('/')[-1]
            boxes = torch.as_tensor(self.prompt_list[prompt_key]["boxes"], dtype=torch.float)
            point_coords = torch.as_tensor(self.prompt_list[prompt_key]["point_coords"], dtype=torch.float)
            point_labels = torch.as_tensor(self.prompt_list[prompt_key]["point_labels"], dtype=torch.int)

        image_input["image"] = image
        image_input["label"] = mask.unsqueeze(0)
        image_input["point_coords"] = point_coords
        image_input["point_labels"] = point_labels
        image_input["boxes"] = boxes
        image_input["original_size"] = (h, w)
        image_input["label_path"] = '/'.join(mask_path.split('/')[:-1])
        
        
        
        #------------------------------------------------------------------------------------
        if self.input_mask:
            image_input["mask_inputs"] = mask_input
        #------------------------------------------------------------------------------------

        if self.return_ori_mask:
            image_input["ori_label"] = ori_mask
     
        image_name = self.label_paths[index].split('/')[-1]
        if self.requires_name:
            image_input["name"] = image_name
            return image_input
        else:
            return image_input

    def __len__(self):
        return len(self.label_paths)

    
    
    
    
    

#-------------------------------------------------------------------------------------------------------------------   
# def charge_mask_format(path):

#     mask_input = Image.open(path).convert('L')
#     mask_input = np.array(mask_input)
#     if mask_input.max() == 255:
#         mask_input = mask_input / 255
#     transform = ResizeLongestSide(256)
#     input_image = transform.apply_image(image)
#     input_image_torch = torch.as_tensor(input_image, device='cuda')
#     transformed_image = input_image_torch[None, None, :]

#         # input_image = sam.preprocess(transformed_image)
#         # original_image_size = image.shape[:2]
#         # input_size = tuple(transformed_image.shape[-2:])

#     return transformed_image
#--------------------------------------------------------------------------------------------------------------------   
    
    
    
    
    
    

class TrainingDataset(Dataset):
    def __init__(self, data_dir, image_size=256, mode='train', requires_name=True, point_num=1, mask_num=1):
        """
        Initializes a training dataset.
        Args:
            data_dir (str): Directory containing the dataset.
            image_size (int, optional): Desired size for the input images. Defaults to 256.
            mode (str, optional): Mode of the dataset. Defaults to 'train'.
            requires_name (bool, optional): Indicates whether to include image names in the output. Defaults to True.
            num_points (int, optional): Number of points to sample. Defaults to 1.
            num_masks (int, optional): Number of masks to sample. Defaults to 5.
        """
        self.image_size = image_size
        self.requires_name = requires_name
        self.point_num = point_num
        self.mask_num = mask_num
        self.pixel_mean = [123.675, 116.28, 103.53]
        self.pixel_std = [58.395, 57.12, 57.375]

        dataset = json.load(open(os.path.join(data_dir, f'image2label_{mode}.json'), "r"))
        self.image_paths = list(dataset.keys())
        self.label_paths = list(dataset.values())
    
    def __getitem__(self, index):
        """
        Returns a sample from the dataset.
        Args:
            index (int): Index of the sample.
        Returns:
            dict: A dictionary containing the sample data.
        """

        image_input = {}
        #try:
            #image = cv2.imread(self.image_paths[index])
        image = Image.open(self.image_paths[index]).convert('RGB')
        image = np.array(image)
        image = (image - self.pixel_mean) / self.pixel_std
        # except:
        #     print(self.image_paths[index])

        h, w, _ = image.shape
        
        transforms = train_transforms(self.image_size, h, w)
        
        
        
        #-----------------------------------------------------
        #transforms_mask = train_transforms(192, h, w)
        #-----------------------------------------------------
        
        
        
        
        masks_list = []
        boxes_list = []
        
        
        
        
        #---------------------
        #masks_input = []
        #---------------------
        
        point_coords_list, point_labels_list = [], []
        #-------------------------------------------------------------------------------------------------------------
        #mask_path = random.choices(self.label_paths[index], k=self.mask_num)
        #print(f"{self.label_paths[index]}======================================,{mask_path}==================================")
        mask_path = [self.label_paths[index][0]] 
        assert  self.label_paths[index][0].split('.')[0].split('/')[-1] == self.image_paths[index].split('.')[0].split('/')[-1]
        #mask_input_path = [self.label_paths[index][1]]
        #pdb.set_trace()
        #assert  self.label_paths[index][1].split('.')[0].split('/')[-1] == self.image_paths[index].split('.')[0].split('/')[-1]
       # print(f"======================================{self.label_paths[index][0]}==================================")
        #-------------------------------------------------------------------------------------------------------------
        
        
        for m in mask_path:
            #pre_mask = cv2.imread(m, 0)

            pre_mask = Image.open(m).convert('L')
            pre_mask = np.array(pre_mask)
            if pre_mask.max() == 255:
                pre_mask = pre_mask / 255

            augments = transforms(image=image, mask=pre_mask)
            image_tensor, mask_tensor = augments['image'], augments['mask'].to(torch.int64)

            boxes = get_boxes_from_mask(mask_tensor)
            point_coords, point_label = init_point_sampling(mask_tensor, self.point_num)
           
            masks_list.append(mask_tensor)
            boxes_list.append(boxes)
            point_coords_list.append(point_coords)
            point_labels_list.append(point_label)
            
            
        #------------------------------------------------------------------------------------------------------------------

#         mask_input = Image.open(mask_input_path[0]).convert('L')
#         mask_input = np.array(mask_input)
#         if mask_input.max() == 255:
#             mask_input = mask_input / 255
#         augments_mask = transforms_mask(image=image, mask=mask_input)
#         mask_input = augments_mask['mask'].unsqueeze(0).unsqueeze(0).float()

        
            
        #-------------------------------------------------------------------------------------------------------------------

        mask = torch.stack(masks_list, dim=0)
        boxes = torch.stack(boxes_list, dim=0)
        point_coords = torch.stack(point_coords_list, dim=0)
        point_labels = torch.stack(point_labels_list, dim=0)

        image_input["image"] = image_tensor.unsqueeze(0)
        image_input["label"] = mask.unsqueeze(1)
        image_input["boxes"] = boxes
        image_input["point_coords"] = point_coords
        image_input["point_labels"] = point_labels
        
        #------------------------------------------------------------------------
        #image_input["mask_inputs"] = mask_input
        
        #------------------------------------------------------------------------

        image_name = self.image_paths[index].split('/')[-1]
        if self.requires_name:
            image_input["name"] = image_name
            return image_input
        else:
            return image_input
    def __len__(self):
        return len(self.image_paths)


def stack_dict_batched(batched_input):
    out_dict = {}
    for k,v in batched_input.items():
        if isinstance(v, list):
            out_dict[k] = v
        else:
            out_dict[k] = v.reshape(-1, *v.shape[2:])
    return out_dict


if __name__ == "__main__":
    train_dataset = TrainingDataset("data_demo", image_size=256, mode='train', requires_name=True, point_num=1, mask_num=1)
    print("Dataset:", len(train_dataset))
    train_batch_sampler = DataLoader(dataset=train_dataset, batch_size=1, shuffle=True, num_workers=4)
    for i, batched_image in enumerate(tqdm(train_batch_sampler)):
        batched_image = stack_dict_batched(batched_image)
        print(batched_image["image"].shape, batched_image["label"].shape)

