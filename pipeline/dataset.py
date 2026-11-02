from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np
import torch
from pipeline.utils import preprocessing_mask, preprocessing_image


class load_dataset(Dataset):
    def __init__(
        self,
        img_dir,
        mask_dir,
        transform = None

    ):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.image = os.listdir(img_dir)
        self.transform = transform 

    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.image[index])
        mask_path = os.path.join(self.mask_dir, self.image[index].replace(".png", "_mask.png"))
        img = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype =np.float32)
        mask[mask == 255.0] = 1.0

        mask = preprocessing_mask(mask)
        img = preprocessing_mask(img)

        # if self.transform is not None:
        #     augmentation = self.transform(image=img, mask=mask)
        #     img = augmentation["image"]
        #     mask = augmentation["mask"]

        # print(img.shape,mask.shape)
      
        return torch.tensor(img), torch.tensor(mask)#.unsqueeze(0).permute(2,1,0)