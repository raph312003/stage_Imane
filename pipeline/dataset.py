from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import numpy as np
import torch
from pipeline.utils import preprocessing_mask, preprocessing_image, patch_collate_fn, PatchDataset


class load_dataset(Dataset):
    def __init__(
        self,
        img_dir,
        mask_dir,
        transform = None,
        unlabeled = False

    ):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.image = os.listdir(img_dir)#[:2]
        self.transform = transform 
        self.unlabeled = unlabeled

    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):
        img_name = self.image[index]

        img_path = os.path.join(
            self.img_dir,
            img_name
        )

        if self.unlabeled:
            mask_name = img_name
        else:
            mask_name = img_name.replace(".png", "_mask.png")

        mask_path = os.path.join(
            self.mask_dir,
            mask_name
        )

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

        # print("img",torch.tensor(img).shape)
        # print("mask",torch.tensor(mask).shape)
      
        return torch.tensor(img), torch.tensor(mask)#.unsqueeze(0).permute(2,1,0)

class PseudoLabelDataset(Dataset):

    def __init__(self, images, masks):
        self.images = images
        self.masks = masks

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.images[index], self.masks[index]


def create_dataloader(
    train_dataset,
    val_dataset,
    unlabeled_dataset,
    SCALES,
    train_transform

):
    train_dataloader = {}
    val_dataloader = {}
    unlabeled_dataloader = {}

    for key, value in SCALES.items():

        train_patchs = PatchDataset(
            train_dataset,
            patch_size=value,
            transform = train_transform
        )

        val_patchs = PatchDataset(
            val_dataset,
            patch_size=value,
            transform = train_transform
        )

        unlabeled_patchs = PatchDataset(
            unlabeled_dataset,
            patch_size=value,
            transform = train_transform
        )

        if key == "patches256":
            batch_size = 16
        elif key == "patches512":
            batch_size = 8 # 8
        elif key == "full_patches":
            batch_size = 1


        train_dataloader[key] = DataLoader(train_patchs, batch_size=batch_size, shuffle=True)
        val_dataloader[key] = DataLoader(val_patchs, batch_size=batch_size, shuffle=False, collate_fn=patch_collate_fn)
        unlabeled_dataloader[key] = DataLoader(unlabeled_patchs, batch_size=batch_size,shuffle=False, collate_fn=patch_collate_fn)

    return train_dataloader, val_dataloader, unlabeled_dataloader