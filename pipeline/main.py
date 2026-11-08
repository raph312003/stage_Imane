import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch
from torch.utils.data import DataLoader
from torch.utils.data import random_split

from pipeline.dataset import load_dataset
from pipeline.utils import PatchDataset, patch_collate_fn
from pipeline.train import training_all, compute_fusion_weights 
from pipeline.test import test, graph_loss_epoch, display_result
import os




def main():

    SCALES = {
        "patches256": 128, # 128
        "patches512": 256, # 256
        "full_patches": None # 512
    }   

    train_transform = A.Compose(
        [  

            A.Rotate(limit=35, p=1.0),
            # Scaling : facteur entre 0.9 et 1.1
            A.Affine(
                scale=(0.9, 1.1),
                translate_percent=(-0.1, 0.1),
                p=0.5,
            ),
            # Brightness +- 10%, contrast +- 10%
            A.RandomBrightnessContrast(
                brightness_limit=0.1,
                contrast_limit=0.1,
                p=0.5,
            ),

            # Color jitter +- 5%
            A.ColorJitter(
                brightness=0.05,
                contrast=0.05,
                saturation=0.05,
                hue=0.05,
                p=0.5,
            ),

            # Elastic deformation : alpha=120, sigma=8
            A.ElasticTransform(
                alpha=120,
                sigma=8,
                p=0.5,
            ),
            A.HorizontalFlip(p=0.5), 
            A.VerticalFlip(p=0.5),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),        
        ],
    )

    
    #print(os.getcwd())

    dataset = load_dataset(
        img_dir= "Dataset_Imane/images/",
        mask_dir= "Dataset_Imane/masque/",
        transform = None
    )

    #dataset_path = "/content/drive/MyDrive/PHD/Datasets/Dataset_Imane"

    #dataset = load_dataset(
     #   img_dir=f"{dataset_path}/images/",
     #   mask_dir=f"{dataset_path}/masque/",
     #   transform=train_transform
    #)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

    train_dataloader = {}
    val_dataloader = {}
    # test_dataloader = {}

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

        if key == "patches256":
            batch_size = 16
        elif key == "patches512":
            batch_size = 8 # 8
        elif key == "full_patches":
            batch_size = 1


        train_dataloader[key] = DataLoader(train_patchs, batch_size=batch_size, shuffle=True)
        val_dataloader[key] = DataLoader(val_patchs, batch_size=batch_size, shuffle=False, collate_fn=patch_collate_fn)
        # test_dataloader[key] = DataLoader(val_patchs, batch_size=batch_size, shuffle=False)

    

    best_weights, best_f1_score, training_loss, val_f1_score = training_all(
        train_dataloader,
        val_dataloader,
        device,
        epoch=1000, # 1000
        early_stopping=10, # 10
        dice_weight=0.5
    )

    # f1_score_weight = compute_fusion_weights(best_f1_score)
    

    # all_preds, all_masks, all_input = test(
    #     val_dataloader,
    #     best_weights,
    #     f1_score_weight,
    #     device
    # )

    # display_result(all_preds, all_masks, all_input)
    

    graph_loss_epoch(training_loss, val_f1_score)


   

if __name__ == "__main__":
    main()