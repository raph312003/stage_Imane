import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data import random_split

from pipeline.dataset import load_dataset, create_dataloader
from pipeline.utils import PatchDataset
from pipeline.train import training_all, compute_fusion_weights 
from pipeline.test import test, graph_loss_epoch, display_result, generate_pseudo_label

from argparse import ArgumentParser
import os




def main():
    # Parse command-line arguments
    parser = ArgumentParser(description="Train semi-supervised model.")


    parser.add_argument(
        "--google_drive",
        action="store_true",
        help="Use Google Drive paths"
    )
    args = parser.parse_args()  # Get args.training_config (Path/None) 
    google_drive_connect = args.google_drive

    if google_drive_connect:
        dataset_path_labeled = "/content/drive/MyDrive/PHD/Datasets/Dataset_Imane"
        dataset_path_unlabeled = "/content/drive/MyDrive/PHD/Datasets/Dataset_Unlabeled"
        general_path = "/content/drive/MyDrive/PHD/"
    else :
        dataset_path_labeled = "Dataset_Imane"
        dataset_path_unlabeled = "Dataset_Unlabeled"
        general_path = ""

    NUM_ITERATIONS = 3

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


    dataset = load_dataset(
       img_dir=f"{dataset_path_labeled}/images/",
       mask_dir=f"{dataset_path_labeled}/masque/",
       transform=None,
       unlabeled=False
    )

    
    unlabeled_dataset = load_dataset(
        img_dir=f"{dataset_path_unlabeled}/images/",
        mask_dir=f"{dataset_path_unlabeled}/masque/",
        transform=None,
        unlabeled=True
    )

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataloader, val_dataloader, unlabeled_dataloader = create_dataloader(
        train_dataset,
        val_dataset,
        unlabeled_dataset,
        SCALES,
        train_transform
    )

    # 1 st training already satisfy
    # best_weights, best_f1_score, training_loss, val_f1_score = training_all(
    #     train_dataloader,
    #     val_dataloader,
    #     device,
    #     epoch=1000, # 1000
    #     early_stopping=10, # 10
    #     dice_weight=0.5,
    #     best_weights ={},
    #     learning_rate=1e-4,
    #     path_drive = general_path
    # )

    # In case there is a problem with the testing part
    best_f1_score = {
        "patches256": 0.4113,
        "patches512": 0.3614,
        "full_patches": 0.4431
    }
      

    best_weights = {}

    for name, value in SCALES.items():
        
        best_weights[name] = torch.load(
            f"model_weights/{name}_best.pth",
            map_location=device,
            weights_only=True,
        )

    # Plotting graph for train and val loss
    # iteration = 0
    # graph_loss_epoch(training_loss, val_f1_score, iteration)

    f1_score_weight = compute_fusion_weights(best_f1_score)
            
    
    all_preds, all_masks, all_input = test(
        val_dataloader,
        best_weights,
        f1_score_weight,
        device
    )
    
    # SELF-TRAINING ITERATION

    for iteration in range(1, NUM_ITERATIONS+1):

        print(f"\n===== SELF-TRAINING ITERATION {iteration + 1} =====")

        f1_score_weight = compute_fusion_weights(best_f1_score)

        # Generate pseudo-labels
        print(" 1) Generate pseudo-labels")
        pseudo_dataset = generate_pseudo_label(
            unlabeled_dataloader,
            best_weights,
            f1_score_weight,
            device,
            iteration=iteration
        )

        # Combine real and pseudo-labeled data
        print(" 2) ConcatDataset")
        combined_dataset = ConcatDataset([
            train_dataset,
            pseudo_dataset
        ])

        # Create new dataloaders
        print(" 3) create_dataloader")
        train_dataloader, val_dataloader, unlabeled_dataloader = create_dataloader(
            combined_dataset,
            val_dataset,
            unlabeled_dataset,
            SCALES,
            train_transform
        )

        # Fine-tuning
        print(" 4) fine-tuning")
        best_weights, best_f1_score, training_loss, val_f1_score = training_all(
            train_dataloader,
            val_dataloader,
            device,
            epoch=1000, # 1000
            early_stopping=10, # 10
            dice_weight=0.5,
            best_weights ={},
            learning_rate=1e-5,
            path_drive = general_path
        )

        graph_loss_epoch(training_loss, val_f1_score, iteration, path_drive=general_path)
        
        f1_score_weight = compute_fusion_weights(best_f1_score)
        

        all_preds, all_masks, all_input = test(
                val_dataloader,
                best_weights,
                f1_score_weight,
                device
            )

    # Plotting results 
    display_result(all_preds, all_masks, all_input, path_drive=general_path)

    
    


   

if __name__ == "__main__":
    main()