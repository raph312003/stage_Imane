import torch
from pipeline.train import fuse_weight
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    jaccard_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score
)

from pipeline.unet import Unet
from pipeline.utils import get_patchs_to_img, patch_collate_fn

def predict_patches(model, loader, device):

    predictions = {}

    model.eval()

    with torch.no_grad():

        for inputs, masks, positions, indices in loader:
            # print("positions",positions)

            inputs = inputs.to(device)

            logits = model(inputs)

            for logit, mask, position, idx in zip(
                logits, masks, positions, indices
            ):
                # print("position",position)
                
                idx = idx.item() if torch.is_tensor(idx) else int(idx)

                if idx not in predictions:
                    predictions[idx] = {
                        "logits": [],
                        "masks": [],
                        "positions": []
                    }

                predictions[idx]["logits"].append(
                    logit.unsqueeze(0)
                )

                predictions[idx]["masks"].append(mask.unsqueeze(0))
                predictions[idx]["positions"].append(position)

    return predictions

def test(
    test_dataloader,
    best_weight,
    f1_score_weight,
    device
):

    names = ["patches256", "patches512", "full_patches", "fusion"]

    all_preds = {name: [] for name in names}
    all_masks = {name: [] for name in names}
    all_input = {name: [] for name in names}

    # Initialisation des modèles
    model1 = Unet().to(device)
    model2 = Unet().to(device)
    model3 = Unet().to(device)

    model1.load_state_dict(best_weight["patches256"])
    model2.load_state_dict(best_weight["patches512"])
    model3.load_state_dict(best_weight["full_patches"])

    # Prédictions des modèles à patches
    pred256 = predict_patches(
        model1, test_dataloader["patches256"], device
    )

    pred512 = predict_patches(
        model2, test_dataloader["patches512"], device
    )

    # Prédictions du modèle full
    model3.eval()

    full_predictions = {}
    full_inputs = {}
    full_masks = {}

    with torch.no_grad():

        for inputs, masks, _, indices in test_dataloader["full_patches"]:

            inputs = inputs.to(device)

            logits = model3(inputs)

            for logit, img, mask, idx in zip(
                logits, inputs, masks, indices
            ):
                idx = idx.item() if torch.is_tensor(idx) else int(idx)

                full_predictions[idx] = logit.unsqueeze(0)
                full_inputs[idx] = img.unsqueeze(0)
                full_masks[idx] = mask.unsqueeze(0).to(device)

    # Fusion et reconstruction image par image
    for idx in full_predictions:

        input_full = full_inputs[idx]
        mask_full = full_masks[idx]

        # Reconstruction des logits 256 liste -> array
        logits256 = torch.cat(
            pred256[idx]["logits"], dim=0
        ).to(device)

        image_full_256 = get_patchs_to_img(
            logits256,
            input_full.shape,
            pred256[idx]["positions"]
        )

        # Reconstruction des logits 512 liste -> array
        logits512 = torch.cat(
            pred512[idx]["logits"], dim=0
        ).to(device)

        image_full_512 = get_patchs_to_img(
            logits512,
            input_full.shape,
            pred512[idx]["positions"]
        )

        # Logits du modèle full
        logits_full = full_predictions[idx].squeeze(0).squeeze(0)

        # Fusion des logits reconstruits
        fused_logits = fuse_weight(
            logit_list=[
                image_full_256,
                image_full_512,
                logits_full
            ],
            weights=f1_score_weight
        )

        # Binarisation des prédictions
        predictions = [
            image_full_256,
            image_full_512,
            logits_full,
            fused_logits
        ]

        for name, logits in zip(names, predictions):

            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).int()

            all_preds[name].append(preds.cpu().numpy())
            all_masks[name].append(mask_full.cpu().numpy())
            all_input[name].append(input_full.cpu().numpy())

    # Concaténation
    for name in names:

        all_preds[name] = np.stack(
            all_preds[name], axis=0
        )

        all_masks[name] = np.concatenate(
            all_masks[name], axis=0
        )[:, 0]

        all_input[name] = np.concatenate(
            all_input[name], axis=0
        )

    # Calcul des métriques
    for name in names:

        print("Predictions shape:", all_preds[name].shape)
        print("Masks shape:", all_masks[name].shape)

        y_pred = all_preds[name].ravel()
        y_true = (all_masks[name] > 0.5).astype(int).ravel()

        print(f"\n--- {name} ---")

        print("IoU :", jaccard_score(
            y_true, y_pred, zero_division=0
        ))

        print("Dice :", f1_score(
            y_true, y_pred, zero_division=0
        ))

        print("Precision :", precision_score(
            y_true, y_pred, zero_division=0
        ))

        print("Recall :", recall_score(
            y_true, y_pred, zero_division=0
        ))

        print("Accuracy :", accuracy_score(y_true, y_pred))

    return all_preds, all_masks, all_input


def display_result(all_preds, all_masks, all_input):
    
    names = ["patches256", "patches512", "full_patches", "fusion"]

    for name in names:
        fig, axes = plt.subplots(3, all_preds[name].shape[0], figsize=(65,60))

        preds = all_preds[name]
        masks = all_masks[name]
        inputs = all_input[name]

        for i in range(preds.shape[0]):

            axes[0][i].imshow(preds[i], cmap = "gray")
            axes[0][0].set_title("pred")
            axes[0][i].axis("off")

            axes[1][i].imshow(masks[i], cmap = "gray")
            axes[1][0].set_title("pred")  
            axes[1][i].axis("off")

            axes[2][i].imshow(inputs[i].transpose(1, 2, 0), cmap = "gray")
            axes[2][0].set_title("input")  
            axes[2][i].axis("off")

        plt.tight_layout()
        plt.savefig(f"images/{name}.png")
        plt.close()  

def graph_loss_epoch(list_avg_train_loss, list_avg_val_loss):

    fig, ax = plt.subplots(figsize=(12, 7))

    for name, train_losses in list_avg_train_loss.items():

        # Train
        ax.plot(
            range(1, len(train_losses) + 1),
            train_losses,
            label=f"{name} - Train"
        )

        # Validation
        val_losses = list_avg_val_loss[name]

        ax.plot(
            range(1, len(val_losses) + 1),
            val_losses,
            linestyle="--",
            label=f"{name} - Validation"
        )

    ax.set_title("Training and validation losses")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Average loss")

    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("images/all_loss_curves.png")
    plt.show()
    plt.close()



           
