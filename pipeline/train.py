import os
import copy

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from pipeline.unet import Unet
import numpy as np
import segmentation_models_pytorch as smp

class DiceFocalLoss(nn.Module):
    def __init__(
        self,
        dice_weight = 0.5,
        alpha = 0.25,
        gamma = 2.0,
    ):
        super().__init__()
        self.dice_weight = dice_weight
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        # Permettre opération mathématiques
        targets = targets.float()

        # convertir valeur en probabilité
        probs = torch.sigmoid(logits)

        dims = (1, 2, 3) # car shape (8,1,256,256) -> résultat 8 intersections avec 1 valeur
        intersection = (probs*targets).sum(dim=dims)
        denominator = probs.sum(dim=dims) + targets.sum(dim=dims)

        dice = (2*intersection + 1e-6) / (denominator + 1e-6)

        dice_loss = (1 - dice).mean()

        # focal loss
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        pt = torch.exp(-bce)

        alpha_t = (
            self.alpha * targets
            + (1 - self.alpha) * (1 - targets)
        )

        focal_loss = (
            alpha_t * (1 - pt).pow(self.gamma) * bce
        ).mean()

        return (
            self.dice_weight * dice_loss + (1-self.dice_weight) * focal_loss
        )


def train_one_epoch(
    train_dataloader,
    model,
    device,
    criterion,
    optimizer
):
    model.train()
    total_loss = 0.0

    for image, mask, _, _ in train_dataloader:
        image = image.to(device)
        mask = mask.to(device)

        logits = model(image)

        loss = criterion(logits, mask)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        train_dataloader.set_postfix(loss=loss.item())

    return total_loss/len(train_dataloader)

def val_one_epoch(
    val_dataloader,
    model,
    device,
    criterion,
):
    model.eval()
    total_loss = 0
    all_preds = []
    all_masks = []
    with torch.no_grad():
        for image, mask, _, _ in val_dataloader:
            image = image.to(device)
            mask = mask.to(device)
    
            logits = model(image)
    
            loss = criterion(logits, mask)
    
            total_loss += loss.item()
            val_dataloader.set_postfix(loss=loss.item())

            # Transformation des logits en prédictions binaires
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).int()

            preds = preds.cpu().numpy().flatten()
            mask = mask.cpu().numpy().flatten()

            all_preds.append(preds)
            all_masks.append(mask)
        
        # print(len(all_masks))
        # print(len(all_preds))
        y_pred = np.concatenate(all_preds)
        y_true = np.concatenate(all_masks).astype(int)

        f1 = f1_score(y_true, y_pred, zero_division=0)
    
    return total_loss/len(val_dataloader), f1


def compute_fusion_weights(f1_score):
    f1_score_weight = []
    total = sum(f1_score.values())

    for key, value in f1_score.items():
        f1_score_weight.append(value/total) 

    return f1_score_weight

def fuse_weight(logit_list, weights):

    if len(logit_list) != len(weights):
        raise ValueError("Nombre de logits différent du nombre de poids.")

    if len(logit_list) == 0:
        raise ValueError("Liste de logits vide.")

    fused = torch.zeros_like(logit_list[0])

    for logit, weight in zip(logit_list, weights):
        if logit.shape != fused.shape:
            raise ValueError("Les logits doivent avoir la même shape.")

        fused += logit * weight

    return fused

def logits_to_mask(fused_logits, threshold=0.5):
    probabilities = torch.sigmoid(fused_logits)
    return (probabilities >= threshold).float()


def training_all(
    train_dataloader,
    val_dataloader,
    device,
    epoch,
    early_stopping,
    dice_weight,
    best_weights,
    learning_rate
):
    best_f1_score = {}
    updated_weights = {}
    dict_training_loss ={}
    dict_val_f1_score={}    

    for name, train_loader in train_dataloader.items():
        training_loss = []
        val_f1_score = []
        
        print(f"name model {name}")
        checkpoint_path = f"model_weights/{name}_best.pth"

        best_f1 = float("-inf")
        patience_counter = 0
    
        # model = Unet().to(device)

        # Initialisation des poids de l'encoder Resnet avec le spoids Imagenet
        # Tous le spoids du réseau est entraînable
        # Pour geler l'encoder on peut utiliser 
        # for param in model.encoder.parameters():
        #   param.requires_grad = False
        model = smp.UnetPlusPlus(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1
        )

        if name in best_weights:
            print("fine tuning")
            model.load_state_dict(best_weights[name])

            # Gel de l'encoder
            for param in model.encoder.parameters():
                param.requires_grad = False

            # Optimisation du decoder uniquement
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=learning_rate
            )
        else :
            print("First training")
            # for name_pa, param in model.named_parameters():
            #     print(name_pa, param.requires_grad)

            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        criterion = DiceFocalLoss(dice_weight=dice_weight)


        for i in range(epoch):

            train_bar = tqdm(
               train_loader,
               desc=f"Epoch{i+1}/{epoch}"
            )

            train_loss = train_one_epoch(
                train_bar,
                model,
                device,
                criterion,
                optimizer
            )
            training_loss.append(train_loss)

            val_bar = tqdm(
               val_dataloader[name],
               desc=f"Epoch{i+1}/{epoch}"
            )

            _, f1_score = val_one_epoch(
                val_bar,
                model,
                device,
                criterion,
            )
            val_f1_score.append(f1_score)

            print(f"Name {name} | Epoch {i+1} | loss {train_loss:.4f} | f1_score {f1_score:.4f}")

            
            if f1_score > best_f1:
                best_f1 = f1_score
                patience_counter = 0

                # updated_weights[name] = copy.deepcopy(model.state_dict())
                os.makedirs(f"model_weights/", exist_ok=True)
                torch.save(
                    model.state_dict(),
                    checkpoint_path
                    
                )

            else:
                patience_counter+=1

            if patience_counter >= early_stopping:
                print(f"Early stopping pour {name}")
                break

        model.load_state_dict(
            torch.load(
                checkpoint_path,
                map_location=device,
                weights_only=True,
            )
        )

        updated_weights[name] = model.state_dict()
        best_f1_score[name] = best_f1
        dict_training_loss[name] = training_loss
        dict_val_f1_score[name] = val_f1_score

        print(f"Meilleur F1 pour {name} : {best_f1:.4f}")
        

    return (updated_weights, best_f1_score, dict_training_loss, dict_val_f1_score)

        

