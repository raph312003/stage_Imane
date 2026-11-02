import torch
from pipeline.train import fuse_weight
import numpy as np

from sklearn.metrics import (
    jaccard_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score
)

from pipeline.unet import Unet

def test(
    test_dataloader,
    model,
    best_weight,
    f1_score_weight,
    device
):
    all_preds = []
    all_masks = []
    all_input = []

    # Créer trois modèles indépendants
    model1 = Unet().to(device)
    model2 = Unet().to(device)
    model3 = Unet().to(device)

    # Charger les paramètres appris
    model1.load_state_dict(best_weight["patches256"])
    model2.load_state_dict(best_weight["patches512"])
    model3.load_state_dict(best_weight["full_patches"])

    # Passer en mode inférence
    model1.eval()
    model2.eval()
    model3.eval()

    with torch.no_grad():
        for input, mask in test_dataloader:
            input = input.to(device)
            mask = mask.to(device)

            logit1 = model1(input)
            logit2 = model2(input)
            logit3 = model3(input)

            fused_logits = fuse_weight(
                logit_list = [logit1,logit2,logit3],
                weights = f1_score_weight
            )

            probs = torch.sigmoid(fused_logits)
            preds = (probs>=0.5).int()

            all_preds.append(preds.cpu().numpy().ravel())
            all_masks.append(mask.cpu().numpy().ravel())
            all_input.append(input.cpu().numpy().ravel())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_masks).astype(int)

    print("IoU :", jaccard_score(y_true, y_pred, zero_division=0))
    print("Dice :", f1_score(y_true, y_pred, zero_division=0))
    print("Precision :", precision_score(y_true, y_pred, zero_division=0))
    print("Recall :", recall_score(y_true, y_pred, zero_division=0))
    print("Accuracy :", accuracy_score(y_true, y_pred))



           
