import cv2
import numpy as np

import torch
# 3. Shades of Gray avec p = 6
def shades_of_gray(image, p=6):
    img = image.astype(np.float32) / 255.0

    # Estimer la dominante de couleur
    illuminant = np.mean(img ** p, axis=(0, 1)) ** (1 / p)

    # Rééquilibrer les trois canaux
    target = np.mean(illuminant)
    gains = target / (illuminant + 1e-8)

    corrected = img * gains

    return np.clip(corrected * 255, 0, 255).astype(np.uint8)


def preprocessing_image(image, input_size = (1024,1024)):
    # Resizing
    img = cv2.resize(image, input_size)

    # convertir rgb en cielab
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    # CLAHE 
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )
    L_ameliore = clahe.apply(l)

    # Go back to rgb
    new_lab = cv2.merge((L_ameliore,a,b))
    clahe_img = cv2.cvtColor(new_lab, cv2.COLOR_LAB2RGB)

    # Shade of gray algorithm
    final_img = shades_of_gray(clahe_img)

    return final_img

def preprocessing_mask(image, input_size = (512,512)): # 512,512
    img = cv2.resize(image, input_size)
    return img

def patch_collate_fn(batch):
    images = torch.stack([item[0] for item in batch])
    masks = torch.stack([item[1] for item in batch])

    positions = [item[2] for item in batch]
    indices = [item[3] for item in batch]

    return images, masks, positions, indices

def get_patchs(img, patch_size = 256):

        padding = patch_size//4
        list_patchs = []
        positions = []

        x = img.shape[0]
        y = img.shape[1]
        init_x = 0
        init_y = 0

        nb_patchs_y = (y-patch_size)//padding + 1
        nb_patchs_x = (x-patch_size)//padding + 1

        nb_patchs = nb_patchs_y * nb_patchs_y
        

        for _ in range(nb_patchs_x):
            for _ in range(nb_patchs_y):
                batch = img[init_x : patch_size + init_x, init_y : patch_size + init_y]
                positions.append((init_x,init_y))
                init_y += padding
                # print(f"x_i = {x_i}, y_i = {y_i}, batch.size = {batch.shape}")
                list_patchs.append(np.array(batch))
                # print(batch)

            init_x+=padding
            init_y = 0
        
        if len(list_patchs) == nb_patchs:
            return list_patchs, positions
        else :
            return KeyError

def get_patchs_to_img(
    patchs,
    img_shape,
    positions
):

    H, W = img_shape[-2:]
    
    # (1024, 1024, 3)
    img_reconstructed = np.zeros(
        (H,W),
        dtype=np.float32
    )
    
    # (1024, 1024, 3) -> (1024, 1024)
    count = np.zeros(
        (H,W),
        dtype=np.float32
    )

    # print("position", positions)
    for patch, (init_x, init_y) in zip(patchs, positions):
        # print("init_x, init_y", init_x, init_y)

        # Pour 1 pixel somme de plusieurs pixels Ex : pixel_0 = 0.9 + 0.9 + 0.9 = 2.7
        if isinstance(patch, torch.Tensor):
            patch = patch.detach().cpu().numpy()

        # (1, 1, 128, 128) ou (1, 128, 128) -> (128, 128)
        patch = np.squeeze(patch)

        h, w = patch.shape

        img_reconstructed[
            init_x:init_x+h,
            init_y:init_y+w
        ] += patch

        # Ajoute 1 pour chaque pixel couvert  Ex : pixel_0 = 1+1+1 = 3
        count[
            init_x:init_x+h,
            init_y:init_y+w
        ] += 1

    # pas de division par 0
    count[count == 0] = 1
    img_reconstructed /= count

    return torch.tensor(img_reconstructed)

        
class PatchDataset():
    def __init__(
            self,
            dataset,
            patch_size,
            transform = None
    
        ):
            self.dataset = dataset
            self.transform = transform 
            self.patches = [] # stocker tous les images et tous les patchs problème en mémoire
            # stocker :
            # - l'image originale
            # - le mask originale
            # - les coordonnées des patches 
            # - l'indice de l'image
            #   (idx, y, x) -> stocker dans self.patches

            for idx in range (len(self.dataset)):
                image, mask = self.dataset[idx]

                if patch_size is not None:
                    list_image, positions = get_patchs(image, patch_size)
                    list_mask, _ = get_patchs(mask, patch_size)

                    for img_patch, mask_patch, position in zip(list_image, list_mask, positions):
                        self.patches.append((img_patch,mask_patch, position, idx))

                else:
                    # position None car 1 image
                    self.patches.append(
                        (np.array(image),np.array(mask), (1,1), idx)
                    )
# Stocke tous les patches en mémoire                   

    def __len__(self):
        return len(self.patches)
         

    def __getitem__(self, index):
        # Récupérer images et masques
        # création du patch en fonction de l'indice
        img, mask, position, idx = self.patches[index]
        
        if self.transform is not None:
            augmentation = self.transform(image=img, mask=mask)
            img = augmentation["image"]
            mask = augmentation["mask"]
        print(img.shape, mask.unsqueeze(0).shape, position,idx )
        return img, mask.unsqueeze(0), position, idx

# ne pas stocker tous les patches en mémoire
# torch.Size([3, 128, 128]) torch.Size([1, 128, 128]) (0, 96) 22


