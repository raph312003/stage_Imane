import cv2
import numpy as np

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

def preprocessing_mask(image, input_size = (1024,1024)):
    img = cv2.resize(image, input_size)
    return img

def get_patchs(img, patch_size = 256):

        padding = patch_size//4
        list_patchs = []

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
                init_y += padding
                # print(f"x_i = {x_i}, y_i = {y_i}, batch.size = {batch.shape}")
                list_patchs.append(np.array(batch))
                # print(batch)

            init_x+=padding
            init_y = 0
        
        if len(list_patchs) == nb_patchs:
            return list_patchs
        else :
            return KeyError
        
class PatchDataset():
    def __init__(
            self,
            dataset,
            patch_size,
            transform = None
    
        ):
            self.dataset = dataset
            self.transform = transform 
            self.patches = []

            for idx in range (len(self.dataset)):
                image, mask = self.dataset[idx]

                if patch_size is not None:
                    list_image = get_patchs(image, patch_size)
                    list_mask = get_patchs(mask, patch_size)

                    for img_patch, mask_patch in zip(list_image,list_mask):
                        self.patches.append((img_patch,mask_patch))

                else:
                    self.patches.append((image,mask))
                     

    def __len__(self):
        return len(self.patches)
         

    def __getitem__(self, index):
        img, mask = self.patches[index]

        if self.transform is not None:
            augmentation = self.transform(image=img, mask=mask)
            img = augmentation["image"]
            mask = augmentation["mask"]

        return img, mask.unsqueeze(0)




