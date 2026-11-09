import os
import numpy as np
import cv2
from pathlib import Path

def yolo_txt_to_mask(txt_path, img_width, img_height):

    masks = {}
    nb_instances_per_class = {}
    image_name_per_class = {}

    for dirpath, dirnames, filenames in os.walk(txt_path):

        # Ne traiter que les dossiers nommés "labels"
        if os.path.basename(dirpath) != "labels":
            continue

        for file_name in filenames:
            if not file_name.endswith(".txt"):
                continue

            file_path = os.path.join(dirpath,file_name)
            image_name = file_name.removesuffix(".txt").removesuffix("copie")
            print(image_name)
            
            masks[image_name] = {}

            # mask = np.zeros((img_height, img_width), dtype=np.uint8)

            output_dir = Path("Dataset_unlabeled/masque/")

            mask_full_zeros = np.zeros((img_height, img_width), dtype=np.uint8)
            masks[image_name]["mask_full"] = mask_full_zeros

            with open(file_path, "r") as f:
                
                for line in f:
                    values = list(map(float, line.strip().split()))
                    # print(values)
                    # 0 0.25 0.20 0.75 0.20 0.70 0.80 0.30 0.80
                    # values = [0.0, 0.25, 0.20, 0.75, 0.20, 0.70, 0.80, 0.30, 0.80]

                    class_id = int(values[0])

                    if class_id not in nb_instances_per_class:
                        nb_instances_per_class[class_id] = 0
                        image_name_per_class[class_id] = []  

                    image_name_per_class[class_id].append(image_name)
                    nb_instances_per_class[class_id] += 1

                    if class_id not in masks[image_name]:
                        masks[image_name][class_id] = []
                    
                    # Première valeur : classe
                    coords = values[1:]
                    # extraire toute les valeurs saufs la première pour la classe
                    # coords = [ 0.25, 0.20, 0.75, 0.20, 0.70, 0.80, 0.30, 0.80 ]

                    # Coordonnées normalisées vers pixels
                    points = np.array([
                        [
                            int(coords[i] * img_width), # x = 0.25 * 640 = 160
                            int(coords[i + 1] * img_height) # y = 0.20 * 480 = 96
                        ]
                        for i in range(0, len(coords), 2)
                    ], dtype=np.int32)

                    mask = np.zeros(
                        (img_height, img_width),
                        dtype=np.uint8
                    )

                    # Remplir le polygone en blanc
                    cv2.fillPoly(mask, [points], 255)
                    
                    masks[image_name][class_id].append(np.array(mask))

                    masks[image_name]["mask_full"] += mask

            destination = output_dir / image_name

            if not image_name.endswith(
                ("12", "18", "19", "32", "33", "41", "44", "45", "46", "50", "51", "53", "72", "78", "81", "85", "86", "87", "92", "93", "94", "95")
            ):
                
                mask = masks[image_name]["mask_full"]

                mask = (mask > 0).astype(np.uint8) * 255

                cv2.imwrite(str(destination.with_suffix(".png")), mask)

    return masks, nb_instances_per_class, image_name_per_class

yolo_txt_to_mask("Dataset", 1024, 1024)