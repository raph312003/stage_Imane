from pathlib import Path
import os
import matplotlib.pyplot as plt
import cv2
from pathlib import Path

def load_unlabeled_dataset(img_dir, img_width, img_height):

    img_dir = Path(img_dir)
    unlabeled_images = []

    # 1. Retrieve unlabeled images
    for folder in img_dir.iterdir():

        if not folder.is_dir():
            continue

        if not folder.name.strip().startswith("FO "):
            continue

        images = list(folder.glob("*"))

        jpg_images = [
            img for img in images
            if img.is_file() and img.suffix.lower() == ".jpg"
        ]

        annote_images = [
            img for img in jpg_images
            if img.stem.lower().endswith("annote")
        ]

        copie_images = [
            img for img in jpg_images
            if img.stem.lower().endswith("copie")
        ]

        if annote_images:

            unlabeled_images.extend([
                img for img in jpg_images
                if img not in annote_images
            ])

        elif copie_images:

            unlabeled_images.extend(copie_images)

    unlabeled_images = sorted(unlabeled_images)

    # 2. Save images
    output_dir = Path("Dataset_unlabeled/images")

    for img_path in unlabeled_images:

        img_name = img_path.stem.removesuffix("copie")

        if not img_name.endswith(
            ("12", "18", "19","32", "33", "41", "44", "45", "46", "50", "51", "53", "72", "78", "81", "85", "86","87", "92", "93", "94", "95")
        ):

            destination = output_dir / img_name #f"{img_name}.jpg"

            # Lecture de l'image
            image = cv2.imread(str(img_path))

            # Resize
            image = cv2.resize(
                image,
                (img_width, img_height),
                interpolation=cv2.INTER_AREA
            )

            # Sauvegarde
            cv2.imwrite(str(destination.with_suffix(".png")), image)

        print(f"{len(unlabeled_images)} images sauvegardées.")

    return unlabeled_images


print(os.getcwd())
a = load_unlabeled_dataset("Dataset/", 1024, 1024)
print(a)
print(len(a))