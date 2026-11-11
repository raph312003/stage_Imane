from pathlib import Path
import cv2

def load_unlabeled_dataset(img_dir):

    img_dir = Path(img_dir)
    output_dir = Path("Dataset/")
    output_dir.mkdir(parents=True, exist_ok=True)

    unlabeled_images = []

    prefixes = (
        "FO 26", "FO 28", "FO 29", "FO 30", "FO 34", "FO 47",
        "FO 55", "FO 60", "FO 61", "FO 64", "FO 67",
        "FO 71", "FO 75", "FO 77", "FO 80", "FO 83",
        "FO 84", "FO 88", "FO 96"
    )

    for folder in img_dir.iterdir():

        if not folder.is_dir():
            continue

        if not folder.name.strip().startswith(prefixes):
            continue

        jpg_images = list(folder.glob("*.jpg"))

        # Images originales et copies
        copies = [
            img for img in jpg_images
            if img.stem.lower().endswith("copie")
        ]

        originals = [
            img for img in jpg_images
            if not img.stem.lower().endswith("copie")
        ]

        for copy in copies:

            original_stem = copy.stem[:-5].rstrip()

            original = next(
                (img for img in originals
                if img.stem.lower() == original_stem.lower()),
                None
            )

            if original is None:
                continue

            # Nom temporaire pour éviter les conflits
            temp = original.with_name(original.stem + "_temp.jpg")

            # Inversion des noms
            original.rename(temp)
            copy.rename(original)
            temp.rename(copy)

            print(f"Inversion effectuée : {original.name} <-> {copy.name}")


load_unlabeled_dataset("Dataset/")