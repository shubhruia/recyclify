import os
import shutil
import random
from pathlib import Path
from tqdm import tqdm

SOURCE_DIR = "images"
TARGET_DIR = "data"
TRAIN_SPLIT = 0.8
VALID_IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png"]

CATEGORY_MAP = {
    "plastic": [
        "plastic_cup_lids", "plastic_detergent_bottles", "plastic_food_containers",
        "plastic_shopping_bags", "plastic_soda_bottles", "plastic_straws",
        "plastic_trash_bags", "plastic_water_bottles", "disposable_plastic_cutlery"
    ],
    "paper": [
        "cardboard_boxes", "cardboard_packaging", "magazines", "newspaper",
        "office_paper", "paper_cups"
    ],
    "glass": [
        "glass_beverage_bottles", "glass_cosmetic_containers", "glass_food_jars"
    ],
    "metal": [
        "aerosol_cans", "aluminum_food_cans", "aluminum_soda_cans", "steel_food_cans"
    ],
    "organic": [
        "eggshells", "food_waste", "tea_bags", "coffee_grounds"
    ],
    "trash": [
        "shoes", "clothing", "styrofoam_cups", "styrofoam_food_containers"
    ]
}

def prepare_dirs():
    for split in ["train", "val"]:
        for broad_cat in CATEGORY_MAP:
            os.makedirs(os.path.join(TARGET_DIR, split, broad_cat), exist_ok=True)

def find_domain_folders(base_path):
    found = []
    for folder in os.listdir(base_path):
        if folder.lower() in ["default", "real world", "real_world", "realworld"]:
            found.append(os.path.join(base_path, folder))
    return found

def process_dataset():
    prepare_dirs()
    total_copied = 0
    total_available = 0

    print("\nScanning dataset...\n")

    for broad_cat, subcats in CATEGORY_MAP.items():
        images = []

        for subcat in subcats:
            subcat_path = os.path.join(SOURCE_DIR, subcat)
            if not os.path.exists(subcat_path):
                print(f"Folder not found: {subcat_path}")
                continue

            domain_paths = find_domain_folders(subcat_path)
            if not domain_paths:
                print(f"No domain folders (default/real world) found in: {subcat_path}")
                continue

            for domain_path in domain_paths:
                domain_name = os.path.basename(domain_path).replace(" ", "_").lower()
                all_files = os.listdir(domain_path)
                valid_imgs = [
                    os.path.join(domain_path, f)
                    for f in all_files
                    if Path(f).suffix.lower() in VALID_IMAGE_EXTENSIONS
                ]
                for img_path in valid_imgs:
                    new_name = f"{subcat}_{domain_name}_{os.path.basename(img_path)}"
                    images.append((img_path, new_name))

        if not images:
            print(f"No images found for {broad_cat}")
            continue

        total_available += len(images)
        random.shuffle(images)
        split_index = int(len(images) * TRAIN_SPLIT)
        train_imgs = images[:split_index]
        val_imgs = images[split_index:]

        for img_path, new_name in tqdm(train_imgs, desc=f"→ train/{broad_cat}", unit="img"):
            target_path = os.path.join(TARGET_DIR, "train", broad_cat, new_name)
            shutil.copy(img_path, target_path)
            total_copied += 1

        for img_path, new_name in tqdm(val_imgs, desc=f"→ val/{broad_cat}", unit="img"):
            target_path = os.path.join(TARGET_DIR, "val", broad_cat, new_name)
            shutil.copy(img_path, target_path)
            total_copied += 1

        print(f"Copied: {len(train_imgs)} train + {len(val_imgs)} val images for '{broad_cat}'")

    print(f"\nDone copying!")
    print(f"Total images found: {total_available}")
    print(f"Total images copied: {total_copied}")

if __name__ == "__main__":
    process_dataset()
