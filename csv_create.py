import csv
import os
import shutil
import random

def create_image_csv(train_csv_path, test_csv_path):
    # Generate file names
    training_pngs = [f"{i:02d}_training.png" for i in range(21, 41)]
    training_jpgs = [f"IDRiD_{i:02d}.jpg" for i in range(1, 55)]

    test_pngs = [f"{i:02d}_test.png" for i in range(1, 21)]
    test_jpgs = [f"IDRiD_{i:02d}.jpg" for i in range(55, 82)]

    # Ensure parent directory exists
    os.makedirs(os.path.dirname(train_csv_path), exist_ok=True)
    os.makedirs(os.path.dirname(test_csv_path), exist_ok=True)

    # Write to train CSV
    with open(train_csv_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["image_name"])
        for img_name in training_pngs + training_jpgs:
            writer.writerow([img_name])

    # Write to test CSV
    with open(test_csv_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["image_name"])
        for img_name in test_pngs + test_jpgs:
            writer.writerow([img_name])

    print(f"CSV files saved:\n- {train_csv_path}\n- {test_csv_path}")

def copy_images_with_log(source_dir, target_dir, num_images, log_txt_path, extensions=(".jpg", ".png", ".jpeg")):
    # Create target directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)

    # Get list of valid image files
    all_images = [f for f in os.listdir(source_dir) if f.lower().endswith(extensions)]
    
    if num_images > len(all_images):
        raise ValueError(f"Requested {num_images} images, but only {len(all_images)} found in {source_dir}")

    # Randomly sample image names
    selected_images = random.sample(all_images, num_images)

    # Copy and log
    with open(log_txt_path, 'w') as log_file:
        for img_name in selected_images:
            src_path = os.path.join(source_dir, img_name)
            dst_path = os.path.join(target_dir, img_name)
            shutil.copy(src_path, dst_path)
            log_file.write(f"{img_name}\n")

    print(f"Copied {num_images} images to {target_dir} and saved log to {log_txt_path}")

if __name__ == '__main__':
    create_image_csv(
    train_csv_path="/home/local/ASURITE/xdong64/Desktop/ISBI/TPOT/TPOT-DA/train_images.csv",
    test_csv_path="/home/local/ASURITE/xdong64/Desktop/ISBI/TPOT/TPOT-DA/test_images.csv"
)