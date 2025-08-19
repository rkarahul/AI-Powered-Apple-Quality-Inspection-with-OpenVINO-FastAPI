import os
import random
import shutil

# Function to split the data into training and validation sets for each class
def split_data_by_class(image_folder, output_train_folder, output_valid_folder, split_ratio=0.8):
    # Get the list of class subfolders (e.g., 'NORM', 'TVS')
    class_folders = [folder for folder in os.listdir(image_folder) if os.path.isdir(os.path.join(image_folder, folder))]

    # Iterate over each class folder
    for class_folder in class_folders:
        # Get the list of images for this class
        class_path = os.path.join(image_folder, class_folder)
        all_images = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]

        # Shuffle the images to ensure randomness
        random.shuffle(all_images)

        # Calculate the number of images to use for training
        train_size = int(len(all_images) * split_ratio)
        
        # Split the data into training and validation
        train_images = all_images[:train_size]
        valid_images = all_images[train_size:]

        # Create the output class directories (train/valid) if they do not exist
        train_class_folder = os.path.join(output_train_folder, class_folder)
        valid_class_folder = os.path.join(output_valid_folder, class_folder)

        os.makedirs(train_class_folder, exist_ok=True)
        os.makedirs(valid_class_folder, exist_ok=True)

        # Move images to the training folder
        for image in train_images:
            src_path = os.path.join(class_path, image)
            dst_path = os.path.join(train_class_folder, image)
            shutil.copy(src_path, dst_path)

        # Move images to the validation folder
        for image in valid_images:
            src_path = os.path.join(class_path, image)
            dst_path = os.path.join(valid_class_folder, image)
            shutil.copy(src_path, dst_path)

        # Print the results for each class
        print(f"Class: {class_folder}")
        print(f"Total images: {len(all_images)}")
        print(f"Training images: {len(train_images)}")
        print(f"Validation images: {len(valid_images)}")
        print()

# Define your paths
image_folder = r"dataset"  # Path to your original folder with class subfolders (NORM, TVS, etc.)
output_train_folder = r"train"  # Path to save training images
output_valid_folder = r"valid"  # Path to save validation images

# Split the data with 90% for training and 10% for validation
split_data_by_class(image_folder, output_train_folder, output_valid_folder, split_ratio=0.9)
