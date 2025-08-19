import os
from ultralytics import YOLO
from PIL import Image

# Load the YOLOv8 classification model
model = YOLO('best.pt')

# Input folder path (change this path to the location of your folder)
image_folder = r'wetransfer_image_20250120_055039_car-png_2025-01-22_1720'  # e.g., 'images/'

# Output folder path to save class-wise results
output_folder = r'wetransfer_image_20250120_055039_car-png_2025-01-22_1720_output' # e.g., 'images/

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Confidence threshold (90%)
CONFIDENCE_THRESHOLD = 0.9

# Get the list of image files in the folder
image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]

# Iterate over each image in the folder
for image_file in image_files:
    # Construct the full path of the image
    image_path = os.path.join(image_folder, image_file)
    
    # Open the image
    image = Image.open(image_path)
    
    # Perform inference
    results = model(image)
    
    # Get class names
    class_names = model.names
    
    # Get the top-1 prediction (most confident class)
    top_prediction = results[0].probs.top1  # Index of the most confident class
    confidence = results[0].probs.top1conf.item()  # Confidence of the top prediction
    
    # Check if the confidence meets the threshold
    if confidence >= CONFIDENCE_THRESHOLD:
        class_name = class_names[top_prediction]
        
        # Create a folder for the predicted class if it doesn't exist
        class_folder = os.path.join(output_folder, class_name)
        os.makedirs(class_folder, exist_ok=True)
        
        # Save the image into the corresponding class folder
        output_path = os.path.join(class_folder, image_file)
        image.save(output_path)
        
        # Print progress
        print(f"Image '{image_file}' classified as '{class_name}' with confidence {confidence:.2f}. Saved to '{class_folder}'.")
    else:
        print(f"Image '{image_file}' skipped due to low confidence ({confidence:.2f}).")
