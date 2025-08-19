# import requests
# import base64
# import os

# # Define the URL for the prediction endpoint
# url = 'http://127.0.0.1:5001/result'  # Ensure this matches your FastAPI endpoint and port

# # Define the image path
# image_path = r"good.bmp"

# # Check if the image file exists
# if not os.path.exists(image_path):
#     print(f"Error: Image file not found at {image_path}")
#     exit()

# # Read the image file and encode it to base64
# try:
#     with open(image_path, 'rb') as image_file:
#         image_base64 = base64.b64encode(image_file.read()).decode('utf-8')
# except Exception as e:
#     print(f"Error reading or encoding the image: {e}")
#     exit()

# # Create the JSON payload with the base64 encoded image
# payload = {'image': image_base64}

# # Send the POST request with the JSON payload
# try:
#     response = requests.post(url, json=payload)
#     response.raise_for_status()  # Raise an error for HTTP codes 4xx/5xx
#     # Print the response JSON if successful
#     print(response.json())
# except requests.exceptions.RequestException as e:
#     print(f"Error sending request: {e}")


import requests
import base64
import os
import time

# Define the URL for the prediction endpoint
url = 'http://127.0.0.1:5001/result'  # Ensure this matches your FastAPI endpoint and port

# Define the folder containing images
image_folder =r"testing_images"

# Optional: time delay between requests (seconds)
sleep_time = 1
# Get list of image files with typical image extensions
image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg','bmp'))]

if not image_files:
    print(f"No image files found in folder: {image_folder}")
    exit()
# Process each image file
for idx, image_file in enumerate(image_files, start=1):
    image_path = os.path.join(image_folder, image_file)

    # Read and encode the image
    try:
        with open(image_path, 'rb') as image_file_obj:
            image_base64 = base64.b64encode(image_file_obj.read()).decode('utf-8')
    except Exception as e:
        print(f"[{image_path}] Error reading or encoding image: {e}")
        continue
    # Prepare payload
    payload = {'image': image_base64}
    # Send POST request
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        # Print response JSON
        print(f"[{idx}/{len(image_files)}] Response for {image_path}: {response.json()}")
    except requests.exceptions.RequestException as e:
        print(f"[{image_path}] Error sending request: {e}")

    # Optional: wait before next request
    time.sleep(sleep_time)

