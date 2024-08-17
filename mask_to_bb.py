import cv2
import numpy as np
import os

def find_bounding_box(mask):
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Initialize variables to store bounding box coordinates
    x_min, y_min, x_max, y_max = float('inf'), float('inf'), 0, 0
    
    # Find the minimum and maximum coordinates of the bounding box
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        x_min = min(x_min, x)
        y_min = min(y_min, y)
        x_max = max(x_max, x + w)
        y_max = max(y_max, y + h)
    
    return x_min, y_min, x_max, y_max

def convert_to_yolo_format(image_path, mask_path):
    # Load the mask and image
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.imread(image_path)
    
    # Find bounding box coordinates
    x_min, y_min, x_max, y_max = find_bounding_box(mask)
    
    # Get image dimensions
    image_height, image_width = image.shape[:2]
    
    # Calculate YOLO format bounding box values
    x_center = (x_min + x_max) / (2.0 * image_width)
    y_center = (y_min + y_max) / (2.0 * image_height)
    box_width = (x_max - x_min) / image_width
    box_height = (y_max - y_min) / image_height
    
    # YOLO format: class x_center y_center width height
    yolo_format = f"0 {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}"
    
    return yolo_format
    
output_folder = "/data/label/"    # Replace with the path of the folder where you want to save the bounding box labels

# Paths to the binary mask and the corresponding image
mask_path = "/data/mask/"    # Replace with the path of the folder containing binary masks
image_path = "/data/image/"  # Replace with the path of the folder containing original images

for filename in os.listdir(mask_path):
    print(os.path.join(mask_path,filename))
    if os.path.exists(os.path.join(mask_path,filename)):
        yolo_info = convert_to_yolo_format(os.path.join(image_path,filename), os.path.join(mask_path,filename))
        output_file_path = os.path.join(output_folder, os.path.splitext(filename)[0] + ".txt")
    
        with open(output_file_path, "w") as f:
                    f.write(yolo_info)

           
