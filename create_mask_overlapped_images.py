import os
import cv2

def mark_regions(original_folder, binary_folder, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Traverse through the subfolders in the original folder. 
    for root, subfolders, files in os.walk(original_folder):
        for subfolder in subfolders:
            for filename in os.listdir(os.path.join(root, subfolder)):
                if filename.endswith(('.jpg', '.png', '.jpeg', '.bmp')):
                    original_image_path = os.path.join(root, subfolder, filename)
                    
                    # binary_folder_subpath = os.path.relpath(root, original_folder)
                    print(os.path.basename(os.path.dirname(filename)))
                    binary_image_path = os.path.join(binary_folder, subfolder, filename.split(".")[0]+".png")

                    # if not os.path.exists(binary_image_path):
                    #     continue

                    original_img = cv2.imread(original_image_path)
                    #print(original_img.shape)
                    binary_img = cv2.imread(binary_image_path, cv2.IMREAD_GRAYSCALE)

                    # Convert the binary image to a binary mask
                    _, binary_mask = cv2.threshold(binary_img, 1, 255, cv2.THRESH_BINARY)

                    # Apply the binary mask to the original image
                    marked_image = cv2.bitwise_or(original_img, cv2.merge((binary_mask, binary_mask, binary_mask)))

                    # Save the marked image to the output folder preserving the subfolder structure
                    # output_subfolder = os.path.join(output_folder, binary_folder_subpath)
                    output_path = os.path.join(output_folder, subfolder, filename)

                    output_subfolder = subfolder

                    if not os.path.exists(os.path.join(output_folder,output_subfolder)):
                        os.makedirs(os.path.join(output_folder,output_subfolder))

                    cv2.imwrite(output_path, marked_image)


if __name__ == "__main__":
    original_folder_path = "/SUN-SEG/TestHardDataset/Seen/Frame"                 # Replace with the path to the folder containing original images over which you want to overlap the mask
    binary_folder_path = "/data/binary_mask"                                     # Replace with the path to the folder containing binary masks
    output_folder_path = "/data/target_binary_masked_images"                     # Replace with the path to the output folder where you want to save your input control, i.e., masked images

    mark_regions(original_folder_path, binary_folder_path, output_folder_path)
