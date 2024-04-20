from PIL import Image
import os
import random

# Set the directory containing the images
directory = 'saved_frames'

# Iterate over each file in the directory
for filename in os.listdir(directory):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
        # Check if the image should be rotated with a probability of 1/2
        if random.choice([True, False]):
            # Construct the full file path
            file_path = os.path.join(directory, filename)
            
            # Open the image
            img = Image.open(file_path)
            
            # Rotate the image 180 degrees
            rotated_img = img.rotate(180)
            
            # Save the rotated image, replacing the original
            rotated_img.save(file_path)
            print(f"Rotated and saved: {filename}")
        else:
            print(f"No rotation for: {filename}")

print("Image processing completed.")
