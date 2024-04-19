import os
import glob

# Specify the directory containing the images
image_directory = 'saved_frames'

# List all image files in the directory of common image formats
image_files = glob.glob(os.path.join(image_directory, '*.[pP][nN][gG]')) + \
              glob.glob(os.path.join(image_directory, '*.[jJ][pP][gG]')) + \
              glob.glob(os.path.join(image_directory, '*.[jJ][pP][eE][gG]'))

# Sort the list to maintain order, optional
image_files.sort()

# Start the naming from 2
start_number = 159

# Rename each image file
for image_file in image_files:
    # Extract the file extension
    file_extension = os.path.splitext(image_file)[1]
    # Generate the new file name
    new_file_name = f"{start_number}{file_extension}"
    # Generate the full path for the new file name
    new_file_path = os.path.join(image_directory, new_file_name)
    # Rename the file
    os.rename(image_file, new_file_path)
    # Increment the starting number for the next file
    start_number += 1

print("Renaming completed.")
