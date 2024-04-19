import os
from sklearn.model_selection import train_test_split
import shutil

# python3.11 train.py --img-size 640 --cfg cfg/training/yolov7.yaml --hyp data/hyp.scratch.custom.yaml --batch 6 --epochs 20 --data data/cards-data.yaml --weights yolov7_training.pt --workers 24 --name yolo_card_det
# python3.11 export.py --img-size 640 --batch 4 --epochs 20 --weights yolov7_training.pt --device mps
# python3.11 test.py --weights runs/train/yolo_card_det5/weights/best.pt --data data/cards-data.yaml --batch-size 16 --task test --name yolo_det

# Read images and annotations

labelimg_path = "labelimg-data"

images = [os.path.join(labelimg_path, x) for x in os.listdir(labelimg_path) if x[-3:] == "png" or x[-3:] == "jpg"]
annotations = [os.path.join(labelimg_path, x) for x in os.listdir(labelimg_path) if x[-3:] == "txt" and x != "classes.txt"]
images.sort()
annotations.sort()

# Split the dataset into train-valid-test splits 
train_images, val_images, train_annotations, val_annotations = train_test_split(images, annotations, test_size = 0.2, random_state = 1)
val_images, test_images, val_annotations, test_annotations = train_test_split(val_images, val_annotations, test_size = 0.5, random_state = 1)

#Utility function to move images 
def copy_files_to_folder(list_of_files, destination_folder):
    for f in list_of_files:
        try:
            shutil.copy(f, destination_folder)
            # shutil.move(f, destination_folder)
        except:
            print(f)
            assert False

def setup_dataset_directory():
    dataset_path = 'Dataset'
    
    # Check if the Dataset directory exists
    if os.path.exists(dataset_path):
        # Delete the Dataset directory and all its content
        shutil.rmtree(dataset_path)
    
    # Create the Dataset directory
    os.makedirs(dataset_path)
    
    # Subdirectories to be created inside images and labels
    subdirs = ['train', 'val', 'test']
    
    # Create images and labels directories with their respective subdirectories
    for subdir in ['images', 'labels']:
        for subsubdir in subdirs:
            os.makedirs(os.path.join(dataset_path, subdir, subsubdir))

setup_dataset_directory()
# Move the splits into their folders
copy_files_to_folder(train_images, 'Dataset/images/train')
copy_files_to_folder(val_images, 'Dataset/images/val/')
copy_files_to_folder(test_images, 'Dataset/images/test/')
copy_files_to_folder(train_annotations, 'Dataset/labels/train/')
copy_files_to_folder(val_annotations, 'Dataset/labels/val/')
copy_files_to_folder(test_annotations, 'Dataset/labels/test/')