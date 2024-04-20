# Create Dataset

The first step is to clone the labelimg repository (outside of this repository) to be able to label the images. To do that, navigate out of this folder and run:

```bash
git clone https://github.com/HumanSignal/labelImg.git
```

This will download an app that is very useful to label images. After that, you can run the following command to open the app:

```bash
cd labelImg
pip3 install pyqt5 lxml
make qt5py3
python3 labelImg.py
```

In your project folder there is a folder called Labeling, create inside of Labeling a folder called `labelimg-data` and put all the images that you want to label inside of `labelimg-data`.

Go back to the labelImg app and open the dir where the images are located (`labelimg-data`). Make sure that the save dir is the same one. After that, you can start labeling the images. After you finish labeling each image make sure that the save format is YOLO and save the data from the image by pressing save. This will create a .txt file with the same name as the image and the same content as the image but with the coordinates of the bounding box.

After you finish labeling all the images, you can run the `partition.py` directiory to partition the data and create the Dataset. `partition.py` is located in the Labeling folder.

By now, you should have a folder inside of Labeling called Dataset with the following structure:

```
Dataset/
├── images/
│   ├── test/
│   ├── train/
│   └── val/
└── labels/
    ├── test/
    ├── train/
    └── val/

```

## Extras

Under the `Extras` folder, you will find a python script called `saved_frame.py` and `rename.py`. `saved_frame.py` is a script that will show you a video where if you press:

- `s` it will save the frame to `saved_frames`.
- `q` it will quit the video.
- `a` it will rewind a little
- `space` it will pause the video.

This is useful to save frames that you want to label.

`rename.py` is a script that will rename the images and labels in the `saved_frames` folder. In the code, there is a variable called `start_number` that you can modify to start the renaming from a specific number.

# Train the model

To train the model, you first need to copy the Dataset folder into the `yolov7` folder. After that, you can run the following command to train the model:

```bash
python3 train.py --img-size 640 --cfg cfg/training/yolov7.yaml --hyp data/hyp.scratch.custom.yaml --batch 16 --epochs 300 --data data/cards-data.yaml --weights yolov7_training.pt --workers 8 --name yolo_card_det
```

Some important parameters worth trying to change are:

- `--img-size`: The size of the images that the model will use to train.
- `--batch`: The batch size.
- `--epochs`: The number of epochs that the model will train.
- `--name`: The name of the model that will be saved.

Once the training is done, under `runs\train` you will find a folder called by the name you specified, for example `yolo_card_det` and a number in case you've trained multiple models.

Under this folder, you will find another folder named `weights` with the weights of the model. You can use these weights to make predictions. Different checkpoints of the training will be saved in this folder. This is because over training the model can lead to overfitting, so the latest epoch may not be the best.

Under `weights` copy the file named `best.pt` outside of the `Train` directory and into `Run\weights`. You may rename the file to something more descriptive if you're trying things out.

Now everything is ready to run the model.

# Run the model

Under the `Run` directory, you will find a file called `detect.py`. You can run this file to make predictions. `detect.py` cointains the following parameters:

```python
vid_path = "videos/video1.mov"
weights = "weights/best.pt"
img_size = 640
conf_thres = 0.25
iou_thres = 0.45
```

Modify them appropriately to your needs.

- `vid_path` is the path to the video you want to make predictions on.
- `weights` is the path to the weights of the model.
- `img_size` is the size of the images that the model will use to make predictions.
- `conf_thres` is the confidence threshold. If the model is less than this confident, it will not make a prediction.
- `iou_thres` is the intersection over union threshold. This is used to filter out the bounding boxes that are too similar.

# Installation to run on Apple Silicon GPU

Uninstall tensorflow if you have it installed:

```bash
pip3 uninstall tensorflow
```

Install tensorflow-macos:

```bash
python3 -m pip install tensorflow-macos==2.12
```

Install tensorflow-metal plug-in

```bash
python3 -m pip install tensorflow-metal
```

Install accelerated pytorch:

```bash
pip3 install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu
```

For more information visit:

- https://developer.apple.com/metal/tensorflow-plugin/
- https://developer.apple.com/metal/pytorch/
