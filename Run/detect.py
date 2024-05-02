import cv2
import torch
from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, xyxy2xywh
import numpy as np
from time import time
from process import process_frame, get_average_color, draw_centroids
from centroidtracker import CentroidTracker
import os

os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'


def get_device():
    if torch.backends.mps.is_available():
        mps_device = torch.device("mps")
        print(f"Will run on GPU! Device: {mps_device}")
        return mps_device
    print("Will run on CPU :(")
    return "cpu"

def translate_bbox(minX, minY, maxX, maxY, original_size, new_size):
    # Unpack the original and new dimensions
    original_width, original_height = original_size
    new_height, new_width = new_size
    
    # Calculate the scale factors for width and height
    scale_width = new_width / original_width
    scale_height = new_height / original_height
    
    # Scale the bounding box coordinates
    new_minX = minX * scale_width
    new_minY = minY * scale_height
    new_maxX = maxX * scale_width
    new_maxY = maxY * scale_height
    
    # Return the new coordinates
    return new_minX, new_minY, new_maxX, new_maxY


def find_cards(vid_path, weights='yolov7.pt', img_size=640, conf_thres=0.25, iou_thres=0.45, write_video=False):

    # classes = ["card", "hand", "point", "camera", "rock", "ok"]
    # classes = ["action", "stop", "jump", "left", "point", "card"]
    # classes = ["peace", "palma", "l-sign", "indice", "punch", "card"]
    # classes_map = {"action": "peace", "stop": "palma", "jump": "l-sign", "left": "indice", "point": "punch", "card": "card"}
    classes = ["point", "hand", "scissor", "rock", "card", "delta"]
    device = get_device()
    model = attempt_load(weights, map_location="cpu")  # load FP32 model
    print(device)
    model.to(device)
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(img_size, s=stride)  # check image size

    # Load image
    cap = cv2.VideoCapture(vid_path)
    centroids = {}
    for class_name in classes:
        centroids[class_name] = CentroidTracker()

    (H, W) = (None, None)
    if write_video:
        out_vid_path = vid_path[:-4] + "_processed.mov"
        print("Will write video to: ", out_vid_path)
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Define the codec and create a VideoWriter object to write the video
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter()
        succes = out.open(out_vid_path,fourcc, fps, (frame_width, frame_height),True)
        print("Success: ", succes)
    counter = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # preprocess_time = time()
        original_size = frame.shape[:2]
        resized_img = cv2.resize(frame.copy(), (imgsz, imgsz))
        img = resized_img.copy()
        assert img is not None, 'Image Not Found '

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(device)
        img = img.float()  # uint8 to fp32
        img /= 255.0  # (0 - 255) to (0.0 - 1.0)
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        # print("Time taken for preprocessing: ", time() - preprocess_time)
        s = time()
        with torch.no_grad():
            pred = model(img, augment=False)[0] # returns inference
        # print("Time taken for prediction: ", time() - s)

        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=None, agnostic=False) # returns list of detections
        
        # Process detections
        update_data = {}
        for class_name in classes:
            update_data[class_name] = {"rects": [], "colors": []}
        if W is None or H is None:
            (H, W) = resized_img.shape[:2]

        for i, det in enumerate(pred):  # detections for image i
            predn = det.clone()
            gn = torch.tensor(img.shape[2:])[[1, 0, 1, 0]] 
            names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
            box_data = []
            for *xyxy, conf, cls in predn.tolist():
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                minX, minY, maxX, maxY = translate_bbox(xyxy[0], xyxy[1], xyxy[2], xyxy[3], (W, H), original_size)
                if names[cls] in classes:
                    update_data[names[cls]]["rects"].append((minX, minY, maxX, maxY))
                    # color = get_average_color(frame, minX, minY, maxX, maxY)
                    # update_data[names[cls]]["colors"].append(color)

                box = {"position": {"minX": minX, "minY": minY, "maxX": maxX, "maxY": maxY},
                                        "class_id": int(cls),
                                        "box_caption": "%s %.3f" % (names[cls], conf),
                                        "scores": {"class_score": conf},
                                        "domain": "pixel"}
                box_data.append(box)
        boxes = {"predictions": {"box_data": box_data, "class_labels": names}}

        # update our centroid tracker using the computed set of bounding
        # box rectangles
        objects = {}
        for class_name in classes:
            rects = update_data[class_name]["rects"]
            # colors = update_data[class_name]["colors"]
            ct = centroids[class_name]
            objects[class_name] = ct.update(rects)
            # objects[class_name] = ct.update(rects, colors)
        # objects = ct.update(rects, colors)
        # draw_centroids(resized_img, objects["card"], "card")
        # print(objects)
        frame = process_frame(frame, boxes, objects)
        
        

        # Show results
        if write_video:
            out.write(frame)
        else:
            cv2.imshow('result', frame)
            if cv2.waitKey(20) & 0xFF==ord('q'):
                break
            # for _ in range(5):
            #     cap.read()
            
    cap.release()
    cv2.destroyAllWindows()
    if write_video:
        out.release()

if __name__ == '__main__':
    # vid_path = "../Videos/test4.mov"
    vid_path = 1
    weights = "weights/best-big.pt"
    img_size = 640
    conf_thres = 0.5
    iou_thres = 0.45
    find_cards(vid_path, weights, img_size, conf_thres, iou_thres, write_video=False)
#(250, 204, 166)
#166, 204, 250