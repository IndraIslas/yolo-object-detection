import cv2
import torch
from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, xyxy2xywh
import numpy as np
from time import time
from process import process_frame


def get_device():
    if torch.backends.mps.is_available():
        mps_device = torch.device("mps")
        print(f"Will run on GPU! Device: {mps_device}")
        return mps_device
    print("Will run on CPU :(")
    return "cpu"



def find_cards(vid_path, weights='yolov7.pt', img_size=640, conf_thres=0.25, iou_thres=0.45, write_video=False):
    device = get_device()
    model = attempt_load(weights, map_location="cpu")  # load FP32 model
    print(device)
    model.to(device)
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(img_size, s=stride)  # check image size

    # Load image
    cap = cv2.VideoCapture(vid_path)
    if write_video:
        out_vid_path = vid_path[:-4] + "_processed.mov"
        print("Will write video to: ", out_vid_path)
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Define the codec and create a VideoWriter object to write the video
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter()
        succes = out.open(out_vid_path,fourcc, fps, (img_size, img_size),True)
        print("Success: ", succes)
    counter = 0
    while cap.isOpened():
        ret, img0 = cap.read()
        if not ret:
            break
        # preprocess_time = time()
        img0 = cv2.resize(img0, (imgsz, imgsz))
        img = img0.copy()
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
        pred = model(img, augment=False)[0] # returns inference
        # print("Time taken for prediction: ", time() - s)

        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=None, agnostic=False) # returns list of detections
        
        # Process detections
        for i, det in enumerate(pred):  # detections for image i
            predn = det.clone()
            gn = torch.tensor(img.shape[2:])[[1, 0, 1, 0]] 
            names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
            box_data = []
            for *xyxy, conf, cls in predn.tolist():
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                box = {"position": {"minX": xyxy[0], "minY": xyxy[1], "maxX": xyxy[2], "maxY": xyxy[3]},
                                        "class_id": int(cls),
                                        "box_caption": "%s %.3f" % (names[cls], conf),
                                        "scores": {"class_score": conf},
                                        "domain": "pixel"}
                box_data.append(box)
        boxes = {"predictions": {"box_data": box_data, "class_labels": names}}
        img0 = process_frame(img0, boxes)
        
        # Show results
        if write_video:
            out.write(img0)
        else:
            cv2.imshow('result', img0)
            if cv2.waitKey(20) & 0xFF==ord('q'):
                break
            for _ in range(5):
                cap.read()
            
    cap.release()
    cv2.destroyAllWindows()
    if write_video:
        out.release()

if __name__ == '__main__':
    vid_path = "../Videos/hand1.mov"
    # vid_path = 0
    weights = "weights/epoch_199.pt"
    img_size = 640
    conf_thres = 0.5
    iou_thres = 0.45
    find_cards(vid_path, weights, img_size, conf_thres, iou_thres, write_video=False)
