import cv2
import numpy as np
import onnxruntime as ort
from time import time
from general import check_img_size, non_max_suppression, xyxy2xywh

def get_device():
    # Note: ONNX Runtime will automatically use the best available execution provider (CPU, CUDA, etc.)
    print("Using ONNX Runtime for inference")

def plot_bounding_boxes(image, data):
    # This function remains unchanged
    ...

def find_cards(vid_path, model_path='weights/best.onnx', img_size=640, conf_thres=0.25, iou_thres=0.45):
    get_device()
    # Load ONNX model
    ort_session = ort.InferenceSession(model_path)
    input_name = ort_session.get_inputs()[0].name

    # Load image
    cap = cv2.VideoCapture(vid_path)
    while cap.isOpened():
        ret, img0 = cap.read()
        if not ret:
            break
        img0 = cv2.resize(img0, (img_size, img_size))
        img = img0.copy()
        assert img is not None, 'Image Not Found '

        # Convert and normalize image
        img = img[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0  # Convert BGR to RGB and normalize
        img = np.expand_dims(img, axis=0)  # Add batch dimension, setting batch size to 1

        # Replicate the image to match the required batch size of 4
        img = np.repeat(img, 4, axis=0)  # This changes the shape from [1, 3, 640, 640] to [4, 3, 640, 640]

        # Model inference
        s = time()
        pred_onnx = ort_session.run(None, {input_name: img})[0]
        print(type(pred_onnx))
        pred = non_max_suppression(pred_onnx, conf_thres, iou_thres, classes=None, agnostic=False) # returns list of detections
        print(pred)
        for i, det in enumerate(pred):  # detections for image i
            predn = det.clone()
            print(predn)
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
        print(boxes)

        # Processing of predictions must be adjusted based on your ONNX model's output format
        # The following is a placeholder for the processing steps, which will vary depending on the model
        # Example: Extracting and processing model outputs
        # pred = process_predictions(pred_onnx, conf_thres, iou_thres)

        # Example of displaying the results - this part will need to be adapted
        # for i, det in enumerate(pred):  # detections for image i
        #     ...

        cv2.imshow('result', img0)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break
        for _ in range(3):
            cap.read()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    vid_path = "videos/video1.mov"
    model_path = "weights/best.onnx"
    img_size = 640  # Ensure this matches the model's expected input dimensions
    conf_thres = 0.25
    iou_thres = 0.45
    find_cards(vid_path, model_path, img_size, conf_thres, iou_thres)
