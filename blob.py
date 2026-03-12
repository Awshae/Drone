import depthai as dai
import cv2
import numpy as np
import os

# ================================
# USER SETTINGS
# ================================
BLOB_PATH = "yolov8n_openvino_2022.1_6shave.blob"
SAVE_FRAMES = True
SAVE_DIR = "detections"
CONF_THRESHOLD = 0.4
IOU_THRESHOLD = 0.45
IMG_SIZE = 640

if SAVE_FRAMES:
    os.makedirs(SAVE_DIR, exist_ok=True)

# ================================
# YOLOv8 DECODER
# ================================
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def nms(boxes, scores, iou_thres):
    idxs = scores.argsort()[::-1]
    keep = []

    while idxs.size > 0:
        i = idxs[0]
        keep.append(i)
        if idxs.size == 1:
            break

        xx1 = np.maximum(boxes[i, 0], boxes[idxs[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[idxs[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[idxs[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[idxs[1:], 3])

        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        inter = w * h
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_j = (boxes[idxs[1:], 2] - boxes[idxs[1:], 0]) * \
                 (boxes[idxs[1:], 3] - boxes[idxs[1:], 1])
        iou = inter / (area_i + area_j - inter)

        idxs = idxs[1:][iou < iou_thres]

    return keep

def decode_yolov8(output, img_w, img_h):
    output = output.reshape(-1, 6)  # [x,y,w,h,conf,class]
    boxes = []
    scores = []

    for det in output:
        conf = det[4]
        if conf < CONF_THRESHOLD:
            continue

        x, y, w, h = det[:4]
        x1 = int((x - w / 2) * img_w)
        y1 = int((y - h / 2) * img_h)
        x2 = int((x + w / 2) * img_w)
        y2 = int((y + h / 2) * img_h)

        boxes.append([x1, y1, x2, y2])
        scores.append(conf)

    if not boxes:
        return []

    boxes = np.array(boxes)
    scores = np.array(scores)
    keep = nms(boxes, scores, IOU_THRESHOLD)

    return [(boxes[i], scores[i]) for i in keep]

# ================================
# DEPTHAI PIPELINE
# ================================
pipeline = dai.Pipeline()

# Color camera
cam = pipeline.createColorCamera()
cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)
cam.setPreviewSize(IMG_SIZE, IMG_SIZE)
cam.setInterleaved(False)
cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
cam.setFps(10)

# Stereo depth
monoLeft = pipeline.createMonoCamera()
monoRight = pipeline.createMonoCamera()
monoLeft.setBoardSocket(dai.CameraBoardSocket.CAM_B)
monoRight.setBoardSocket(dai.CameraBoardSocket.CAM_C)
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

stereo = pipeline.createStereoDepth()
stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
stereo.setLeftRightCheck(True)
stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)

# Neural network
nn = pipeline.createNeuralNetwork()
nn.setBlobPath(BLOB_PATH)
nn.setNumInferenceThreads(2)
nn.input.setBlocking(False)

cam.preview.link(nn.input)

# Outputs
xoutRgb = pipeline.createXLinkOut()
xoutRgb.setStreamName("color")
cam.preview.link(xoutRgb.input)

xoutDepth = pipeline.createXLinkOut()
xoutDepth.setStreamName("depth")
stereo.depth.link(xoutDepth.input)

xoutNN = pipeline.createXLinkOut()
xoutNN.setStreamName("nn")
nn.out.link(xoutNN.input)

# ================================
# RUN
# ================================
with dai.Device(pipeline) as device:
    qRgb = device.getOutputQueue("color", 4, False)
    qDepth = device.getOutputQueue("depth", 4, False)
    qNN = device.getOutputQueue("nn", 4, False)

    print("🔥 YOLOv8 running ON OAK-D (Myriad X)")

    frame_count = 0

    while True:
        inRgb = qRgb.tryGet()
        inDepth = qDepth.tryGet()
        inNN = qNN.tryGet()

        if inRgb is None or inDepth is None or inNN is None:
            continue

        frame = inRgb.getCvFrame()
        depth = inDepth.getFrame()

        nn_out = np.array(inNN.getFirstLayerFp16())
        detections = decode_yolov8(nn_out, frame.shape[1], frame.shape[0])

        nearest = None

        for (x1, y1, x2, y2), conf in detections:
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(frame, f"{conf:.2f}", (x1, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

            crop = depth[y1:y2, x1:x2]
            valid = crop[crop > 0]
            if valid.size:
                d = np.median(valid) / 1000
                cv2.putText(frame, f"{d:.2f} m", (x1, y2+15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
                nearest = d if nearest is None else min(nearest, d)

        if nearest:
            cv2.putText(frame, f"Nearest Fire: {nearest:.2f} m",
                        (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            print(f"Nearest Fire: {nearest:.2f} m")

        cv2.imshow("OAK-D YOLOv8 Fire Detection", frame)

        if SAVE_FRAMES:
            cv2.imwrite(f"{SAVE_DIR}/frame_{frame_count:04d}.jpg", frame)

        frame_count += 1
        if cv2.waitKey(1) == 27:
            break

cv2.destroyAllWindows()
