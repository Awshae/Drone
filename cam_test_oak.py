import depthai as dai
import cv2
import numpy as np
import os
from ultralytics import YOLO

# ================================
# USER SETTINGS
# ================================
YOLO_MODEL_PATH = "/home/drone/v8/best.pt"
SAVE_FRAMES = True
SAVE_DIR = "detections"
CONF_THRESHOLD = 0.9
YOLO_EVERY_N_FRAMES = 4  # Run YOLO every 4 frames

if SAVE_FRAMES:
    os.makedirs(SAVE_DIR, exist_ok=True)

# ================================
# Load YOLOv8 model
# ================================
model = YOLO(YOLO_MODEL_PATH)
model.conf = CONF_THRESHOLD

# ================================
# DepthAI pipeline
# ================================
pipeline = dai.Pipeline()

# Color camera
cam = pipeline.createColorCamera()
cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)  # for OAK-D-Lite
cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
cam.setInterleaved(False)
cam.setFps(30)

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

# Outputs
xoutRgb = pipeline.createXLinkOut()
xoutRgb.setStreamName("color")
cam.video.link(xoutRgb.input)

xoutDepth = pipeline.createXLinkOut()
xoutDepth.setStreamName("depth")
stereo.depth.link(xoutDepth.input)

# ================================
# Run pipeline
# ================================
with dai.Device(pipeline) as device:
    qRgb = device.getOutputQueue("color", maxSize=4, blocking=False)
    qDepth = device.getOutputQueue("depth", maxSize=4, blocking=False)

    print("✅ OAK-D-Lite Fire Detection running — press ESC to quit.")

    frame_count = 0
    last_results = []
    nearest_distance = None

    while True:
        inRgb = qRgb.tryGet()
        inDepth = qDepth.tryGet()
        if inRgb is None or inDepth is None:
            continue

        colorFrame = inRgb.getCvFrame()
        depthFrame = inDepth.getFrame()

        frame_count += 1

        # --- Run YOLO only every Nth frame ---
        if frame_count % YOLO_EVERY_N_FRAMES == 0:
            results = model.predict(colorFrame, imgsz=640, verbose=False)
            last_results = results
        else:
            results = last_results

        nearest_distance = None

        # --- Process YOLO detections ---
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy() if len(result.boxes) > 0 else []
            for box in boxes:
                x1, y1, x2, y2 = map(int, box[:4])
                conf = box[4] if len(box) > 4 else 0

                cv2.rectangle(colorFrame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(colorFrame, f"{conf:.2f}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                # Depth from stereo
                depthCrop = depthFrame[y1:y2, x1:x2]
                valid = depthCrop[depthCrop > 0]

                if valid.size > 0:
                    distance_m = np.median(valid) / 1000.0
                    cv2.putText(colorFrame, f"{distance_m:.2f} m",
                                (x1, y2 + 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    if nearest_distance is None or distance_m < nearest_distance:
                        nearest_distance = distance_m

        # --- Overlay nearest distance on screen ---
        if nearest_distance is not None:
            text = f"Nearest Fire: {nearest_distance:.2f} m"
            cv2.putText(colorFrame, text, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            print(text)

        # --- Display and save ---
        cv2.imshow("OAK-D Fire Detection + Distance", colorFrame)
        if SAVE_FRAMES:
            cv2.imwrite(os.path.join(SAVE_DIR, f"frame_{frame_count:04d}.jpg"), colorFrame)

        if cv2.waitKey(1) == 27:  # ESC
            break

cv2.destroyAllWindows()
