#from ultralytics import YOLO

#model = YOLO("/home/drone/FINAL/yolov8n.pt")
#model.export(format="onnx", opset=12)
import blobconverter

blob_path = blobconverter.from_onnx(
    model="/home/drone/FINAL/yolov8n.onnx",
    data_type="FP16",        # REQUIRED for Myriad X
    shaves=6                 # OAK-D-Lite optimal
)

print(blob_path)
