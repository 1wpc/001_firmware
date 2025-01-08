import cv2
from picamera2 import Picamera2
import numpy as np
from ultralytics import YOLO

def compute_distance(f, B, disparity):
    """
    通过视差计算物体到摄像头的距离。
    
    参数:
        f (float): 焦距（单位：像素）
        B (float): 基线距离（单位：米）
        disparity (float): 视差值（单位：像素）
        
    返回:
        distance (float): 物体的距离（单位：米）
    """
    if disparity == 0:
        return float('inf')  # 避免除以0
    return (f * B) / disparity

# Initialize the Picamera2
picam2a = Picamera2(0)
picam2a.preview_configuration.main.size = (1280, 720)
picam2a.preview_configuration.main.format = "RGB888"
picam2a.preview_configuration.align()
picam2a.configure("preview")
picam2a.start()


picam2b = Picamera2(1)
picam2b.preview_configuration.main.size = (1280, 720)
picam2b.preview_configuration.main.format = "RGB888"
picam2b.preview_configuration.align()
picam2b.configure("preview")
picam2b.start()
# Load the YOLO11 model
model = YOLO("yolo11n_ncnn_model")

while True:
    # Capture frame-by-frame
    frame1 = picam2a.capture_array()
    frame2 = picam2b.capture_array()

    # Run YOLO11 inference on the frame
    results1 = model.track(frame1)
    results2 = model.track(frame2, persist=True)
    show_frame = results1[0].plot()

    if results1[0].boxes.is_track and results2[0].boxes.is_track:
        id1s = results1[0].boxes.id
        id2s = results2[0].boxes.id

        box1 = results1[0].boxes.xywh
        box2 = results2[0].boxes.xywh
        for i, id1 in enumerate(id1s):
            for j, id2 in enumerate(id2s):
                if id1 == id2:
                    x1 = box1[i][0]
                    y1 = box1[i][1]
                    w1 = box1[i][2]
                    h1 = box1[i][3]
                    x2 = box2[j][0]
                    disparity = abs(x1 - x2)
                    distance = compute_distance(f=2020.8, B=10, disparity=disparity)
                    cv2.putText(show_frame, f"Distance: {distance:.2f} cm", (int(x1+w1/2), int(y1-h1/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    break

    # Display the resulting frame
    cv2.imshow("Camera", show_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) == ord("q"):
        break

# Release resources and close windows
cv2.destroyAllWindows()
