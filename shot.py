from picamera2 import Picamera2
import cv2
import numpy as np

picam1 = Picamera2(0)
picam1.preview_configuration.main.size = (1280, 720)
picam1.preview_configuration.main.format = "RGB888"
picam1.preview_configuration.align()
picam1.configure("preview")

picam2 = Picamera2(1)
picam2.preview_configuration.main.size = (1280, 720)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")

picam1.start()
picam2.start()

count = 1

while True:
    frame1 = picam1.capture_array()
    frame2 = picam2.capture_array()

    all_frame = np.hstack((frame1, frame2))
    
    scale_factor = 0.5

    # 计算新的尺寸
    height, width = all_frame.shape[:2]
    new_dim = (int(width * scale_factor), int(height * scale_factor))

    cv2.imshow("Camera", cv2.resize(all_frame, new_dim, interpolation=cv2.INTER_LINEAR))
    
    key = cv2.waitKey(1)
    if key == ord("q"):
        break
    elif key == ord("s"):
        cv2.imwrite(f"shot_a{count}.jpg", frame1)
        cv2.imwrite(f"shot_b{count}.jpg", frame2)
        count += 1
