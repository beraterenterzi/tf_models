import numpy as np
import cv2
import os

path = r"C:\Users\moveon\Desktop\training"

cap = cv2.VideoCapture(os.path.join(path, "moveon_celikformgestamp_surface_inspection_case_study_video1.mp4"))
frame_count = 0

img_path = os.path.join(path, "images")
if not os.path.exists(img_path):
    os.mkdir(img_path)

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_LINEAR)
    if frame_count % 4 == 0:
        cv2.imwrite(os.path.join(img_path, "frame%d.jpg" % frame_count), frame)
    frame_count += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
