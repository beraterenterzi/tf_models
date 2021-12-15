import cv2
import numpy as np
import os
from datetime import date


video_source = 0
cap = cv2.VideoCapture(video_source)

# cap.set(3, 640)
# cap.set(4, 480)

today_date = str(date.today())

path = os.getcwd()
title = "Data Acquisition Module - Move On Technology"
print(title)
print("initiating...")

data_path = os.path.join(path, "data")
if not os.path.exists(data_path):
    os.mkdir(data_path)

images_path = os.path.join(data_path, "images")
if not os.path.exists(images_path):
    os.mkdir(images_path)

today_images_path = os.path.join(images_path, today_date)
if not os.path.exists(today_images_path):
    os.mkdir(today_images_path)

videos_path = os.path.join(data_path, "videos")
if not os.path.exists(videos_path):
    os.mkdir(videos_path)

today_videos_path = os.path.join(videos_path, today_date)
if not os.path.exists(today_videos_path):
    os.mkdir(today_videos_path)

image_dir_index = 0
image_write_path = os.path.join(today_images_path, str(image_dir_index))

video_index = 1

frame_count = 0

video_type = cv2.VideoWriter_fourcc(*'XVID')
video_writer = cv2.VideoWriter()

cv2.namedWindow('frame')

save_image = False
save_video = False

while 1:
    ret, frame = cap.read()
    # if np.shape(frame) != ():
    #     frame = cv2.resize(frame, (640, 480))
    # else:
    #     print("camera allocation error")
    #     break
    k = cv2.waitKey(5) & 0xFF
    cv2.imshow("frame", frame)

    if k == ord("a"):
        save_image = False
        print("New image dir created")
        while True:
            image_write_path = os.path.join(today_images_path, str(image_dir_index))
            if os.path.exists(image_write_path):
                image_dir_index += 1
            else:
                os.mkdir(image_write_path)
                break
    if k == ord("s"):
        save_image = True
    if k == ord("d"):
        save_image = False

    if save_image:
        print("Saving images...")
        frame_copy = frame.copy()
        frame_copy = cv2.resize(frame_copy, (640, 480))
        if frame_count % 2 == 0:
            cv2.imwrite(image_write_path + "/frame" + str(frame_count) + ".jpg", frame_copy)

    if k == ord("q"):
        while True:
            video_file_name = today_videos_path + '/video{}.avi'.format(video_index)
            if os.path.isfile(video_file_name):
                video_index += 1
            else:
                break
        video_writer.open(video_file_name, video_type, 25.0, (640, 480))
    if k == ord("w"):
        save_video = True
    if k == ord("e"):
        save_video = False

    if save_video:
        print("Saving video...")
        frame_copy = frame.copy()
        frame_copy = cv2.resize(frame_copy, (640, 480))
        video_writer.write(frame_copy)

    if k == 27:
        cap.release()
        cv2.destroyAllWindows()
        print("Exitting...")
        break
    frame_count += 1
