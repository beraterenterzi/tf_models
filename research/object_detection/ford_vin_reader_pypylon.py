import numpy as np
import os
import tensorflow as tf
from utils import label_map_util

from utils import visualization_utils as vis_util
import cv2
from datetime import datetime
from pypylon import pylon


camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
converter = pylon.ImageFormatConverter()
converter.OutputPixelFormat = pylon.PixelType_BGR8packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

pred_resize_w = 1024
pred_resize_h = 768

vis_resize_w = 1920
vis_resize_h = 1080

MODEL_NAME = 'ford_vin_reader'
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
PATH_TO_LABELS = os.path.join('data', 'label.txt')
NUM_CLASSES = 18


max_boxes_to_draw = 3
min_confidence = 0.3

frame_rate_calc = 1
freq = cv2.getTickFrequency()

color_move_on = (255, 200, 90)
color_red = (25, 20, 240)
color = color_move_on
text_x_align = 10
inference_time_y = 30
fps_y = 90
analysis_time_y = 60
font_scale = 0.7
thickness = 2
rect_thickness = 3

video_type = cv2.VideoWriter_fourcc(*'XVID')
video_writer = cv2.VideoWriter()
video_writer.open("video.avi", video_type, 25.0,
                  (vis_resize_w, vis_resize_h))


def load_labels(path, encoding='utf-8'):
    with open(path, 'r', encoding=encoding) as f:
        lines = f.readlines()
        if not lines:
            return {}

        if lines[0].split(' ', maxsplit=1)[0].isdigit():
            pairs = [line.split(' ', maxsplit=1) for line in lines]
            return {int(index + 1): label.strip() for index, label in pairs}
        else:
            return {index + 1: line.strip() for index, line in enumerate(lines)}


img_count = 0
labels = load_labels(PATH_TO_LABELS)

with tf.device('/cpu:0'):
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()
        with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:

            while camera.IsGrabbing():
                grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

                if grabResult.GrabSucceeded():
                    t1 = cv2.getTickCount()
                    start = datetime.now()

                    image = converter.Convert(grabResult)
                    image_np = image.GetArray()
                    image = cv2.resize(image_np, (vis_resize_w, vis_resize_h))
                    out = image.copy()

                    image_np = cv2.resize(image_np, (pred_resize_w, pred_resize_h), interpolation=cv2.INTER_LINEAR)
                    image_np_expanded = np.expand_dims(image_np, axis=0)
                    s = datetime.now()

                    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                    scores = detection_graph.get_tensor_by_name('detection_scores:0')
                    classes = detection_graph.get_tensor_by_name('detection_classes:0')
                    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
                    (boxes, scores, classes, num_detections) = sess.run(
                        [boxes, scores, classes, num_detections],
                        feed_dict={image_tensor: image_np_expanded})

                    e = datetime.now()
                    d = e - s
                    inf_time = round(d.total_seconds(), 3)

                    vin_array = []

                for i in range(min(max_boxes_to_draw, int(num_detections))):
                    if scores[0][i] >= min_confidence:
                        y1, x1, y2, x2 = tuple(np.squeeze(boxes)[i].tolist())
                        xmin = int(x1 * vis_resize_w)
                        ymin = int(y1 * vis_resize_h)
                        xmax = int(x2 * vis_resize_w)
                        ymax = int(y2 * vis_resize_h)

                        x_mid = int(xmin + ((xmax - xmin) / 2))
                        y_mid = int(ymin + ((ymax - ymin) / 2))

                        object_name = labels.get(int(classes[0][i]), int(classes[0][i]))

                        label = '%s: %d%%' % (object_name, int(scores[0][i] * 100))
                        label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                                                                thickness)
                        label_ymin = max(ymin, label_size[1] + 10)
                        cv2.rectangle(out, (xmin, label_ymin - label_size[1] - 10),
                                      (xmin + label_size[0], label_ymin + base_line - 10), (255, 255, 255),
                                      cv2.FILLED)
                        cv2.rectangle(out, (xmin, label_ymin - label_size[1] - 10),
                                      (xmin + label_size[0], label_ymin + base_line - 10), (0, 0, 0))
                        cv2.putText(out, label, (xmin, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                                    color,
                                    thickness,
                                    cv2.LINE_AA)

                        cv2.rectangle(out, (xmin, ymin), (xmax, ymax), color, rect_thickness)

                end = datetime.now()
                duration = end - start
                a_time = round(duration.total_seconds(), 3)

                inference_time = 'Inference Time: {}'.format(inf_time)
                label_size, base_line = cv2.getTextSize(inference_time, cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                                                        thickness)
                label_ymin = max(inference_time_y, label_size[1] + 10)
                cv2.rectangle(out, (text_x_align, label_ymin - label_size[1] - 10),
                              (text_x_align + label_size[0], label_ymin + base_line - 10), (255, 255, 255),
                              cv2.FILLED)
                cv2.rectangle(out, (text_x_align - 2, label_ymin - label_size[1] - 12),
                              (text_x_align + 2 + label_size[0], label_ymin + base_line - 8), (0, 0, 0))
                cv2.putText(out, inference_time, (text_x_align, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX,
                            font_scale,
                            color_move_on,
                            thickness,
                            cv2.LINE_AA)

                fps = 'FPS: {0:.2f}'.format(frame_rate_calc)
                label_size, base_line = cv2.getTextSize(fps, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                label_ymin = max(fps_y, label_size[1] + 10)
                cv2.rectangle(out, (text_x_align, label_ymin - label_size[1] - 10),
                              (text_x_align + label_size[0], label_ymin + base_line - 10), (255, 255, 255),
                              cv2.FILLED)
                cv2.rectangle(out, (text_x_align - 2, label_ymin - label_size[1] - 12),
                              (text_x_align + 2 + label_size[0], label_ymin + base_line - 8), (0, 0, 0))
                cv2.putText(out, fps, (text_x_align, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                            color_move_on,
                            thickness,
                            cv2.LINE_AA)

                analysis_time = 'Analysis Time: {}'.format(a_time)
                label_size, base_line = cv2.getTextSize(analysis_time, cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                                                        thickness)
                label_ymin = max(analysis_time_y, label_size[1] + 10)
                cv2.rectangle(out, (text_x_align, label_ymin - label_size[1] - 10),
                              (text_x_align + label_size[0], label_ymin + base_line - 10), (255, 255, 255),
                              cv2.FILLED)
                cv2.rectangle(out, (text_x_align - 2, label_ymin - label_size[1] - 12),
                              (text_x_align + 2 + label_size[0], label_ymin + base_line - 8), (0, 0, 0))
                cv2.putText(out, analysis_time, (text_x_align, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX,
                            font_scale,
                            color_move_on,
                            thickness,
                            cv2.LINE_AA)

                cv2.imshow('QCAI - MOVE ON', out)
                video_writer.write(out)
                t2 = cv2.getTickCount()
                time1 = (t2 - t1) / freq
                frame_rate_calc = 1 / time1
                img_count += 1

                if cv2.waitKey(5) & 0xFF == 27:
                    cv2.destroyAllWindows()
                    cap.release()
                    cv2.destroyAllWindows()
                    break
