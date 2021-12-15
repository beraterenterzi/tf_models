
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import io
import pandas as pd
import tensorflow as tf

from PIL import Image
from utils import dataset_util
from collections import namedtuple, OrderedDict


def class_text_to_int(row_label):
    if row_label.__eq__('a'):
        return 1
    else:
        None


def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path):
    with tf.io.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(str(row['class']).encode('utf8'))
        classes.append(class_text_to_int(row['class']))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def train_main():
    writer = tf.compat.v1.python_io.TFRecordWriter(outputPath_train)
    path = os.path.join(os.getcwd(), image_dir_train)
    examples = pd.read_csv(csv_input_train)
    grouped = split(examples, 'filename')
    for group in grouped:
        tf_example = create_tf_example(group, path)
        writer.write(tf_example.SerializeToString())
    writer.close()
    print('\n\nSuccessfully created the train TFRecords')


def test_main():
    writer = tf.compat.v1.python_io.TFRecordWriter(outputPath_test)
    path = os.path.join(os.getcwd(), image_dir_test)
    examples = pd.read_csv(csv_input_test)
    grouped = split(examples, 'filename')
    for group in grouped:
        tf_example = create_tf_example(group, path)
        writer.write(tf_example.SerializeToString())
    writer.close()
    print('\n\nSuccessfully created the test TFRecords')


WORKS_NAME = 'intern_training'

csv_input_train = 'data/' + WORKS_NAME + '/train_labels.csv'
#image_dir_train = 'training_images/' + WORKS_NAME + '/train'
image_dir_train = 'training_images/' + WORKS_NAME + '/train/augmented_imgs'
outputPath_train = 'data/' + WORKS_NAME + '/train.record'

csv_input_test = 'data/' + WORKS_NAME + '/test_labels.csv'
#image_dir_test = 'training_images/' + WORKS_NAME + '/test'
image_dir_test = 'training_images/' + WORKS_NAME + '/test/augmented_imgs'
outputPath_test = 'data/' + WORKS_NAME + '/test.record'

train_main()
test_main()
