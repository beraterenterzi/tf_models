from data_aug_helper.XmlListConfig import *
from data_aug_helper.data_aug import *
from data_aug_helper.bbox_util import *
import os
import pickle as pkl
import numpy as np
import cv2
import pandas as pd


def xml_to_pickle(directory_name):
    path = os.getcwd() + directory_name
    file_count = 0
    for file in os.listdir(path):
        extension = file.split('.')[-1]
        filename = str(file.split('.')[0])
        new_file_loc = path + filename
        if extension == 'xml':
            file_count += 1
            file_loc = path + file
            tree = ElementTree.parse(file_loc)
            root = tree.getroot()
            xmldict = XmlDictConfig(root)
            boxes = np.empty((0, 5))
            for keys in xmldict:
                if keys == "object":
                    if str(type(xmldict[keys])) == "<class 'data_aug_helper.XmlListConfig.XmlDictConfig'>":
                        xmin = float(xmldict[keys]["bndbox"]["xmin"])
                        ymax = float(xmldict[keys]["bndbox"]["ymax"])
                        xmax = float(xmldict[keys]["bndbox"]["xmax"])
                        ymin = float(xmldict[keys]["bndbox"]["ymin"])
                        classname = xmldict[keys]["name"]
                        class_id = class_text_to_int(classname)
                        box = []
                        box.append([xmin, ymax, xmax, ymin, class_id])
                        boxes = np.append(boxes, box, axis=0)
                    else:
                        for object in xmldict[keys]:
                            xmin = float(object["bndbox"]["xmin"])
                            ymax = float(object["bndbox"]["ymax"])
                            xmax = float(object["bndbox"]["xmax"])
                            ymin = float(object["bndbox"]["ymin"])
                            classname = object["name"]
                            class_id = class_text_to_int(classname)
                            box = []
                            box.append([xmin, ymax, xmax, ymin, class_id])
                            boxes = np.append(boxes, box, axis=0)
            pkl.dump(boxes, open(new_file_loc + ".pkl", "wb"))


def create_csv_row(bnd_box, img_name, img_width, img_height):
    for row in bnd_box:
        xmin = int(row[0])
        ymin = int(row[3])
        xmax = int(row[2])
        ymax = int(row[1])
        
        if xmin > xmax:
            tempx = xmax
            xmax = xmin
            xmin = tempx
            
        if ymin > ymax:
            tempy = ymax
            ymax = ymin
            ymin = tempy
           
        if xmin < 0 or ymin < 0 or xmax > img_width or ymax > img_height:
            break
        value = (img_name,
                 img_width,
                 img_height,
                 class_int_to_text(row[4]),
                 xmin,
                 ymin,
                 xmax,
                 ymax)
        value_list.append(value)


def HorizontalFlip(img, bnd_boxes, file_path, file_name, index):
    img_, bnd_boxes_ = RandomHorizontalFlip(1)(img.copy(), bnd_boxes.copy())
    img_name = file_name + str(index) + ".jpg"
    width = img.shape[1]
    height = img.shape[0]
    create_csv_row(bnd_boxes_, img_name, width, height)
    img_loc_name = file_path + img_name
    cv2.imwrite(img_loc_name, img_)
    index += 1
    return index


def Scaling(img, bnd_boxes, file_path, file_name, index):
    img_, bnd_boxes_ = RandomScale(0.3, diff=True)(img.copy(), bnd_boxes.copy())
    img_name = file_name + str(index) + ".jpg"
    width = img.shape[1]
    height = img.shape[0]
    create_csv_row(bnd_boxes_, img_name, width, height)
    img_loc_name = file_path + img_name
    cv2.imwrite(img_loc_name, img_)
    index += 1
    return index


def Translation(img, bnd_boxes, file_path, file_name, index):
    img_, bnd_boxes_ = RandomTranslate(0.3, img.copy(), bnd_boxes.copy())
    img_name = file_name + str(index) + ".jpg"
    width = img.shape[1]
    height = img.shape[0]
    create_csv_row(bnd_boxes_, img_name, width, height)
    img_loc_name = file_path + img_name
    cv2.imwrite(img_loc_name, img_)
    index += 1
    return index


def Rotation(img, bnd_boxes, file_path, file_name, index):
    j = 0
    for i in range(10, 120, 10):
        j += 1
        img_, bnd_boxes_ = RandomRotate(i)(img.copy(), bnd_boxes.copy())
        img_name = file_name + str(index + j) + ".jpg"
        width = img.shape[1]
        height = img.shape[0]
        create_csv_row(bnd_boxes_, img_name, width, height)
        img_loc_name = file_path + img_name
        cv2.imwrite(img_loc_name, img_)
    return index + j + 1


def Shearing(img, bnd_boxes, file_path, file_name, index):
    img_, bnd_boxes_ = RandomShear(0.2)(img.copy(), bnd_boxes.copy())
    img_name = file_name + str(index) + ".jpg"
    width = img.shape[1]
    height = img.shape[0]
    create_csv_row(bnd_boxes_, img_name, width, height)
    img_loc_name = file_path + img_name
    cv2.imwrite(img_loc_name, img_)
    index += 1
    return index


def bnd_box_data_augmentation(directory_name, horizontal_flip=True, scaling=True, rotation=True, shearing=True):
    print("horizontal_flip: " + str(horizontal_flip))
    print("scaling: " + str(scaling))
    print("rotation: " + str(rotation))
    print("shearing: " + str(shearing))
    file_count = 0
    aug_file_count = 0
    path = os.getcwd() + directory_name
    new_path = os.getcwd() + directory_name + 'augmented_imgs/'
    if not os.path.exists(new_path):
        os.mkdir(new_path)
    for file in os.listdir(path):
        index = 0
        filename = file.split('.')[0]
        extension = file.split('.')[-1]
        if extension == "jpg":
            file_count += 1
            img = cv2.imread(path + file)
            bnd_boxes = pkl.load(open(path + filename + ".pkl", "rb"))
            if horizontal_flip:
                index = HorizontalFlip(img, bnd_boxes, new_path, filename, index)
            if scaling:
                index = Scaling(img, bnd_boxes, new_path, filename, index)
            if rotation:
                index = Rotation(img, bnd_boxes, new_path, filename, index)
            if shearing:
                index = Shearing(img, bnd_boxes, new_path, filename, index)
            aug_file_count += index
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    values_df = pd.DataFrame(value_list, columns=column_name)
    return values_df, file_count, aug_file_count - file_count


def class_text_to_int(row_label):
    if row_label.__eq__('a'):
        return 1
    else:
        None


def class_int_to_text(class_id):
    if class_id.__eq__(1):
        return 'a'
    else:
        None


value_list = []
if __name__ == "__main__":

    print('PROCESS STARTING...')

    WORKS_NAME = 'intern_training'

    data_works_path = os.getcwd() + '/data/' + WORKS_NAME
    if not os.path.exists(data_works_path):
        print('Directory Created.. ', data_works_path)
        os.mkdir(data_works_path)

    print('AUGMENTATION STARTING...')
    print('Processing train images... ')
    train_csv_path = 'data/' + WORKS_NAME + '/train_labels.csv'
    train_img_dir_name = '/training_images/' + WORKS_NAME + '/train/'
    xml_to_pickle(train_img_dir_name)
    train_df, train_img_count, train_aug_img_count = bnd_box_data_augmentation(train_img_dir_name, horizontal_flip=True,
                                                                               scaling=True, rotation=True,
                                                                               shearing=True)
    train_df.to_csv(train_csv_path, index=None)
    train_dataset_length = len(value_list)
    print("Created Train CSV")

    value_list = []

    print('Processing test images... ')
    test_csv_path = 'data/' + WORKS_NAME + '/test_labels.csv'
    test_img_dir_name = '/training_images/' + WORKS_NAME + '/test/'
    xml_to_pickle(test_img_dir_name)
    test_df, test_img_count, test_aug_img_count = bnd_box_data_augmentation(test_img_dir_name, horizontal_flip=True,
                                                                            scaling=True, rotation=True, shearing=True)
    test_df.to_csv(test_csv_path, index=None)
    test_dataset_length = len(value_list)
    print("Created Test CSV")

    print("\nFINISHED...")
    print("Train img count:", train_img_count)
    print("Train augmented img count:", train_aug_img_count)
    print('Train Dataset Length: ', train_dataset_length)
    print("Test img count:", test_img_count)
    print("Test augmented img count:", test_aug_img_count)
    print('Test Dataset Length: ', test_dataset_length)
