import os
import cv2
import json
import argparse
from tqdm import tqdm
import xml.etree.ElementTree as ET

COCO_DICT = ['images', 'annotations', 'categories']

IMAGES_DICT = ['file_name', 'height', 'width', 'id']
## {'license': 1, 'file_name': '000000516316.jpg', 'coco_url': '',
## 'height': 480, 'width': 640, 'date_captured': '2013-11-18 18:15:05',
## 'flickr_url': '', 'id': 516316}

ANNOTATIONS_DICT = ['image_id', 'iscrowd', 'area', 'bbox', 'category_id', 'id']
## {'segmentation': [[]],
## 'area': 58488.148799999995, 'iscrowd': 0, 
## 'image_id': 170893, 'bbox': [270.55, 80.55, 367.51, 393.7],
## 'category_id': 18, 'id': 9940}

CATEGORIES_DICT = ['id', 'name']
## {'supercategory': 'person', 'id': 1, 'name': 'person'}
## {'supercategory': 'vehicle', 'id': 2, 'name': 'bicycle'}

# change categories according to the labels of yolo,
# 0 -> b ... 7 -> s, for the reason that the yolo labels are converted from voc
YOLO_CATEGORIES = ['b', 'h', 'pb', 'ph', 'm', 'j', 's']
VOC_CATEGORIES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', \
                  'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
                  'tvmonitor']

parser = argparse.ArgumentParser(description='2COCO')
parser.add_argument('--image_path', type=str, default='./train.txt', help='the relative path of dataset,train,val,test')
parser.add_argument('--annotation_path', type=str, default='./labels/', help='the relative path of all labels')
parser.add_argument('--dataset', type=str, default='YOLO', help='the annotation type of dataset')
parser.add_argument('--save', type=str, default='./train.json', help='destination json file')

args = parser.parse_args()


def load_json(path):
    with open(path, 'r') as f:
        json_dict = json.load(f)
        for i in json_dict:
            print(i)
        print(json_dict['annotations'])


def save_json(dict, path):
    print('SAVE_JSON...')
    with open(path, 'w') as f:
        json.dump(dict, f)
    print('SUCCESSFUL_SAVE_JSON:', path)


def load_image(path):
    img = cv2.imread(path)
    return img.shape[0], img.shape[1]


def generate_categories_dict(category):
    print('GENERATE_CATEGORIES_DICT...')
    return [{CATEGORIES_DICT[0]: category.index(x) + 1, CATEGORIES_DICT[1]: x} for x in category]


def generate_images_dict(imagelist, image_path, start_image_id):
    print('GENERATE_IMAGES_DICT...')
    images_dict = []
    with tqdm(total=len(imagelist)) as load_bar:
        for x in imagelist:
            dict = {IMAGES_DICT[0]: x, IMAGES_DICT[1]: load_image(image_path + x)[0], \
                    IMAGES_DICT[2]: load_image(image_path + x)[1], IMAGES_DICT[3]: imagelist.index(x) + start_image_id}
            load_bar.update(1)
            images_dict.append(dict)
    return images_dict


# return [{IMAGES_DICT[0]:x,IMAGES_DICT[1]:load_image(image_path+x)[0],\
# 				IMAGES_DICT[2]:load_image(image_path+x)[1],IMAGES_DICT[3]:imagelist.index(x)+start_image_id} for x in imagelist]


def YOLO_Dataset(image_path, annotation_path, start_image_id=0, start_id=0):

    # change YOLO_CATEGORIES to your own caterories
    categories_dict = generate_categories_dict(YOLO_CATEGORIES)

    #imgname = os.listdir(image_path)
    #images_dict = generate_images_dict(imgname, image_path, start_image_id)
    f = open(image_path)
    imgname = []
    line = f.readline()
    while line:
        imgname.append(line[48:-1])
        line = f.readline()
    f.close()
    print(imgname)
    images_dict = generate_images_dict(imgname, './images/', start_image_id)
    print('GENERATE_ANNOTATIONS_DICT...')
    annotations_dict = []
    id = start_id
    for i in images_dict:
        image_id = i['id']
        image_name = i['file_name']
        W, H = i['width'], i['height']
        annotation_txt = annotation_path + image_name.split('.')[0] + '.txt'

        txt = open(annotation_txt, 'r')
        lines = txt.readlines()

        for j in lines:
            category_id = int(j.split(' ')[0]) + 1
            category = YOLO_CATEGORIES
            x = float(j.split(' ')[1])
            y = float(j.split(' ')[2])
            w = float(j.split(' ')[3])
            h = float(j.split(' ')[4])
            x_min = (x - w / 2) * W
            y_min = (y - h / 2) * H
            w = w * W
            h = h * H
            area = w * h
            bbox = [x_min, y_min, w, h]
            dict = {'image_id': image_id, 'iscrowd': 0, 'area': area, 'bbox': bbox, 'category_id': category_id,
                    'id': id}
            annotations_dict.append(dict)
            id = id + 1
    print('SUCCESSFUL_GENERATE_YOLO_JSON')
    return {COCO_DICT[0]: images_dict, COCO_DICT[1]: annotations_dict, COCO_DICT[2]: categories_dict}



if __name__ == '__main__':

    dataset = args.dataset
    save = args.save
    image_path = args.image_path
    annotation_path = args.annotation_path

    if dataset == 'YOLO':
        json_dict = YOLO_Dataset(image_path, annotation_path, 0)
    save_json(json_dict, save)
