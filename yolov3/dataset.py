#================================================================
#
#   File name   : dataset.py
#   Author      : PyLessons
#   Created date: 2020-07-31
#   Website     : https://pylessons.com/
#   GitHub      : https://github.com/pythonlessons/TensorFlow-2.x-YOLOv3
#   Description : functions used to prepare dataset for custom training
#
#================================================================
# TODO: transfer numpy to tensorflow operations
import os
import cv2
import random
import numpy as np
import tensorflow as tf
from yolov3.utils import read_class_names, image_preprocess
from yolov3.yolov3 import bbox_iou
from yolov3.configs import *


class Dataset(object):
    # Dataset preprocess implementation
    def __init__(self, dataset_type, TEST_INPUT_SIZE=TEST_INPUT_SIZE):
        self.annot_path  = TRAIN_ANNOT_PATH if dataset_type == 'train' else TEST_ANNOT_PATH
        self.input_sizes = TRAIN_INPUT_SIZE if dataset_type == 'train' else TEST_INPUT_SIZE
        self.batch_size  = TRAIN_BATCH_SIZE if dataset_type == 'train' else TEST_BATCH_SIZE
        self.data_aug    = TRAIN_DATA_AUG   if dataset_type == 'train' else TEST_DATA_AUG

        self.train_yolo_tiny = TRAIN_YOLO_TINY
        self.train_input_sizes = TRAIN_INPUT_SIZE
        self.strides = np.array(YOLO_STRIDES)
        self.classes = read_class_names(TRAIN_CLASSES)
        self.num_classes = len(self.classes)
        self.anchors = (np.array(YOLO_ANCHORS).T/self.strides).T
        self.anchor_per_scale = YOLO_ANCHOR_PER_SCALE
        self.max_bbox_per_scale = YOLO_MAX_BBOX_PER_SCALE

        self.annotations = self.load_annotations(dataset_type)
        self.num_samples = len(self.annotations)
        self.num_batchs = int(np.ceil(self.num_samples / self.batch_size))
        self.batch_count = 0


    def load_annotations(self, dataset_type):
        final_annotations = []
        with open(self.annot_path, 'r') as f:
            txt = f.read().splitlines()
            annotations = [line.strip() for line in txt if len(line.strip().split()[1:]) != 0]
        np.random.shuffle(annotations)

        # for annotation in annotations:
        #     image_extension = '.jpg'
        #     extension_index = annotation.find(image_extension)
        #     image_path = annotation[:extension_index+len(image_extension)]
        #     line = annotation[extension_index+len(image_extension):].split()
        #     if not os.path.exists(image_path):
        #         raise KeyError("%s does not exist ... " %image_path)
        #     if TRAIN_LOAD_IMAGES_TO_RAM:
        #         image = cv2.imread(image_path)
        #     else:
        #         image = ''
        #     final_annotations.append([image_path, line, image])
        # return final_annotations
        
        # annotations는 총 dataset에 들어있는 data 개수와 같음 3582
        for annotation in annotations: # img_path\20160725-7-158.jpg 373,55,417,471,393,75,397,451,0 396,209,492,386,472,229,416,366,0
            # fully parse annotations
            line = annotation.split() # 이미지에 바운딩 박스가 여러개 있을 수 있으니 split으로 나눠줌 ' ' 스페이스바 기준
            image_path, index = "", 1
            for i, one_line in enumerate(line): # ex) one_line = 215,37,260,247,235,227,240,57,0
                if not one_line.replace(",","").replace("-","").replace(".","").replace("e","").isnumeric(): # one_line이 img_path일 경우
                    if image_path != "": image_path += " " # 만약 이미 주소가 추가 됐다면 " "추가한 후
                    image_path += one_line # image_path 변수에 annotation 추가
                else:
                    index = i
                    break
            if not os.path.exists(image_path):
                raise KeyError("%s does not exist ... " %image_path)
            if TRAIN_LOAD_IMAGES_TO_RAM:
                image = cv2.imread(image_path)
            else:
                image = ''
            final_annotations.append([image_path, line[index:], image]) # annotaions[image_path, bboxes, image]
            
        # print(len(final_annotations)) = 3582 총 data 개수 -> annotation 검증하는 곳
        return final_annotations

    def __iter__(self):
        return self

    def Delete_bad_annotation(self, bad_annotation):
        print(f'Deleting {bad_annotation} annotation line')
        bad_image_path = bad_annotation[0]
        bad_image_name = bad_annotation[0].split('/')[-1] # can be used to delete bad image
        bad_xml_path = bad_annotation[0][:-3]+'xml' # can be used to delete bad xml file

        # remove bad annotation line from annotation file
        with open(self.annot_path, "r+") as f:
            d = f.readlines()
            f.seek(0)
            for i in d:
                if bad_image_name not in i:
                    f.write(i)
            f.truncate()

    def __next__(self):
        with tf.device('/cpu:0'):
            self.train_input_size = random.choice([self.train_input_sizes]) # 416
            self.train_output_sizes = self.train_input_size // self.strides # 416 / [8, 16, 32] -> train_output_sizes = [52, 26, 13]

            batch_image = np.zeros((self.batch_size, self.train_input_size, self.train_input_size, 3), dtype=np.float32) # batch에 들어가는 이미지 변수 생성

            if self.train_yolo_tiny:
                batch_label_mbbox = np.zeros((self.batch_size, self.train_output_sizes[0], self.train_output_sizes[0], self.anchor_per_scale, 12 + self.num_classes), dtype=np.float32)
                batch_label_lbbox = np.zeros((self.batch_size, self.train_output_sizes[1], self.train_output_sizes[1], self.anchor_per_scale, 12 + self.num_classes), dtype=np.float32)
            else:
                batch_label_sbbox = np.zeros((self.batch_size, self.train_output_sizes[0], self.train_output_sizes[0], self.anchor_per_scale, 12 + self.num_classes), dtype=np.float32) # 5->12
                # 52, 52, 3, 12 + self.num_classes
                batch_label_mbbox = np.zeros((self.batch_size, self.train_output_sizes[1], self.train_output_sizes[1], self.anchor_per_scale, 12 + self.num_classes), dtype=np.float32) # 5->12
                # 26, 26, 3, 12 + self.num_classes
                batch_label_lbbox = np.zeros((self.batch_size, self.train_output_sizes[2], self.train_output_sizes[2], self.anchor_per_scale, 12 + self.num_classes), dtype=np.float32) # 5->12
                # 13, 13, 3, 12 + self.num_classes

                batch_sbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4), dtype=np.float32)
                # 4, 100, 4

            batch_mbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4), dtype=np.float32) # 4, 100, 4
            batch_lbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4), dtype=np.float32) # 4, 100, 4

            exceptions = False
            num = 0
            if self.batch_count < self.num_batchs: # 896번 동안 반복, batch_coun는 0에서 시작
                while num < self.batch_size: # 4개 보다 적을때
                    index = self.batch_count * self.batch_size + num # 처음 0 -> 3 까지 1씩 증가 총 4번
                    if index >= self.num_samples: index -= self.num_samples # index가 num_samples = 3582개를 넘을 경우 빼서 다시 0으로 만듦
                    annotation = self.annotations[index] # 첫번째 이미지에 대한 annotation 가져오기
                    image, bboxes = self.parse_annotation(annotation) # 이미지와 바운딩 박스 2개로 나누고 이미지 사이즈에 따라 조정
                    # print(np.shape(bboxes))
                    try:
                        if self.train_yolo_tiny:
                            label_mbbox, label_lbbox, mbboxes, lbboxes = self.preprocess_true_boxes(bboxes)
                        else: # label 이라는 변수로 정리해서 넣어줌
                            label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes = self.preprocess_true_boxes(bboxes)
                    except IndexError:
                        exceptions = True
                        self.Delete_bad_annotation(annotation)
                        print("IndexError, something wrong with", annotation[0], "removed this line from annotation file")

                    batch_image[num, :, :, :] = image
                    batch_label_mbbox[num, :, :, :, :] = label_mbbox
                    batch_label_lbbox[num, :, :, :, :] = label_lbbox
                    batch_mbboxes[num, :, :] = mbboxes # 바운딩 박스의 x, y, w, h 넣어줌
                    batch_lbboxes[num, :, :] = lbboxes
                    if not self.train_yolo_tiny:
                        batch_label_sbbox[num, :, :, :, :] = label_sbbox
                        batch_sbboxes[num, :, :] = sbboxes
                    
                    # print(batch_label_lbbox[num,:,:,:,9])
                    
                    num += 1

                if exceptions:
                    print('\n')
                    raise Exception("There were problems with dataset, I fixed them, now restart the training process.")
                self.batch_count += 1
                if not self.train_yolo_tiny:
                    batch_smaller_target = batch_label_sbbox, batch_sbboxes
                batch_medium_target  = batch_label_mbbox, batch_mbboxes
                batch_larger_target  = batch_label_lbbox, batch_lbboxes

                if self.train_yolo_tiny:
                    return batch_image, (batch_medium_target, batch_larger_target)
                return batch_image, (batch_smaller_target, batch_medium_target, batch_larger_target)
            else:
                self.batch_count = 0
                np.random.shuffle(self.annotations)
                raise StopIteration

    def random_horizontal_flip(self, image, bboxes):
        if random.random() < 0.5:
            _, w, _ = image.shape
            image = image[:, ::-1, :]
            bboxes[:, [0,2]] = w - bboxes[:, [2,0]] # 0, 1, 2, 3, 4, 5, 6, 7
            bboxes[:, [4,6]] = w - bboxes[:, [4,6]]
        return image, bboxes

    def random_crop(self, image, bboxes):
        if random.random() < 0.5:
            h, w, _ = image.shape # 600 X 600
            max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1) # xmin, ymin, xmax, ymax <0,1,2,3>

            max_l_trans = max_bbox[0]
            max_u_trans = max_bbox[1]
            max_r_trans = w - max_bbox[2]
            max_d_trans = h - max_bbox[3]

            crop_xmin = max(0, int(max_bbox[0] - random.uniform(0, max_l_trans)))
            crop_ymin = max(0, int(max_bbox[1] - random.uniform(0, max_u_trans)))
            crop_xmax = max(w, int(max_bbox[2] + random.uniform(0, max_r_trans)))
            crop_ymax = max(h, int(max_bbox[3] + random.uniform(0, max_d_trans)))

            image = image[crop_ymin : crop_ymax, crop_xmin : crop_xmax]

            bboxes[:, [0, 2, 4, 6]] = bboxes[:, [0, 2, 4, 6]] - crop_xmin
            bboxes[:, [1, 3, 5, 7]] = bboxes[:, [1, 3, 5, 7]] - crop_ymin

        return image, bboxes

    def random_translate(self, image, bboxes):
        if random.random() < 0.5:
            h, w, _ = image.shape
            max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)

            max_l_trans = max_bbox[0]
            max_u_trans = max_bbox[1]
            max_r_trans = w - max_bbox[2]
            max_d_trans = h - max_bbox[3]

            tx = random.uniform(-(max_l_trans - 1), (max_r_trans - 1))
            ty = random.uniform(-(max_u_trans - 1), (max_d_trans - 1))

            M = np.array([[1, 0, tx], [0, 1, ty]])
            image = cv2.warpAffine(image, M, (w, h))

            bboxes[:, [0, 2, 4, 6]] = bboxes[:, [0, 2, 4, 6]] + tx
            bboxes[:, [1, 3, 5, 7]] = bboxes[:, [1, 3, 5, 7]] + ty

        return image, bboxes

    def parse_annotation(self, annotation, mAP = 'False'):
         # annotaions[image_path, bboxes, image]
        if TRAIN_LOAD_IMAGES_TO_RAM:
            image_path = annotation[0]
            image = annotation[2]
        else:
            image_path = annotation[0]
            image = cv2.imread(image_path)

        bboxes = np.array([list(map(float, box.split(','))) for box in annotation[1]]) # 한 이미지의  bboxes 값들
        
        if self.data_aug:
            image, bboxes = self.random_horizontal_flip(np.copy(image), np.copy(bboxes))
            image, bboxes = self.random_crop(np.copy(image), np.copy(bboxes))
            image, bboxes = self.random_translate(np.copy(image), np.copy(bboxes))

        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if mAP == True:
            return image, bboxes

        image, bboxes = image_preprocess(np.copy(image), [self.input_sizes, self.input_sizes], np.copy(bboxes))
        return image, bboxes

    def preprocess_true_boxes(self, bboxes): # 한 이미지에 대한 여러개의 바운딩 박스를 받음
        OUTPUT_LEVELS = len(self.strides) # [8, 16, 32] 총 3개

        label = [np.zeros((self.train_output_sizes[i], self.train_output_sizes[i], self.anchor_per_scale, # train_output_sizes = [52, 26, 13]
                           12 + self.num_classes)) for i in range(OUTPUT_LEVELS)]
        # 52, 52, 3, 12 + self.num_classes
        bboxes_xywh = [np.zeros((self.max_bbox_per_scale, 4)) for _ in range(OUTPUT_LEVELS)] # [100, 4] 3개 생성
        bbox_count = np.zeros((OUTPUT_LEVELS,)) # [0, 0, 0]

        for bbox in bboxes:
            bbox_coor = bbox[:4] # xmin, ymin, xmax, ymax <0,1,2,3>
            bbox_junc = bbox[4:8]
            bbox_vector = bbox[8:10]
            bbox_occupancy = bbox[10]
            bbox_class_ind = np.uint8(bbox[11])
            
            onehot = np.zeros(self.num_classes, dtype=np.float)
            onehot[bbox_class_ind] = 1.0
            uniform_distribution = np.full(self.num_classes, 1.0 / self.num_classes)
            deta = 0.01
            smooth_onehot = onehot * (1 - deta) + deta * uniform_distribution
            
            bbox_xywh = np.concatenate([(bbox_coor[2:] + bbox_coor[:2]) * 0.5, bbox_coor[2:] - bbox_coor[:2]], axis=-1)
            # (xmax, ymax + xmin, ymin)*0.5, (xmax, ymax - xmin, ymin) = x, y, w, h
            
            bbox_xywh_scaled = 1.0 * bbox_xywh[np.newaxis, :] / self.strides[:, np.newaxis] # xywh / [8, 16, 32] -> 3개의 feature map에 해당하는 scale로 바꿈
            iou = []
            exist_positive = False
            for i in range(OUTPUT_LEVELS): # range(3):
                anchors_xywh = np.zeros((self.anchor_per_scale, 4))
                anchors_xywh[:, 0:2] = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32) + 0.5 # anchor의 중심점 x,y
                anchors_xywh[:, 2:4] = self.anchors[i] # anchor의 w, h

                iou_scale = bbox_iou(bbox_xywh_scaled[i][np.newaxis, :], anchors_xywh)
                iou.append(iou_scale)
                iou_mask = iou_scale > 0.3

                if np.any(iou_mask):
                    xind, yind = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32)
                    
                    label[i][yind, xind, iou_mask, :] = 0
                    label[i][yind, xind, iou_mask, 0:4] = bbox_xywh
                    label[i][yind, xind, iou_mask, 4:5] = 1.0 # confidence
                    label[i][yind, xind, iou_mask, 5:9] = bbox_junc
                    label[i][yind, xind, iou_mask, 9:11] = bbox_vector
                    label[i][yind, xind, iou_mask, 11] = bbox_occupancy
                    label[i][yind, xind, iou_mask, 12:] = smooth_onehot
                    # print(label[i][yind, xind, iou_mask, 9])
                    # label = x, y, w, h, 1.0, x1, y1, x2, y2 
                    
                    bbox_ind = int(bbox_count[i] % self.max_bbox_per_scale)
                    bboxes_xywh[i][bbox_ind, :4] = bbox_xywh
                    bbox_count[i] += 1

                    exist_positive = True

            if not exist_positive:
                best_anchor_ind = np.argmax(np.array(iou).reshape(-1), axis=-1)
                best_detect = int(best_anchor_ind / self.anchor_per_scale)
                best_anchor = int(best_anchor_ind % self.anchor_per_scale)
                xind, yind = np.floor(bbox_xywh_scaled[best_detect, 0:2]).astype(np.int32)

                label[best_detect][yind, xind, best_anchor, :] = 0
                label[best_detect][yind, xind, best_anchor, 0:4] = bbox_xywh
                label[best_detect][yind, xind, best_anchor, 4:5] = 1.0
                label[best_detect][yind, xind, best_anchor, 5:9] = bbox_junc
                label[best_detect][yind, xind, best_anchor, 9:11] = bbox_vector
                label[best_detect][yind, xind, best_anchor, 11] = bbox_occupancy
                label[best_detect][yind, xind, best_anchor, 12:] = smooth_onehot
                
                bbox_ind = int(bbox_count[best_detect] % self.max_bbox_per_scale)
                bboxes_xywh[best_detect][bbox_ind, :4] = bbox_xywh
                bbox_count[best_detect] += 1

        # if self.train_yolo_tiny:
        #     label_mbbox, label_lbbox = label
        #     mbboxes, lbboxes = bboxes_xywh
        #     return label_mbbox, label_lbbox, mbboxes, lbboxes

        label_sbbox, label_mbbox, label_lbbox = label
        sbboxes, mbboxes, lbboxes = bboxes_xywh
        return label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes

    def __len__(self):
        return self.num_batchs
