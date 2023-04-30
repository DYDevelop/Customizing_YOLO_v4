#================================================================
#
#   File name   : evaluate_mAP.py
#   Author      : PyLessons
#   Created date: 2020-08-17
#   Website     : https://pylessons.com/
#   GitHub      : https://github.com/pythonlessons/TensorFlow-2.x-YOLOv3
#   Description : used to evaluate model mAP and FPS
#
#================================================================
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
from yolov3.dataset import Dataset
from yolov3.yolov4 import Create_Yolo
from yolov3.utils import load_yolo_weights, detect_image, image_preprocess, postprocess_boxes, nms, read_class_names
from yolov3.configs import *
import shutil
import json
import time

gpus = tf.config.experimental.list_physical_devices('GPU')
if len(gpus) > 0:
    try: tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError: print("RuntimeError in tf.config.experimental.list_physical_devices('GPU')")


def voc_ap(rec, prec):
    """
    --- Official matlab code VOC2012---
    mrec=[0 ; rec ; 1];
    mpre=[0 ; prec ; 0];
    for i=numel(mpre)-1:-1:1
            mpre(i)=max(mpre(i),mpre(i+1));
    end
    i=find(mrec(2:end)~=mrec(1:end-1))+1;
    ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """
    rec.insert(0, 0.0) # insert 0.0 at begining of list
    rec.append(1.0) # insert 1.0 at end of list
    mrec = rec[:]
    prec.insert(0, 0.0) # insert 0.0 at begining of list
    prec.append(0.0) # insert 0.0 at end of list
    mpre = prec[:]
    """
     This part makes the precision monotonically decreasing
        (goes from the end to the beginning)
        matlab:  for i=numel(mpre)-1:-1:1
                                mpre(i)=max(mpre(i),mpre(i+1));
    """
    # matlab indexes start in 1 but python in 0, so I have to do:
    #   range(start=(len(mpre) - 2), end=0, step=-1)
    # also the python function range excludes the end, resulting in:
    #   range(start=(len(mpre) - 2), end=-1, step=-1)
    for i in range(len(mpre)-2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i+1])
    """
     This part creates a list of indexes where the recall changes
        matlab:  i=find(mrec(2:end)~=mrec(1:end-1))+1;
    """
    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i-1]:
            i_list.append(i) # if it was matlab would be i + 1
    """
     The Average Precision (AP) is the area under the curve
        (numerical integration)
        matlab: ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """
    ap = 0.0
    for i in i_list:
        ap += ((mrec[i]-mrec[i-1])*mpre[i])
    return ap, mrec, mpre


def get_mAP(Yolo, dataset, score_threshold=0.25, iou_threshold=0.50, TEST_INPUT_SIZE=TEST_INPUT_SIZE):
    MINOVERLAP = 0.5 # default value (defined in the PASCAL VOC2012 challenge)
    NUM_CLASS = read_class_names(TRAIN_CLASSES)

    ground_truth_dir_path = 'mAP/ground-truth'
    if os.path.exists(ground_truth_dir_path): shutil.rmtree(ground_truth_dir_path)

    if not os.path.exists('mAP'): os.mkdir('mAP')
    os.mkdir(ground_truth_dir_path)

    print(f'\ncalculating mAP{int(iou_threshold*100)}...\n')

    gt_counter_per_class = {}
    for index in range(dataset.num_samples):
        ann_dataset = dataset.annotations[index]

        original_image, bbox_data_gt = dataset.parse_annotation(ann_dataset, True)

        if len(bbox_data_gt) == 0:
            bboxes_gt = []
            classes_gt = []
        else:
            bboxes_gt, classes_gt = bbox_data_gt[:, :11], bbox_data_gt[:, 11]
        ground_truth_path = os.path.join(ground_truth_dir_path, str(index) + '.txt')
        num_bbox_gt = len(bboxes_gt)

        bounding_boxes = []
        for i in range(num_bbox_gt):
            class_name = NUM_CLASS[classes_gt[i]]
            xmin, ymin, xmax, ymax, x1, y1, x2, y2, vec_x, vec_y, occu = list(map(str, bboxes_gt[i]))
            bbox = xmin + " " + ymin + " " + xmax + " " + ymax + " " + x1 + " " + y1 + " " + x2 + " " + y2 + " " + vec_x + " " + vec_y + " " + occu
            bounding_boxes.append({"class_name":class_name, "bbox":bbox, "used":False})

            # count that object
            if class_name in gt_counter_per_class:
                gt_counter_per_class[class_name] += 1
            else:
                # if class didn't exist yet
                gt_counter_per_class[class_name] = 1
            bbox_mess = ' '.join([class_name, xmin, ymin, xmax, ymax, x1, y1, x2, y2, vec_x, vec_y, occu]) + '\n'
        with open(f'{ground_truth_dir_path}/{str(index)}_ground_truth.json', 'w') as outfile:
            json.dump(bounding_boxes, outfile)

    gt_classes = list(gt_counter_per_class.keys())
    # sort the classes alphabetically
    gt_classes = sorted(gt_classes)
    n_classes = len(gt_classes)

    times = []
    json_pred = [[] for i in range(n_classes)]
    for index in range(dataset.num_samples):
        ann_dataset = dataset.annotations[index]

        image_name = ann_dataset[0].split('/')[-1]
        original_image, bbox_data_gt = dataset.parse_annotation(ann_dataset, True)
        
        image = image_preprocess(np.copy(original_image), [TEST_INPUT_SIZE, TEST_INPUT_SIZE])
        image_data = image[np.newaxis, ...].astype(np.float32)

        t1 = time.time()
        if YOLO_FRAMEWORK == "tf":
            if tf.__version__ > '2.4.0':
                pred_bbox = Yolo(image_data)
            else:
                pred_bbox = Yolo.predict(image_data)
        elif YOLO_FRAMEWORK == "trt":
            batched_input = tf.constant(image_data)
            result = Yolo(batched_input)
            pred_bbox = []
            for key, value in result.items():
                value = value.numpy()
                pred_bbox.append(value)
        
        t2 = time.time()
        
        times.append(t2-t1)
        
        pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
        pred_bbox = tf.concat(pred_bbox, axis=0)

        bboxes = postprocess_boxes(pred_bbox, original_image, TEST_INPUT_SIZE, score_threshold)
        bboxes = nms(bboxes, iou_threshold, method='nms')
        # coors, scores, juncs, vectors, occupancy, classes 총 12개
        
        for bbox in bboxes:
            coor = np.array(bbox[:4], dtype=np.int32)
            score = bbox[4]
            junctions = np.array(bbox[5:9], dtype=np.int32)
            vector = np.array(bbox[9:12], dtype=np.float16)
            class_ind = int(bbox[12])
            class_name = NUM_CLASS[class_ind]
            score = '%.4f' % score
            xmin, ymin, xmax, ymax = list(map(str, coor))
            x1, y1, x2, y2 = list(map(str, junctions))
            vec_x, vec_y, occu = list(map(str, vector))
            bbox = xmin + " " + ymin + " " + xmax + " " + ymax + " " + x1 + " " + y1 + " " + x2 + " " + y2 + " " + vec_x + " " + vec_y + " " + occu
            json_pred[gt_classes.index(class_name)].append({"confidence": str(score), "file_id": str(index), "bbox": str(bbox)})

    ms = sum(times)/len(times)*1000
    fps = 1000 / ms

    for class_name in gt_classes:
        json_pred[gt_classes.index(class_name)].sort(key=lambda x:float(x['confidence']), reverse=True)
        with open(f'{ground_truth_dir_path}/{class_name}_predictions.json', 'w') as outfile:
            json.dump(json_pred[gt_classes.index(class_name)], outfile)

    # Calculate the AP for each class
    sum_AP = 0.0
    ap_dictionary = {}
    TP_in_detection = 0
    # open file to store the results
    with open("mAP/results.txt", 'w') as results_file:
        results_file.write("# AP and precision/recall per class\n")
        count_true_positives = {}
        for class_index, class_name in enumerate(gt_classes):
            count_true_positives[class_name] = 0
            # Load predictions of that class
            predictions_file = f'{ground_truth_dir_path}/{class_name}_predictions.json'
            predictions_data = json.load(open(predictions_file))

            # Assign predictions to ground truth objects
            nd = len(predictions_data)
            tp = [0] * nd # creates an array of zeros of size nd
            fp = [0] * nd
            for idx, prediction in enumerate(predictions_data):
                file_id = prediction["file_id"]
                # assign prediction to ground truth object if any
                #   open ground-truth with that file_id
                gt_file = f'{ground_truth_dir_path}/{str(file_id)}_ground_truth.json'
                ground_truth_data = json.load(open(gt_file))
                ovmax = -1
                gt_match = -1
                # load prediction bounding-box
                bb = [ float(x) for x in prediction["bbox"].split() ] # bounding box of prediction
                for obj in ground_truth_data:
                    # look for a class_name match
                    if obj["class_name"] == class_name:
                        bbgt = [ float(x) for x in obj["bbox"].split() ] # bounding box of ground truth
                        bi = [max(bb[0],bbgt[0]), max(bb[1],bbgt[1]), min(bb[2],bbgt[2]), min(bb[3],bbgt[3])]
                        iw = bi[2] - bi[0] + 1
                        ih = bi[3] - bi[1] + 1
                        if iw > 0 and ih > 0:
                            # compute overlap (IoU) = area of intersection / area of union
                            ua = (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1) + (bbgt[2] - bbgt[0]
                                            + 1) * (bbgt[3] - bbgt[1] + 1) - iw * ih
                            ov = iw * ih / ua
                            if ov > ovmax:
                                ovmax = ov
                                gt_match = obj

                # assign prediction as true positive/don't care/false positive
                if ovmax >= MINOVERLAP:# if ovmax > minimum overlap
                    if not bool(gt_match["used"]):
                        # true positive
                        tp[idx] = 1
                        gt_match["used"] = True
                        count_true_positives[class_name] += 1
                        # update the ".json" file
                        with open(gt_file, 'w') as f:
                            f.write(json.dumps(ground_truth_data))
                        TP_in_detection += 1
                    else:
                        # false positive (multiple detection)
                        fp[idx] = 1
                else:
                    # false positive
                    fp[idx] = 1

            # compute precision/recall
            cumsum = 0
            for idx, val in enumerate(fp):
                fp[idx] += cumsum
                cumsum += val
            cumsum = 0
            for idx, val in enumerate(tp):
                tp[idx] += cumsum
                cumsum += val
            #print(tp)
            rec = tp[:]
            for idx, val in enumerate(tp):
                rec[idx] = float(tp[idx]) / gt_counter_per_class[class_name] # Recall
            #print(rec)
            prec = tp[:]
            for idx, val in enumerate(tp):
                prec[idx] = float(tp[idx]) / (fp[idx] + tp[idx]) # Precision
            #print(prec)

            ap, mrec, mprec = voc_ap(rec, prec)
            sum_AP += ap
            text = "{0:.3f}%".format(ap*100) + " = " + class_name + " AP  " #class_name + " AP = {0:.2f}%".format(ap*100)

            rounded_prec = [ '%.3f' % elem for elem in prec ]
            rounded_rec = [ '%.3f' % elem for elem in rec ]
            # Write to results.txt
            results_file.write(text + "\n Precision: " + str(rounded_prec) + "\n Recall   :" + str(rounded_rec) + "\n\n")

            print(text)
            ap_dictionary[class_name] = ap

        results_file.write("\n# mAP of all classes\n")
        mAP = sum_AP / n_classes

        text = "mAP = {:.3f}%, {:.2f} FPS".format(mAP*100, fps)
        results_file.write(text + "\n")
        print(text)
        
        detection_TP_count = idx = GT_count = 0
        
        for _, class_name in enumerate(gt_classes):
            detection_TP_count += count_true_positives[class_name]
        
        # Assign predictions to ground truth objects
        for index in range(dataset.num_samples):
            gt_file = f'{ground_truth_dir_path}/{str(index)}_ground_truth.json'
            ground_truth_data = json.load(open(gt_file))
            
            for obj in ground_truth_data:
                bbgt = [ float(x) for x in obj["bbox"].split() ] # bounding box of prediction
                ovmax = -1
                for _, class_name in enumerate(gt_classes):
                    predictions_file = f'{ground_truth_dir_path}/{class_name}_predictions.json'
                    predictions_data = json.load(open(predictions_file))
                    for _, prediction in enumerate(predictions_data):
                        file_id = prediction["file_id"]
                        if np.uint16(file_id) == index:
                            bb = [ float(x) for x in prediction["bbox"].split() ] # bounding box of ground truth
                            bi = [max(bb[0],bbgt[0]), max(bb[1],bbgt[1]), min(bb[2],bbgt[2]), min(bb[3],bbgt[3])]
                            iw = bi[2] - bi[0] + 1
                            ih = bi[3] - bi[1] + 1
                            if iw > 0 and ih > 0:
                                # compute overlap (IoU) = area of intersection / area of union
                                ua = (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1) + (bbgt[2] - bbgt[0]
                                                + 1) * (bbgt[3] - bbgt[1] + 1) - iw * ih
                                ov = iw * ih / ua
                                if ov > ovmax:
                                    ovmax = ov
            
                # assign prediction as true positive/don't care/false positive
                if ovmax >= MINOVERLAP:# if ovmax > minimum overlap
                    GT_count += 1
        
        # Assign predictions to ground truth objects
        tp_junc1 = [0] * detection_TP_count # creates an array of zeros of size detection_TP_count
        fp_junc1 = [0] * detection_TP_count
        tp_junc2 = [0] * detection_TP_count # creates an array of zeros of size detection_TP_count
        fp_junc2 = [0] * detection_TP_count
        tp_vector = [0] * detection_TP_count # creates an array of zeros of size detection_TP_count
        fp_vector = [0] * detection_TP_count
        tp_occupancy = [0] * detection_TP_count # creates an array of zeros of size detection_TP_count
        fp_occupancy = [0] * detection_TP_count
        
        for _, class_name in enumerate(gt_classes):
            predictions_file = f'{ground_truth_dir_path}/{class_name}_predictions.json'
            predictions_data = json.load(open(predictions_file))
            for _, prediction in enumerate(predictions_data):
                file_id = prediction["file_id"]
                # assign prediction to ground truth object if any
                #   open ground-truth with that file_id
                gt_file = f'{ground_truth_dir_path}/{str(file_id)}_ground_truth.json'
                ground_truth_data = json.load(open(gt_file))
                ovmax = -1
                gt_match = []
                # load prediction bounding-box
                bb = [ float(x) for x in prediction["bbox"].split() ] # bounding box of prediction
                for obj in ground_truth_data:
                    # look for a class_name match
                    if obj["class_name"] == class_name:
                        bbgt = [ float(x) for x in obj["bbox"].split() ] # bounding box of ground truth
                        bi = [max(bb[0],bbgt[0]), max(bb[1],bbgt[1]), min(bb[2],bbgt[2]), min(bb[3],bbgt[3])]
                        iw = bi[2] - bi[0] + 1
                        ih = bi[3] - bi[1] + 1
                        if iw > 0 and ih > 0:
                            # compute overlap (IoU) = area of intersection / area of union
                            ua = (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1) + (bbgt[2] - bbgt[0]
                                            + 1) * (bbgt[3] - bbgt[1] + 1) - iw * ih
                            ov = iw * ih / ua
                            if ov > ovmax:
                                ovmax = ov
                                gt_match = bbgt

                # assign prediction as true positive/don't care/false positive
                if ovmax >= MINOVERLAP:# if ovmax > minimum overlap
                    junc_pred =  np.array(bb[4:8], dtype=np.int32) # coors, scores, juncs, vectors, occupancy, classes
                    junc_label = np.array(gt_match[4:8], dtype=np.int32) # coors, juncs, vectors, occupancy, classes
                    junc1_dis = np.sqrt(np.sum(np.power(junc_pred[0:2] - junc_label[0:2], 2)))
                    junc2_dis = np.sqrt(np.sum(np.power(junc_pred[2:4] - junc_label[2:4], 2)))                    
                    if junc1_dis < 20: # 두 점 사이의 거리가 20미만일 경우 TP
                        tp_junc1[idx] = 1
                    else:                        
                        fp_junc1[idx] = 1
                    if junc2_dis < 20: # 두 점 사이의 거리가 20미만일 경우 TP
                        tp_junc2[idx] = 1
                    else:
                        fp_junc2[idx] = 1
                        
                    vector_pred = np.array(bb[8:10], dtype=np.float16)
                    vector_label = np.array(gt_match[8:10], dtype=np.float16)
                    angle = np.arccos((vector_pred[0] * vector_label[0] + vector_pred[1] * vector_label[1]) / 
                                      (np.sqrt(np.sum(np.power(vector_pred, 2))) * np.sqrt(np.sum(np.power(vector_label, 2))) + 0.001))
                    if angle < 10 * (np.pi / 180): # 두 벡터사이의 각이 10도 미만일때 TP
                        tp_vector[idx] = 1
                    else:
                        # print(angle)
                        fp_vector[idx] = 1
                    occupancy_pred = np.array(bb[10], dtype=np.float16)
                    occupancy_label = np.array(gt_match[10], dtype=np.float16)
                    if np.abs(occupancy_label - occupancy_pred) < 0.5: # Pred한 점유 확률과 label과의 차가 0.3 미만 일때 TP
                        tp_occupancy[idx] = 1
                    else:
                        fp_occupancy[idx] = 1
                    idx += 1
                    
        # compute precision/recall
        cumsum = 0
        for idx, val in enumerate(fp_junc1):
            fp_junc1[idx] += cumsum
            cumsum += val
        # print(fp_junc)
        cumsum = 0
        for idx, val in enumerate(tp_junc1):
            tp_junc1[idx] += cumsum
            cumsum += val
        # print(tp_junc)
        rec = tp_junc1[:]
        for idx, val in enumerate(tp_junc1):
            rec[idx] = float(tp_junc1[idx]) / GT_count # Recall
        #print(rec)
        prec = tp_junc1[:]
        for idx, val in enumerate(tp_junc1):
            prec[idx] = float(tp_junc1[idx]) / (fp_junc1[idx] + tp_junc1[idx]) # Precision
        #print(prec)

        junction1_ap, mrec, mprec = voc_ap(rec, prec)
        sum_AP += junction1_ap
        
        # compute precision/recall
        cumsum = 0
        for idx, val in enumerate(fp_junc2):
            fp_junc2[idx] += cumsum
            cumsum += val
        # print(fp_junc)
        cumsum = 0
        for idx, val in enumerate(tp_junc2):
            tp_junc2[idx] += cumsum
            cumsum += val
        # print(tp_junc)
        rec = tp_junc2[:]
        for idx, val in enumerate(tp_junc2):
            rec[idx] = float(tp_junc2[idx]) / GT_count # Recall
        #print(rec)
        prec = tp_junc2[:]
        for idx, val in enumerate(tp_junc2):
            prec[idx] = float(tp_junc2[idx]) / (fp_junc2[idx] + tp_junc2[idx]) # Precision
        #print(prec)

        junction2_ap, mrec, mprec = voc_ap(rec, prec)
        sum_AP += junction2_ap
        
        cumsum = 0
        for idx, val in enumerate(fp_vector):
            fp_vector[idx] += cumsum
            cumsum += val
        cumsum = 0
        for idx, val in enumerate(tp_vector):
            tp_vector[idx] += cumsum
            cumsum += val
        #print(tp)
        rec = tp_vector[:]
        for idx, val in enumerate(tp_vector):
            rec[idx] = float(tp_vector[idx]) / GT_count # Recall
        #print(rec)
        prec = tp_vector[:]
        for idx, val in enumerate(tp_vector):
            prec[idx] = float(tp_vector[idx]) / (fp_vector[idx] + tp_vector[idx]) # Precision
        #print(prec)

        vector_ap, mrec, mprec = voc_ap(rec, prec)
        sum_AP += vector_ap
        
        cumsum = 0
        for idx, val in enumerate(fp_occupancy):
            fp_occupancy[idx] += cumsum
            cumsum += val
        cumsum = 0
        for idx, val in enumerate(tp_occupancy):
            tp_occupancy[idx] += cumsum
            cumsum += val
        #print(tp)
        rec = tp_occupancy[:]
        for idx, val in enumerate(tp_occupancy):
            rec[idx] = float(tp_occupancy[idx]) / GT_count # Recall
        #print(rec)
        prec = tp_occupancy[:]
        for idx, val in enumerate(tp_occupancy):
            prec[idx] = float(tp_occupancy[idx]) / (fp_occupancy[idx] + tp_occupancy[idx]) # Precision
        #print(prec)

        occupancy_ap, mrec, mprec = voc_ap(rec, prec)
        sum_AP += occupancy_ap
        
        print("\n\nFirst Junction AP :{:7.2f}%, Second Junction AP :{:7.2f}%, Vector AP :{:7.2f}%, Occupancy AP :{:7.2f}%, mAP :{:7.2f}%\n\n".
            format(junction1_ap * 100, junction2_ap * 100, vector_ap * 100, occupancy_ap * 100, 
                   (junction1_ap + junction2_ap + vector_ap + occupancy_ap) / 4.0 * 100))
    
        return mAP*100

if __name__ == '__main__':       
    if YOLO_FRAMEWORK == "tf": # TensorFlow detection
        # if YOLO_TYPE == "yolov4":
        #     Darknet_weights = YOLO_V4_TINY_WEIGHTS if TRAIN_YOLO_TINY else YOLO_V4_WEIGHTS
        # if YOLO_TYPE == "yolov3":
        #     Darknet_weights = YOLO_V3_TINY_WEIGHTS if TRAIN_YOLO_TINY else YOLO_V3_WEIGHTS

        # if YOLO_CUSTOM_WEIGHTS == False:
        #     yolo = Create_Yolo(input_size=YOLO_INPUT_SIZE, CLASSES=YOLO_COCO_CLASSES)
        #     load_yolo_weights(yolo, Darknet_weights) # use Darknet weights
        # else:
            yolo = Create_Yolo(input_size=YOLO_INPUT_SIZE, CLASSES=TRAIN_CLASSES)
            yolo.load_weights(f"./checkpoints/{TRAIN_MODEL_NAME}") # use custom weights
        
    elif YOLO_FRAMEWORK == "trt": # TensorRT detection
        saved_model_loaded = tf.saved_model.load(f"./checkpoints/{TRAIN_MODEL_NAME}", tags=[tag_constants.SERVING])
        signature_keys = list(saved_model_loaded.signatures.keys())
        yolo = saved_model_loaded.signatures['serving_default']

    testset = Dataset('test', TEST_INPUT_SIZE=YOLO_INPUT_SIZE)
    get_mAP(yolo, testset, score_threshold=TEST_SCORE_THRESHOLD, iou_threshold=TEST_IOU_THRESHOLD, TEST_INPUT_SIZE=YOLO_INPUT_SIZE)
