#================================================================
#
#   File name   : train.py
#   Author      : PyLessons
#   Created date: 2020-08-06
#   Website     : https://pylessons.com/
#   GitHub      : https://github.com/pythonlessons/TensorFlow-2.x-YOLOv3
#   Description : used to train custom object detector
#
#================================================================
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
import shutil
import numpy as np
import tensorflow as tf
#from tensorflow.keras.utils import plot_model
from yolov3.dataset import Dataset
from yolov3.yolov4 import Create_Yolo, compute_loss
from yolov3.utils import *
from yolov3.configs import *
from evaluate_mAP import *
    
if YOLO_TYPE == "yolov4":
    Darknet_weights = YOLO_V4_TINY_WEIGHTS if TRAIN_YOLO_TINY else YOLO_V4_WEIGHTS
if YOLO_TYPE == "yolov3":
    Darknet_weights = YOLO_V3_TINY_WEIGHTS if TRAIN_YOLO_TINY else YOLO_V3_WEIGHTS
if TRAIN_YOLO_TINY: TRAIN_MODEL_NAME += "_Tiny"

def main():
    global TRAIN_FROM_CHECKPOINT
    
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print(f'GPUs {gpus}')
    if len(gpus) > 0:
        try: tf.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError: pass

    if os.path.exists(TRAIN_LOGDIR): shutil.rmtree(TRAIN_LOGDIR)
    writer = tf.summary.create_file_writer(TRAIN_LOGDIR)

    trainset = Dataset('train')
    testset = Dataset('test')
    
    steps_per_epoch = len(trainset)
    global_steps = tf.Variable(1, trainable=False, dtype=tf.int64)
    warmup_steps = TRAIN_WARMUP_EPOCHS * steps_per_epoch
    total_steps = TRAIN_EPOCHS * steps_per_epoch

    if TRAIN_TRANSFER:
        Darknet = Create_Yolo(input_size=YOLO_INPUT_SIZE, CLASSES=YOLO_COCO_CLASSES)
        load_yolo_weights(Darknet, Darknet_weights) # use darknet weights

    yolo = Create_Yolo(input_size=YOLO_INPUT_SIZE, training=True, CLASSES=TRAIN_CLASSES)
    if TRAIN_FROM_CHECKPOINT:
        try:
            yolo.load_weights(f"./checkpoints/{TRAIN_MODEL_NAME}")
        except ValueError:
            print("Shapes are incompatible, transfering Darknet weights")
            TRAIN_FROM_CHECKPOINT = False

    if TRAIN_TRANSFER and not TRAIN_FROM_CHECKPOINT:
        for i, l in enumerate(Darknet.layers):
            layer_weights = l.get_weights()
            if layer_weights != []:
                try:
                    yolo.layers[i].set_weights(layer_weights)
                except:
                    print("skipping", yolo.layers[i].name)
    
    optimizer = tf.keras.optimizers.Adam()

    def train_step(image_data, target):
        with tf.GradientTape() as tape:
            pred_result = yolo(image_data, training=True)
            giou_loss=conf_loss=coor_loss=angle_loss=occu_loss=prob_loss=0

            # optimizing process
            grid = 3 if not TRAIN_YOLO_TINY else 2
            for i in range(grid): # 3번 반복문
                conv, pred = pred_result[i*2], pred_result[i*2+1] # small, mediaum, large
                loss_items = compute_loss(pred, conv, *target[i], i, CLASSES=TRAIN_CLASSES)
                giou_loss += loss_items[0]
                conf_loss += loss_items[1]
                coor_loss += loss_items[2]
                angle_loss += loss_items[3]
                occu_loss += loss_items[4]
                prob_loss += loss_items[5]

            total_loss = giou_loss + conf_loss + coor_loss + angle_loss + occu_loss + prob_loss

            gradients = tape.gradient(total_loss, yolo.trainable_variables)
            optimizer.apply_gradients(zip(gradients, yolo.trainable_variables))

            # update learning rate
            # about warmup: https://arxiv.org/pdf/1812.01187.pdf&usg=ALkJrhglKOPDjNt6SHGbphTHyMcT0cuMJg
            global_steps.assign_add(1)
            if global_steps < warmup_steps:# and not TRAIN_TRANSFER:
                lr = global_steps / warmup_steps * TRAIN_LR_INIT
            else:
                lr = TRAIN_LR_END + 0.5 * (TRAIN_LR_INIT - TRAIN_LR_END)*(
                    (1 + tf.cos((global_steps - warmup_steps) / (total_steps - warmup_steps) * np.pi)))
                
            # lr = TRAIN_LR_INIT - global_steps / total_steps * (TRAIN_LR_INIT - TRAIN_LR_END)
            
            # 여기 조금 변경함 lr 고정시킴
            # optimizer.lr.assign(TRAIN_LR_INIT)
            optimizer.lr.assign(lr.numpy())

            # writing summary data
            with writer.as_default():
                tf.summary.scalar("lr", optimizer.lr, step=global_steps)
                tf.summary.scalar("loss/total_loss", total_loss, step=global_steps)
                tf.summary.scalar("loss/giou_loss", giou_loss, step=global_steps)
                tf.summary.scalar("loss/conf_loss", conf_loss, step=global_steps)
                tf.summary.scalar("loss/coor_loss", coor_loss, step=global_steps)
                tf.summary.scalar("loss/angle_loss", angle_loss, step=global_steps)
                tf.summary.scalar("loss/occu_loss", occu_loss, step=global_steps)
                tf.summary.scalar("loss/prob_loss", prob_loss, step=global_steps)
            writer.flush()
            
        return global_steps.numpy(), optimizer.lr.numpy(), giou_loss.numpy(), conf_loss.numpy(), coor_loss.numpy(), angle_loss.numpy(), occu_loss.numpy(), prob_loss.numpy(), total_loss.numpy()

    validate_writer = tf.summary.create_file_writer(TRAIN_LOGDIR)
    def validate_step(image_data, target):
        with tf.GradientTape() as tape:
            pred_result = yolo(image_data, training=False)
            giou_loss=conf_loss=coor_loss=angle_loss=occu_loss=prob_loss=0

            # optimizing process
            grid = 3 if not TRAIN_YOLO_TINY else 2
            for i in range(grid):
                conv, pred = pred_result[i*2], pred_result[i*2+1]
                loss_items = compute_loss(pred, conv, *target[i], i, CLASSES=TRAIN_CLASSES)
                giou_loss += loss_items[0]
                conf_loss += loss_items[1]
                coor_loss += loss_items[2]
                angle_loss += loss_items[3]
                occu_loss += loss_items[4]
                prob_loss += loss_items[5]

            total_loss = giou_loss + conf_loss + coor_loss + angle_loss + occu_loss + prob_loss
            
        return giou_loss.numpy(), conf_loss.numpy(), coor_loss.numpy(), angle_loss.numpy(), occu_loss.numpy(), prob_loss.numpy(), total_loss.numpy()

    mAP_model = Create_Yolo(input_size=YOLO_INPUT_SIZE, CLASSES=TRAIN_CLASSES) # create second model to measure mAP

    best_val_loss = 10000 # should be large at start
    for epoch in range(TRAIN_EPOCHS):
        for image_data, target in trainset:
            results = train_step(image_data, target)
            cur_step = results[0]%steps_per_epoch
            print("epoch:{:2.0f} step:{:5.0f}/{}, lr:{:.6f}, giou_loss:{:7.2f}, conf_loss:{:7.2f}, coor_loss:{:7.2f}, angle_loss:{:7.2f}, occu_loss:{:7.2f}, prob_loss:{:7.2f}, total_loss:{:7.2f}"
                  .format(epoch, cur_step, steps_per_epoch, results[1], results[2], results[3], results[4], results[5], results[6], results[7], results[8]))

        if len(testset) == 0:
            print("configure TEST options to validate model")
            yolo.save_weights(os.path.join(TRAIN_CHECKPOINTS_FOLDER, TRAIN_MODEL_NAME))
            continue
        
        count, giou_val, conf_val, coor_val, angle_val, occu_val, prob_val, total_val = 0., 0, 0, 0, 0, 0, 0, 0
        for image_data, target in testset:
            results = validate_step(image_data, target)
            count += 1
            giou_val += results[0]
            conf_val += results[1]
            coor_val += results[2]
            angle_val += results[3]
            occu_val += results[4]
            prob_val += results[5]
            total_val += results[6]
            
        # writing validate summary data
        with validate_writer.as_default():
            tf.summary.scalar("validate_loss/total_val", total_val/count, step=epoch)
            tf.summary.scalar("validate_loss/giou_val", giou_val/count, step=epoch)
            tf.summary.scalar("validate_loss/conf_val", conf_val/count, step=epoch)
            tf.summary.scalar("validate_loss/coor_val", coor_val/count, step=epoch)
            tf.summary.scalar("validate_loss/angle_val", angle_val/count, step=epoch)
            tf.summary.scalar("validate_loss/occu_val", occu_val/count, step=epoch)
            tf.summary.scalar("validate_loss/prob_val", prob_val/count, step=epoch)
        validate_writer.flush()
            
        print("\n\ngiou_val_loss:{:7.2f}, conf_val_loss:{:7.2f}, coor_val_loss:{:7.2f}, angle_loss:{:7.2f}, occu_loss:{:7.2f}, prob_loss:{:7.2f}, total_val_loss:{:7.2f}\n\n".
              format(giou_val/count, conf_val/count, coor_val/count, angle_val/count, occu_val/count, prob_val/count, total_val/count))

        if TRAIN_SAVE_CHECKPOINT and not TRAIN_SAVE_BEST_ONLY:
            save_directory = os.path.join(TRAIN_CHECKPOINTS_FOLDER, TRAIN_MODEL_NAME+"_val_loss_{:7.2f}".format(total_val/count))
            yolo.save_weights(save_directory)
        if TRAIN_SAVE_BEST_ONLY and best_val_loss>total_val/count:
            save_directory = os.path.join(TRAIN_CHECKPOINTS_FOLDER, TRAIN_MODEL_NAME)
            yolo.save_weights(save_directory)
            best_val_loss = total_val/count
        if not TRAIN_SAVE_BEST_ONLY and not TRAIN_SAVE_CHECKPOINT:
            save_directory = os.path.join(TRAIN_CHECKPOINTS_FOLDER, TRAIN_MODEL_NAME)
            yolo.save_weights(save_directory)

    # measure mAP of trained custom model
    try:
        save_directory = os.path.join(TRAIN_CHECKPOINTS_FOLDER, TRAIN_MODEL_NAME)
        mAP_model.load_weights(save_directory) # use keras weights
        get_mAP(mAP_model, testset, score_threshold=TEST_SCORE_THRESHOLD, iou_threshold=TEST_IOU_THRESHOLD)
        
        TP_pre=FP_pre=FN_re=occu_acc_pre=occu_acc_re=TP_re=0
        Junction_model = Create_Yolo(input_size=YOLO_INPUT_SIZE, CLASSES=TRAIN_CLASSES)
        Junction_model.load_weights(save_directory)
        annotations = testset.annotations
        test_len = testset.num_samples
        for ID in range(test_len):
            original_image, bbox_data_gt = testset.parse_annotation(annotations[ID], True)
            image_data = image_preprocess(np.copy(original_image), [TEST_INPUT_SIZE, TEST_INPUT_SIZE])
            image_data = image_data[np.newaxis, ...].astype(np.float32)
            pred_bbox = Junction_model(image_data)
            
            pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
            pred_bbox = tf.concat(pred_bbox, axis=0)

            bboxes = postprocess_boxes(pred_bbox, original_image, TEST_INPUT_SIZE, TEST_SCORE_THRESHOLD)
            bboxes = nms(bboxes, TEST_IOU_THRESHOLD, method='nms')
             
            for bb in enumerate(bboxes):
                bb = bb[1]
                ovmax = -1
                gt_match = []
                for bbgt in enumerate(bbox_data_gt):
                    bbgt = bbgt[1]
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
                # confidence > 0.5 & Iou > 0.5 & junctions distance < 20 & angle diff < 10
                if ovmax >= 0.5:
                    junc_pred =  np.array(bb[5:9], dtype=np.int32) # coors, scores, juncs, vectors, occupancy, classes
                    junc_label = np.array(gt_match[4:8], dtype=np.int32) # coors, juncs, vectors, occupancy, classes
                    junc1_dis = np.sqrt(np.sum(np.power(junc_pred[0:2] - junc_label[0:2], 2)))
                    junc2_dis = np.sqrt(np.sum(np.power(junc_pred[2:4] - junc_label[2:4], 2)))                    
                    vector_pred = np.array(bb[9:11], dtype=np.float16)
                    vector_label = np.array(gt_match[8:10], dtype=np.float16)
                    angle = np.arccos((vector_pred[0] * vector_label[0] + vector_pred[1] * vector_label[1]) / 
                                        (np.sqrt(np.sum(np.power(vector_pred, 2))) * np.sqrt(np.sum(np.power(vector_label, 2))) + 0.001))
                    if junc1_dis < 20 and junc2_dis < 20 and angle < 10 * (np.pi / 180):
                        TP_pre += 1
                        occupancy_pred = np.array(bb[11], dtype=np.float16)
                        occupancy_label = np.array(gt_match[10], dtype=np.float16)
                        if np.abs(occupancy_label - occupancy_pred) < 0.5:
                            occu_acc_pre += 1 # Occupancy TP + TN
                    else:
                        FP_pre += 1
            
            for bbgt in enumerate(bbox_data_gt):
                bbgt = bbgt[1]
                ovmax = -1
                pred_match = []
                for bb in enumerate(bboxes):
                    bb = bb[1]
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
                            pred_match = bb
                # confidence > 0.5 & Iou > 0.5 & junctions distance < 20 & angle diff < 10
                if ovmax >= 0.5:
                    junc_pred =  np.array(pred_match[5:9], dtype=np.int32) # coors, scores, juncs, vectors, occupancy, classes
                    junc_label = np.array(bbgt[4:8], dtype=np.int32) # coors, juncs, vectors, occupancy, classes
                    junc1_dis = np.sqrt(np.sum(np.power(junc_pred[0:2] - junc_label[0:2], 2)))
                    junc2_dis = np.sqrt(np.sum(np.power(junc_pred[2:4] - junc_label[2:4], 2)))                    
                    vector_pred = np.array(pred_match[9:11], dtype=np.float16)
                    vector_label = np.array(bbgt[8:10], dtype=np.float16)
                    angle = np.arccos((vector_pred[0] * vector_label[0] + vector_pred[1] * vector_label[1]) / 
                                        (np.sqrt(np.sum(np.power(vector_pred, 2))) * np.sqrt(np.sum(np.power(vector_label, 2))) + 0.001))
                    if junc1_dis < 20 and junc2_dis < 20 and angle < 10 * (np.pi / 180):
                        TP_re += 1
                        occupancy_pred = np.array(pred_match[11], dtype=np.float16)
                        occupancy_label = np.array(bbgt[10], dtype=np.float16)
                        if np.abs(occupancy_label - occupancy_pred) < 0.5:
                            occu_acc_re += 1 # Occupancy TP + TN
                    else:
                        FN_re += 1
                        
        print("\n\nA parking slot is considered as a true positive if it satisfies under these conditions\n\n")
        print("1. confidence rate of the detected slot must exceed 0.5\n")
        print("2. Iou with the ground truth must exceed 0.5\n")
        print("3. two junctions are within 20 pixels from their ground truth locations\n")
        print("4. orientation is within 10 Degrees from the ground truth orientation\n")
        print("\nPrecison :{:7.2f}%, Recall :{:7.2f}%, Occupancy Classification rate :{:7.2f}%\n\n".
            format(TP_pre / (TP_pre + FP_pre) * 100, TP_re / (TP_re + FN_re) * 100, (occu_acc_re / TP_re) * 100))
        
        
    except UnboundLocalError:
        print("You don't have saved model weights to measure mAP, check TRAIN_SAVE_BEST_ONLY and TRAIN_SAVE_CHECKPOINT lines in configs.py")
        
if __name__ == '__main__':
    main()