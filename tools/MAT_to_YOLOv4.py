import os
import glob
import numpy as np
from scipy import io

foldername = os.path.basename(os.getcwd())
if foldername == "tools": os.chdir("..")


data_dir = '/custom_dataset/'
Dataset_names_path = "model_data/parking_space_names.txt"
Dataset_train = "model_data/parking_space_train.txt"
Dataset_test = "model_data/parking_space_test.txt"
is_subfolder = False

Dataset_names = []
      
def ParseMAT(img_folder, file):
    img_paths = sorted([os.path.join(img_folder,fname) 
                            for fname in os.listdir(img_folder)
                            if fname.endswith('.jpg')])
    ann_paths = sorted([os.path.join(img_folder,fname) 
                            for fname in os.listdir(img_folder)
                            if fname.endswith('.mat') and not fname.startswith('.')])
    print(len(img_paths))

    for l in range(len(img_paths)//2):
        annotation =io.loadmat(ann_paths[l])
        img_path = img_paths[l]
        marks = annotation.get('marks')
        slots = annotation.get('slots')
        sl = slots.shape[0]
        if sl==0:
            continue
        for i in range (sl):
            s1 = np.uint16(slots[i][0]) # slot의 첫번째 mark_1
            s2 = np.uint16(slots[i][1]) # slot의 두번째 mark_2
            vecter_x = np.float16(slots[i][4])
            vecter_y = np.float16(slots[i][5])
            cls_id = np.uint8(slots[i][7]) - 1
            occupancy = np.uint8(slots[i][8])
            m1 = np.uint16(marks[s1-1][0:2]) # mark_1의 좌표
            m2 = np.uint16(marks[s2-1][0:2]) # mark_2의 좌표
            margin = 20
            xmin = m1[0]-margin if not m1[0]>=m2[0] else m2[0]-margin
            ymin = m1[1]-margin if not m1[1]>=m2[1] else m2[1]-margin
            xmax = m1[0]+margin if not m1[0]<m2[0] else m2[0]+margin
            ymax = m1[1]+margin if not m1[1]<m2[1] else m2[1]+margin
            x1 = m1[0] # mark_1의 x 좌표
            y1 = m1[1] # mark_1의 y 좌표
            x2 = m2[0] # mark_2의 x 좌표
            y2 = m2[1] # mark_2의 y 좌표
            OBJECT = (str(xmin)+','
                      +str(ymin)+','
                      +str(xmax)+','
                      +str(ymax)+','
                      +str(x1)+','
                      +str(y1)+','
                      +str(x2)+','
                      +str(y2)+','
                      +str(vecter_x)+','
                      +str(vecter_y)+','
                      +str(occupancy)+','
                      +str(cls_id))
            img_path += ' '+OBJECT
        print(img_path)
        file.write(img_path+'\n')

def run_MAT_to_YOLOv4():
    for i, folder in enumerate(['training','testing']):
        with open([Dataset_train,Dataset_test][i], "w") as file:
            print(os.getcwd()+data_dir+folder)
            img_path = os.path.join(os.getcwd()+data_dir+folder)
            ParseMAT(img_path, file)
    Dataset_names = ["Perpendicular", "Parallel", "Oblique"]
    print("Dataset_names:", Dataset_names)
    with open(Dataset_names_path, "w") as file:
        for name in Dataset_names:
            file.write(str(name)+'\n')

run_MAT_to_YOLOv4()