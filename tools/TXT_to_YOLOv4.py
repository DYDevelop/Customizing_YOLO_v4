#================================================================
#
#   File name   : XML_to_YOLOv3.py
#   Author      : PyLessons
#   Created date: 2020-06-04
#   Website     : https://pylessons.com/
#   GitHub      : https://github.com/pythonlessons/TensorFlow-2.x-YOLOv3
#   Description : used to convert XML labels to YOLOv3 training labels
#
#================================================================
import os
import glob

foldername = os.path.basename(os.getcwd())
if foldername == "tools": os.chdir("..")


data_dir = '/custom_dataset/'
Dataset_names_path = "model_data/car_names.txt"
Dataset_total = "model_data/car_total.txt"
Dataset_train = "model_data/car_train.txt"
Dataset_test = "model_data/car_test.txt"

Dataset_names = []
      
def ParseXML(img_folder, file):
    for txt_file in glob.glob(img_folder+'/*.txt'):
        tree = open(txt_file)
        image_name = txt_file.split('/')[-1].split('\\')[-1].split('.')[0]
        img_path = img_folder+'/'+image_name+'.png'
        for line in tree.readlines():
            num = line.split(' ')
            cls = int(float(num[0]))
            if cls not in Dataset_names:
                    Dataset_names.append(cls)
            cls_id = Dataset_names.index(cls)
            OBJECT = (str(int(float(num[1])))+','
                      +str(int(float(num[2])))+','
                      +str(int(float(num[5])))+','
                      +str(int(float(num[6])))+','
                      +str(cls_id))
            img_path += ' '+OBJECT
        print(img_path)
        file.write(img_path+'\n')
        
def make_val_set(totalfile, trainfile, testfile):
    print('Spliting Trainset and Testset from Total Dataset')
    total = open(totalfile)
    with open(testfile, 'w') as test, open(trainfile, 'w') as train:
        for i, line in enumerate(total.readlines()):
            if i % 10 == 0: test.write(line)
            else: train.write(line)
    print('Done!')
    

def run_TXT_to_YOLOv3():
    for folder in (['train']):
        with open(Dataset_total, "w") as file:
            print(os.getcwd()+data_dir+folder)
            img_path = os.path.join(os.getcwd()+data_dir+folder)
            ParseXML(img_path, file)

    print("Dataset_names:", Dataset_names)
    with open(Dataset_names_path, "w") as file:
        for name in Dataset_names:
            file.write(str(name)+'\n')

run_TXT_to_YOLOv3()
make_val_set(Dataset_total, Dataset_train, Dataset_test)
