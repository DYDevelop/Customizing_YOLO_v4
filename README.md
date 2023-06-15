# Customizing YOLO v4 to Detect Parking Spaces

YOLOv3 and YOLOv4 implementation in TensorFlow 2.x, with support for training, transfer training, object tracking mAP and so on...
Code was tested with following specs:
- R5 5600X CPU and Nvidia RTX 3080 GPU
- OS Windows 11
- CUDA 11.7
- cuDNN v11.8
- Tensorflow-GPU 2.9.2
- Code was tested on only Windows 11
- You need custom_dataset to train the code (it's not an open source sorry T.T)

## Installation
First, clone or download this GitHub repository.
Install requirements:
```
pip install -r ./requirements.txt
```

## Looking into Custom Dataset
![Dataset](IMAGES/Dataset0.png)  
![Dataset](IMAGES/Dataset4.png)  
![Dataset](IMAGES/Dataset2.png)  
![Dataset](IMAGES/Dataset3.png)  

## Traing with Custom Dataset
![Train](IMAGES/Train0.png)  

## Decoding Code

![Train](IMAGES/Train1.png)  
![Train](IMAGES/Train1_1.png)  

## Loss Function
![Loss](IMAGES/Loss.png)  

## Training
![Training](IMAGES/Training.png)  

## Results
- Yellow : Slot is Occupied
- Red : Slot is Empty
  
![Result](IMAGES/Results0.png)  
![Result](IMAGES/Results1.png)  
![Result](IMAGES/Results2.png)  

