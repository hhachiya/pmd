import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

import pdb

#data_dir = '../data/theta'
data_dir = '../data/PMD_split/PMD/train'

# output path
outpath = os.path.join(data_dir,'edge') 
if not os.path.exists(outpath):
    os.mkdir(outpath)

# input path
y_train_dir = os.path.join(data_dir,'mask')
Y_train = os.listdir(y_train_dir)

# edge makiking
print("making now ...")
for filename in Y_train:

    filepath = os.path.join(y_train_dir,filename)

    image = cv2.imread(filepath)
    edge = cv2.Canny(image, 100,200) # 疑問点なぜ100,200なのかがわからない。
    dilate = cv2.dilate(edge, (6,6), iterations=4)
    cv2.imwrite(os.path.join(outpath,filename), dilate)
print("finished!")






