from keras import models
import numpy as np
import cv2
import matplotlib.pyplot as plt

def get_inputs(src = []):
    pre_x = []
    for s in src:
        input = cv2.imread(s)
        # can be modify
        input = cv2.resize(input, (128,128))
        input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
        pre_x.append(input) # append one image
    # 标准化？？？？
    pre_x = np.array(pre_x)/255.0
    return pre_x


def put_prey(pre_y, label):
    output = []
    for y in pre_y:
        if y[0]<0.5: # 二分类，此处只用一个神经元输出
            output.append([label[0],1-y[0]])
        else:
            output.append([label[1], y[0]])
    return output


model = models.load_model('outputs/catdogs_model.h5')

pre_x = get_inputs(['test1/1.jpg','test1/11.jpg'])

pre_y = model.predict(pre_x)

import morph as mp
output = put_prey(pre_y,list(mp.train_flow.class_indices.keys()))
print(output)
