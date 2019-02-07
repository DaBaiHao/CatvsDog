from keras import models
import numpy as np
import cv2

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
