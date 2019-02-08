from keras.applications import VGG16
from keras.models import Sequential,Model
from keras.layers import Flatten,Dense
from sklearn.decomposition import PCA

########### 加载原模型 ###############
conv_base=VGG16(weights='imagenet',include_top=False,input_shape=(128,128,3))

model = Sequential()
model.add(conv_base)
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.trainable=False
model.summary()

model.load_weights('outputs/weights_vgg16_use.h5')

###构建cnn_svm模型，并完成训练#############



import morph as mp
dense1_layer_model = Model(inputs=model.input,outputs=model.layers[-2].output)

from sklearn.svm import SVC
import numpy as np
clf=SVC()


def predict_cnnsvm(k,pca=None):#测试批数，共用数据32*k
    s,al=0,0
    for i in range(k):
        res=mp.test_flow.next()
        x,y=res[0],res[1]
        x_temp=dense1_layer_model.predict(x)
        if pca!=None:
            y_temp=clf.predict(pca.transform(x_temp))
        else:
            y_temp = clf.predict(x_temp)
        s+=np.sum(y_temp==y)
        al+=len(y)
    return s*1.0/al


X = np.ones((0,256))#原特征维度个数256
Y = np.array([])
for i in range(100): #共用数据100*32个
    res=mp.train_flow.next()
    x,y=res[0],res[1]
    x_temp=dense1_layer_model.predict(x)
    X=np.row_stack((X,x_temp))
    Y=np.append(Y,y)
    print("%d inserted!"%(i*32+32))
print(X.shape)
print(Y.shape)

print("no use pca:")
clf.fit(X,Y)
for _ in range(5):
    pre=predict_cnnsvm(20)
    print("correct_rate:%.3f"%(pre))

print("use pca:")
pca = PCA(n_components=10)
pca.fit(X)
X_new = pca.transform(X)
clf.fit(X_new,Y)
for _ in range(5):
    pre=predict_cnnsvm(20,pca)
    print("correct_rate:%.3f"%(pre))
