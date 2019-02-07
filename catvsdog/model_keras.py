from keras.models import Sequential
from keras.layers import Convolution2D,MaxPool2D,Flatten,Dense,Dropout
from keras.callbacks import TensorBoard

model=Sequential([
    Convolution2D(32,3,3,input_shape=(128,128,3),activation='relu'),
    MaxPool2D(pool_size=(2,2)),
    Convolution2D(64,3,3,input_shape=(128,128,3),activation='relu'),
    MaxPool2D(pool_size=(2,2)),
    Flatten(),
    Dense(64,activation='relu'),
    Dropout(0.5),
    Dense(1,activation='sigmoid')
])

model.summary()

model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])

import morph as morph#引用上文1的数据增加代码

model.fit_generator(
    morph.train_flow,steps_per_epoch=100,epochs=50,verbose=1,validation_data=morph.test_flow,validation_steps=100,
    callbacks=[TensorBoard(log_dir='./logs/1')]
)

model.save('outputs/catdogs_model.h5')
