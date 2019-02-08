from keras.applications import VGG16
from keras import models,layers,optimizers
from keras.callbacks import TensorBoard

conv_base=VGG16(weights='imagenet',include_top=False,input_shape=(128,128,3))

model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

conv_base.trainable=False

model.summary()

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['acc'])

import morph as mp

model.fit_generator(
      mp.train_flow,
      steps_per_epoch=32,
      epochs=50,
      validation_data=mp.test_flow,
      validation_steps=32,callbacks=[TensorBoard(log_dir='logs/3')])
model.save_weights('outputs/weights_vgg16_use.h5')
