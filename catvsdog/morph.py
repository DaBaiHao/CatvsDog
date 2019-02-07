from keras.preprocessing.image import ImageDataGenerator

train_dir='min_trainfordata/train'

test_dir='min_trainfordata/test'

train_pic_gen=ImageDataGenerator(rescale=1./255,rotation_range=20,width_shift_range=0.2,height_shift_range=0.2,
                                 shear_range=0.2,zoom_range=0.5,horizontal_flip=True,fill_mode='nearest')

test_pic_gen=ImageDataGenerator(rescale=1./255)


train_flow=train_pic_gen.flow_from_directory(train_dir,(128,128),batch_size=32,class_mode='binary')

test_flow=test_pic_gen.flow_from_directory(test_dir,(128,128),batch_size=32,class_mode='binary')
