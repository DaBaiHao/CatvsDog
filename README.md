# Cats vs. Dogs
This project is for the Kaggle Cats vs. Dogs based on tenorflow. The dataset can be donwnload from [Cats vs. Dogs](https://www.kaggle.com/c/dogs-vs-cats/data)

###  First time experiment report

## Aim of this experiment:
 - Learn tensorflow
 - Learn CNN

## About the code:
The code is fully followed [Guoqing Xu tutorial](https://www.youtube.com/watch?v=8EXMxQwuCrs&index=17&list=PLnUknG7KBFzqMSDZC1mnYMN0zMoRaH68r).

## Changes
#### 1. `tf.image.resize_image_with_crop_or_pad()`
- **Location:** pre-process stage, the tutorial using the `tf.image.resize_image_with_crop_or_pad()` method to resize the trainning images.
- **Problems:** This method is to resize the image by cut or add the black bar to resize the given images from center. Might influence the accurcy of the trainning performance. For example the if one image is very large, the animals might cut only the body, from an animals body is hard to predict what animals is.
- **Solution:** change the resize method to `tf.image.resize_images(image, [image_H, image_W], method=tf.image.ResizeMethod.BICUBIC)`.
- **Benifit:** This method resize the images by bicubic interpolation and just Stretch the images

## Quentions:
1. bicubic interpolation
2. shape=[3,3,3,16], the [3,3,**3**,16] is what
``` python
weights = tf.get_variable('weights',
                         shape=[3,3,3,16],
                         dtype=tf.float32,
                         initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
```
3. poolling layer

### Result
1. The whole result shows [here](https://github.com/DaBaiHao/CatvsDog/blob/master/train/first_train.txt).
2. tabel:
### Limation:
The error also makes, for example the predicte a dog image, it also might classified to cats.s

----
Learining Point
#### How to get better performance
 - use more complex model
 - data argumentation (调整图片对比度之类)
 - split data into **train** and **validation** and evaluate the validation dataset
   - generate **batch** from **validation** and feed in



#### 池化层
在卷积神经网络中，卷积层之间往往会加上一个池化层。池化层可以非常有效地缩小参数矩阵的尺寸，从而减少最后全连层中的参数数量。使用池化层即可以加快计算速度也有防止过拟合的作用。


``` python
#input表示上一层的输出上例图片中大左边大的矩阵，ksize定义了池化层过滤器的尺寸，strides定义的步长信息第一维和第四维都为1，padding定义了是否用全0填充SAME表示全0填充VALID表示不填充
pool = tf.nn.max_pool(input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

```
