# Cats vs. Dogs
This project is for the Kaggle Cats vs. Dogs based on tenorflow. The dataset can be donwnload from [Cats vs. Dogs](https://www.kaggle.com/c/dogs-vs-cats/data)
### method
#### method `get_files(file_dir)`  :
This method is to get all of the training data from given `file_dir`, and return as a list.

Because half of the training dataset is cats and another half is dogs, so the order of data is being Disrupted.

Using `np.hstack()` to combine all of the cat and dog images.

Batch:
 - 5000 images
 - Batch size = 10
 - iteration = 5000/10 = 500 iterations to train (one epoch)

----

# How to get better performance
 - use more complex model
 - data argumentation (调整图片对比度之类)
 - split data into **train** and **validation** and evaluate the validation dataset
   - generate **batch** from **validation** and feed in



# 池化层
在卷积神经网络中，卷积层之间往往会加上一个池化层。池化层可以非常有效地缩小参数矩阵的尺寸，从而减少最后全连层中的参数数量。使用池化层即可以加快计算速度也有防止过拟合的作用。


``` python
#input表示上一层的输出上例图片中大左边大的矩阵，ksize定义了池化层过滤器的尺寸，strides定义的步长信息第一维和第四维都为1，padding定义了是否用全0填充SAME表示全0填充VALID表示不填充
pool = tf.nn.max_pool(input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

```
