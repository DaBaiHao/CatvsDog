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


# How to get better performance
 - use more complex model
 - data argumentation (调整图片对比度之类)
 - split data into **train** and **validation** and evaluate the validation dataset
   - generate **batch** from **validation** and feed in
