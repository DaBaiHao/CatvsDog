import tensorflow as tf

def inference(images, batch_size, n_classes):
    # conv1
    with tf.variable_scope("conv1") as scope:
        weights = tf.get_variable("weights",
                                 shape=[3,3,3,16],
                                 dtype=tf.float32,
                                 initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        biases = tf.get_variable("biases",
                                 shape=[16],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer)
        conv = tf.nn.conv2d(images, weights, strides=[1, 1, 1, 1],
                            padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name="conv1")

    # pool1 && norm1
    with tf.variable_scope("pooling1_Irn") as scope:
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1],
                               strides=[1, 2, 2, 1],padding="SAME",
                               name="pooling1")
        norm1 = tf.nn.lrn(pool1, depth_radius=4,
                          bias= 1.0, alpha=0.001/9.0,
                          beta=0.75, name='norm1')

    # conv2
