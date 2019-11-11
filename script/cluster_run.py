#-*- coding=utf-8 -*-
from sklearn.datasets.samples_generator import make_blobs
import numpy as np
import tensorflow as tf
from utils import kmeans

cn = 10
X, y_true = make_blobs(n_samples=150, centers=4,
                       cluster_std=0.5, random_state=10)
kmeans_ops = kmeans(cluster_num=cn)
mask = np.concatenate([np.ones((100,2)), np.zeros((50, 2))], axis=0)
X = X * mask
points = tf.constant(X, dtype=tf.float32)
centers = kmeans_ops(tf.reshape(points, [1] + points.shape.as_list()))
gpu_options = tf.GPUOptions(allow_growth=True)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    predict = sess.run(centers)
print(predict)
y = y_true.reshape((-1, 1))
for i in range(4):
    print("center i %d: " %i)
    label = np.sum(X * (y == i), axis=0) / np.sum((y==i), axis=0)
    print(label)
save = np.concatenate([X, predict[0]])
np.save('result.npy', save)