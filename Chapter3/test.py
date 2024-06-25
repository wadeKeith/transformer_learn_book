import numpy as np
import pandas as pd
import tensorflow as tf
import torch

# print(f"TensorFlow has access to the following devices:\n{tf.config.list_physical_devices()}")

# # See TensorFlow version
# print(f"TensorFlow version: {tf.__version__}")

# cifar = tf.keras.datasets.cifar100
# (x_train, y_train), (x_test, y_test) = cifar.load_data()
# model = tf.keras.applications.ResNet50(
#     include_top=True,
#     weights=None,
#     input_shape=(32, 32, 3),
#     classes=100,)

# loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
# model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])
# model.fit(x_train, y_train, epochs=5, batch_size=64) 
import time

# 创建一个随机的大型矩阵
matrix_size = 10000
matrix_a = tf.random.normal((matrix_size, matrix_size))
matrix_b = tf.random.normal((matrix_size, matrix_size))

# 使用CPU执行矩阵乘法
with tf.device('/CPU:0'):
    start_time = time.time()
    result_cpu = tf.matmul(matrix_a, matrix_b)
    end_time = time.time()
    print("使用CPU执行矩阵乘法所需时间：", end_time - start_time, "秒")

# 使用GPU执行矩阵乘法
with tf.device('/GPU:0'):
    start_time = time.time()
    result_gpu = tf.matmul(matrix_a, matrix_b)
    end_time = time.time()
    print("使用GPU执行矩阵乘法所需时间：", end_time - start_time, "秒")

# mps_available = torch.backends.mps.is_available()
# print("MPS available:", mps_available)
# device = torch.device("mps" if mps_available else "cpu")
# print("Using device:", device)

# if mps_available:
#     tensor_gpu = torch.tensor([1.0, 2.0, 3.0], device=device)
#     print("GPU tensor created:", tensor_gpu)


