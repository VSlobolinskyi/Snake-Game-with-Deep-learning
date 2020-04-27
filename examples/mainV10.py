import tensorflow as tf
import time
import numpy as np
from random import randrange
tf.keras.backend.set_floatx('float64')
# device_name = tf.test.gpu_device_name()
# if device_name != '/device:GPU:0':
#   print(
#       '\n\nThis error most likely means that this notebook is not '
#       'configured to use a GPU.  Change this in Notebook Settings via the '
#       'command palette (cmd/ctrl-shift-P) or the Edit menu.\n\n')
#   raise SystemError('GPU device not found')

# def cpu():
#   with tf.device('/cpu:0'):
#     random_image_cpu = tf.random.normal((100, 100, 100, 3))
#     net_cpu = tf.keras.layers.Conv2D(32, 7)(random_image_cpu)
#     return tf.math.reduce_sum(net_cpu)

# def gpu():
#   with tf.device('/device:GPU:0'):
#     random_image_gpu = tf.random.normal((100, 100, 100, 3))
#     net_gpu = tf.keras.layers.Conv2D(32, 7)(random_image_gpu)
#     return tf.math.reduce_sum(net_gpu)
  
# # We run each op once to warm up; see: https://stackoverflow.com/a/45067900
# cpu()
# gpu() 

# # Run the op several times.
# print('Time (s) to convolve 32x7x7x3 filter over random 100x100x100x3 images '
#       '(batch x height x width x channel). Sum of ten runs.')
# print('CPU (s):')
# cpu_time = timeit.timeit('cpu()', number=10, setup="from __main__ import cpu")
# print(cpu_time)
# print('GPU (s):')
# gpu_time = timeit.timeit('gpu()', number=10, setup="from __main__ import gpu")
# print(gpu_time)
# print('GPU speedup over CPU: {}x'.format(int(cpu_time/gpu_time)))

print(tf.__version__)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=4, activation='relu', input_shape=(28, 28, 3)))
model.add(tf.keras.layers.MaxPooling2D(pool_size=4))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(64, activation="relu"))
model.add(tf.keras.layers.Dense(2, activation='softmax'))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
x_train = []
y_train = []
start_img = time.time()
size = 6000
x_train.append(tf.random.normal((size, 28, 28, 3)))
for i in range(size):
  if randrange(0, 2) == 0:
    y_train.append(0)
  else:
    y_train.append(1)
end_img = time.time()
print("IMAGE GENERATION TIME:",end_img-start_img)
start = time.time()
model.fit(x_train, y_train, steps_per_epoch=10, verbose=0)
end = time.time()
print("MODEL FIT TIME:{:.2f}".format(end-start))
start = time.time()
model.predict(x_train, steps=1)
end = time.time()
print("MODEL PREDICT TIME:{:.4f}".format(end-start))
start = time.time()
model.predict(tf.random.normal((1, 28, 28, 3)), steps=1)
end = time.time()
print("MODEL PREDICT ONCE TIME:{:.4f}".format(end-start))