import tensorflow as tf
from tensorflow.python.client import device_lib

print("TensorFlow version:", tf.__version__)
print("Built with CUDA:", tf.test.is_built_with_cuda())
print("GPU Available (via tf):", tf.config.list_physical_devices('GPU'))

print("\nDevices listed by device_lib:")
print(device_lib.list_local_devices())

tf.debugging.set_log_device_placement(True)

# Trigger any GPU-op to force detection
a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
c = tf.matmul(a, b)
print(c)

