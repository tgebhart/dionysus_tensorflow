import os
import tensorflow as tf
from tensorflow.python import debug as tf_debug

landscape_module = tf.load_op_library('landscape_from_grid.so')

sess = tf.Session('')
sess = tf_debug.LocalCLIDebugWrapperSession(sess)

vertices = [[2,2,2,2,2,2,2,2,2],
            [2,2,2,2,2,2,2,2,2],
            [2,2,2,2,2,2,2,2,2],
            [2,2,2,0,0,0,1,1,2],
            [2,2,2,0,1,0,2,1,2],
            [2,2,2,0,0,0,1,1,2],
            [1,1,1,1,2,1,2,2,2],
            [1,2,2,1,2,1,2,2,2],
            [1,1,1,1,1,1,2,2,2]]

dimension = [1];
depth = [5]
sample_points = [0, 0.5, 1, 1.5, 2]

result = landscape_module.landscape_from_grid(vertices, dimension, depth, sample_points)
ret = result.eval(session=sess);

print(ret)
