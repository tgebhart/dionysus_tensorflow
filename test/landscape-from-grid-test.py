import os
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python import debug as tf_debug

landscape_module = tf.load_op_library('landscape_from_grid.so')


@ops.RegisterGradient("LandscapeFromGrid")
def _zero_out_grad(op, grad):
  ret = landscape_module.landscape_from_grid_gradient(op.inputs[0], op.inputs[1], op.inputs[2], op.inputs[3], grad)
  return [ret, None, None, None]


sess = tf.Session('')
sess = tf_debug.LocalCLIDebugWrapperSession(sess)

# vertices = [[0,0,0,2,1,3,1,1,1],
#             [0,3,0,2,1,3,1,3,1],
#             [0,0,2,3,3,3,1,1,1],
#             [3,3,3,0,0,0,3,3,3],
#             [2,1,3,0,3,0,2,2,2],
#             [2,1,3,0,0,0,2,3,2],
#             [1,1,3,3,3,3,2,2,2],
#             [1,2,2,1,2,1,2,3,2],
#             [1,1,1,1,1,1,2,2,2]]

vertices = [[0,0,0,3,3,3,0,0,0],
            [0,1,0,3,3,3,0,2,0],
            [0,0,0,3,3,3,0,0,0],
            [3,3,3,3,3,3,3,3,3],
            [3,3,3,3,3,3,3,3,3],
            [3,3,3,3,3,3,3,3,3],
            [3,3,3,3,3,3,0,0,0],
            [3,3,3,3,3,3,0,2,0],
            [3,3,3,3,3,3,0,0,0]]


dimension = [1];
depth = [5]
sample_points = [0, 0.5, 1, 1.5, 2, 2.5, 3]

ones = tf.ones([depth[0], len(sample_points)], tf.float32)

result = landscape_module.landscape_from_grid(vertices, dimension, depth, sample_points)
gradient = landscape_module.landscape_from_grid_gradient(vertices, dimension, depth, sample_points, ones)

ret = result.eval(session=sess);
g_ret = gradient.eval(session=sess);

print(ret)
print(g_ret)
