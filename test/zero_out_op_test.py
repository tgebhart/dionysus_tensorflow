import os
import tensorflow as tf

class ZeroOutTest(tf.test.TestCase):
  def testZeroOut(self):
    #zero_out_module = tf.load_op_library(os.path.join(tf.resource_loader.get_data_files_path(), 'zero_out.so'))
    zero_out_module = tf.load_op_library('/home/tgebhart/Projects/tensorflow/bazel-bin/tensorflow/core/user_ops/zero_out.so')
    with self.test_session():
      result = zero_out_module.zero_out([5, 4, 3, 2, 1])
      self.assertAllEqual(result.eval(), [5, 0, 0, 0, 0])

if __name__ == "__main__":
  tf.test.main()
