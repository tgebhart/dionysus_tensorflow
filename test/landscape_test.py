import os
import tensorflow as tf

class LandscapeTest(tf.test.TestCase):
  def testLandscape(self):
    #zero_out_module = tf.load_op_library(os.path.join(tf.resource_loader.get_data_files_path(), 'zero_out.so'))
    landscape_module = tf.load_op_library('/home/tgebhart/Projects/tensorflow/bazel-bin/tensorflow/core/user_ops/landscape_test.so')
    with self.test_session():
        result = landscape_module.landscape_test([5, 4, 3, 2, 1])
        self.assertAllEqual(result.eval(), [5, 0, 0, 0, 0])

if __name__ == "__main__":
  tf.test.main()
