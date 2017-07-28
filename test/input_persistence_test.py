import os
import tensorflow as tf

class InputPersistenceTest(tf.test.TestCase):
  def testPersistence(self):
    #zero_out_module = tf.load_op_library(os.path.join(tf.resource_loader.get_data_files_path(), 'zero_out.so'))
    persistence_module = tf.load_op_library('/home/tgebhart/Projects/tensorflow/bazel-bin/tensorflow/core/user_ops/nn_graph_persistence.so')
    with self.test_session():
        result = persistence_module.input_graph_persistence([[5, 5, 1, 32]], [0, 1])
        # self.assertAllEqual(result.eval(), [5, 0, 0, 0, 0])
        result.eval()

if __name__ == "__main__":
  tf.test.main()
