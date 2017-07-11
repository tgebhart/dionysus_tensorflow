import os
import tensorflow as tf

class FiltrationGridTest(tf.test.TestCase):
  def testFiltrationGrid(self):
    filtration_grid_module = tf.load_op_library('filtration_grid_test.so')
    with self.test_session():
        result = filtration_grid_module.filtration_grid_test([5, 4, 3, 2, 1])
        #self.assertAllEqual(result.eval(), [5, 0, 0, 0, 0])

if __name__ == "__main__":
  tf.test.main()
