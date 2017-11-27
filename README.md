## Dionysus Tensorflow

Persistent Homology from Dionysus adapted as user ops for inclusion in Tensorflow.

### Building

Copy the `include` folder into `/tensorflow/core/user_ops` folder of your
tensorflow directory:

```
$ cp include ~/<path-to>/tensorflow/tensorflow/core/user_ops
```

As well, copy the BUILD file located in the root directory to the same location:

```
$ cp BUILD ~/<path-to>/tensorflow/tensorflow/core/user_ops
```

Also copy the `wasserstein` folder into the same location:

```
$ cp wasserstein ~/<path-to>/tensorflow/tensorflow/core/user_ops
```

Finally, copy the `bottleneck` folder into the same location:

```
$ cp bottleneck ~/<path-to>/tensorflow/tensorflow/core/user_ops
```

From the root of your tensorflow directory, build the new operation using Bazel:

```
$ bazel build -c opt //tensorflow/core/user_ops:<name-of-op-file>.so
```

Upon successful build, you can use the operation in tensorflow in python.

#### Helpful Links
  - https://www.tensorflow.org/versions/r0.12/how_tos/adding_an_op/
  - https://groups.google.com/forum/#!topic/bazel-discuss/64v_Oxnav3I

### To use

Import in your python program the user_ops module as described in the above links:

```
import tensorflow as tf

persistence_module = tf.load_op_library('<path-to-tensorflow>/tensorflow/bazel-bin/tensorflow/core/user_ops/nn_graph_persistence.so')
```

#### Percentile Calculation

```
net, keep_prob = model.fit(x)
p = 99
percentiles = persistence_module.layerwise_percentile([net['input'],
                                                        net['W_conv1'],
                                                        net['h_conv1'],
                                                        net['h_conv1'],
                                                        net['W_fc1'],
                                                        net['h_fc1'],
                                                        net['h_fc1_drop'],
                                                        net['W_fc2'],
                                                        net['y_conv']],
                                                        [0, 1, 2, 2, 1, 4, 4, 1, 4],
                                                        [p,p,p])

ps = percentiles.eval(feed_dict={x: test_inputs[0:1], keep_prob:1.0})
```


#### Persistence Diagram

```
import matplotlib.pyplot as plt
import numpy as np

diagram_filename = 'persistence.csv'

result = persistence_module.input_graph_persistence([net['input'],
                                                    net['W_conv1'],
                                                    net['h_conv1'],
                                                    net['h_conv1'],
                                                    net['W_fc1'],
                                                    net['h_fc1'],
                                                    net['h_fc1_drop'],
                                                    net['W_fc2'],
                                                    net['y_conv']],
                                                    [0, 1, 2, 2, 1, 4, 4, 1, 4],
                                                    np.stack((ps, ps)),
                                                    0,
                                                    diagram_filename
                                                    )
r = result.eval(feed_dict={x: test_inputs[1:], keep_prob:1.0})

diag = np.genfromtxt(diagram_filename, delimiter=',')

ax = plt.subplot()

ax.scatter(diag[:,0], diag[:,1], s=25, c=(diag[:,0] - diag[:,1])**2, cmap=plt.cm.coolwarm, zorder=10)
lims = [
    np.min([0]),  # min of both axes
    np.max(diag[:,0]),  # max of both axes
]

ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
ax.set_aspect('equal')
ax.set_xlim(lims)
ax.set_ylim(lims)
plt.xlabel('Birth Time')
plt.ylabel('Death Time')

plt.show()
```


#### Wasserstein Distance

```
result = persistence_module.wasserstein_distance([net['input'],
                                                  net['W_conv1'],
                                                  net['h_conv1'],
                                                  net['h_conv1'],
                                                  net['W_fc1'],
                                                  net['h_fc1'],
                                                  net['h_fc1_drop'],
                                                  net['W_fc2'],
                                                  net['y_conv']],
                                                  [0, 1, 2, 2, 1, 4, 4, 1, 4],
                                                  np.stack((ps, ps)),
                                                  0
                                                  )
persistence_distance = result.eval(feed_dict={x: test_inputs, keep_prob:1.0})
```
