## Dionysus Tensorflow

Dionysus functionality adapted as user ops for inclusion in Tensorflow.


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
