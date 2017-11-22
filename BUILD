load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")

tf_custom_op_library(
    name = "nn_graph_persistence.so",
    srcs = ["nn_graph_persistence.cc"],
    deps = ["//tensorflow/core/user_ops/include/dionysus:dionysus",
            "//tensorflow/core/user_ops/wasserstein:wasserstein"],
)

tf_custom_op_library(
    name = "nn_graph_bottleneck.so",
    srcs = ["nn_graph_bottleneck.cc"],
    deps = ["//tensorflow/core/user_ops/include/dionysus:dionysus",
            "//tensorflow/core/user_ops/bottleneck:bottleneck"],
)

tf_custom_op_library(
    name = "nn_train_persistence.so",
    srcs = ["nn_train_persistence.cc"],
    deps = ["//tensorflow/core/user_ops/include/dionysus:dionysus",
            "//tensorflow/core/user_ops/wasserstein:wasserstein"],
)
