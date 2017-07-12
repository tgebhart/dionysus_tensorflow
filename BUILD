load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")

tf_custom_op_library(
    name = "zero_out.so",
    srcs = ["zero_out.cc"],
)

tf_custom_op_library(
    name = "landscape_test.so",
    srcs = ["landscape_test.cc"],
    deps = ["//tensorflow/core/user_ops/include/topology:topology",
            "//tensorflow/core/user_ops/include/utilities:utilities"],
)


tf_custom_op_library(
    name = "filtration_grid_test.so",
    srcs = ["filtration_grid_test.cc"],
    deps = ["//tensorflow/core/user_ops/include/topology:topology",
            "//tensorflow/core/user_ops/include/utilities:utilities"],
)

tf_custom_op_library(
    name = "landscape_from_grid.so",
    srcs = ["landscape-from-grid.cc"],
    deps = ["//tensorflow/core/user_ops/include/topology:topology",
            "//tensorflow/core/user_ops/include/utilities:utilities"],
)
