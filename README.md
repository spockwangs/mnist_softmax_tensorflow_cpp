# Introduction

Train mnist softmax regression model using TensorFlow C++ API.

# Requirements

- TensorFlow C++ API 1.5
- Python 2.x
- [MNIST data, 4 files](http://yann.lecun.com/exdb/mnist/)

# How to use

Export the graph:

```
$ create_mnist_softmax_model.py --export_dir models
```

Compile the training code:
```
$ g++ train_mnist_softmax_model.cpp -o train_mnist_softmax_model -ltensorflow_cc -ltensorflow_framework -std=c++11 -ldl
```

Download the MNIST data to a directory and unzip.

Run the training:
```
$ train_mnist_softmax_model <data_dir> <graph file>
```

