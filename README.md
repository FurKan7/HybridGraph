<h1 align="center">Image Matching Using Hybrid Graph Neural Networks</h1>


PyTorch implementation of Image Keypoint Matching for Graph Neural Networks. Code based on [DGMC](https://github.com/rusty1s/deep-graph-matching-consensus). To run the aglorithm presented in the paper use: 

```
$ cd examples/
$ python pascal.py
$ python willow.py
$ python pascal_pf.py
```

The unmodified scripts use the basline DGMC algorithm. 

## Requirements

* **[PyTorch](https://pytorch.org/get-started/locally/)** (>=1.2.0)
* **[PyTorch Geometric](https://github.com/rusty1s/pytorch_geometric)** (>=1.5.0)
* **[KeOps](https://github.com/getkeops/keops)** (>=1.1.0)

## Installation
```
$ python setup.py install