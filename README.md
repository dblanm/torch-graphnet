# PyTorch Graph Networks

[PyTorch](https://pytorch.org/) implementation of DeepMind [Graph Nets](https://github.com/deepmind/graph_nets).
The original code depends on [Tensorflow](https://www.tensorflow.org/) and
[Sonnet](https://sonnet.readthedocs.io/en/latest/index.html).

This implementation is based on [PyTorch Geometric](https://github.com/rusty1s/pytorch_geometric)
which is a geometric deep learning extension library for [PyTorch](https://pytorch.org/)

## Graph Networks
Graph Networks are a general framework that generalizes graph neural networks.
It unifies [Message Passing Neural Networks](https://arxiv.org/pdf/1704.01212v2.pdf) (MPNNs) 
and [Non-Local Neural Networks](https://arxiv.org/pdf/1711.07971v3.pdf) (NLNNs),
as well as other variants like [Interaction Networks](https://arxiv.org/abs/1612.00222) (INs) 
or[Relation Networks](https://arxiv.org/pdf/1702.05068.pdf) (RNs).

You can have a look at Graph Networks in their arXiV paper:
[Battaglia, Peter W., et al. "Relational inductive biases, deep learning, and graph networks." arXiv preprint arXiv:1806.01261 (2018)](https://arxiv.org/pdf/1806.01261.pdf)

## Available Models
The following models are available:
- Interaction Network
- Graph Independent

You can also build your own models using the Blocks:
- Node Model
- Edge Model

## Requirements
PyTorch 1.8.0 and PyTorch Geometric.

## Example
We provide an example that tests the output against DeepMind's graph_nets. 




