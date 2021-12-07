# An Equivalence between Loss Functions and Non-Uniform Sampling in Experience Replay

PyTorch implementation of Loss-Adjusted Prioritized (LAP) experience replay and Prioritized Approximation Loss (PAL). LAP is an improvement to prioritized experience replay which eliminates the importance sampling weights in a principled manner, by considering the relationship to the loss function. PAL is a uniformly sampled loss function with the same expected gradient as LAP. 

The [paper](https://arxiv.org/abs/2007.06049) will be presented at NeurIPS 2020. Code is provided for both continuous (with TD3) and discrete (with DDQN) domains.

### Bibtex

```
@article{fujimoto2020equivalence,
  title={An Equivalence between Loss Functions and Non-Uniform Sampling in Experience Replay},
  author={Fujimoto, Scott and Meger, David and Precup, Doina},
  journal={Advances in Neural Information Processing Systems},
  volume={33},
  year={2020}
}
```
