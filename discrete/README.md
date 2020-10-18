# LAP/PAL with DDQN for Discrete Action Domains

Code for Loss-Adjusted Prioritized (LAP) experience replay and Prioritized Approximation Loss (PAL) with Double DQN.

Paper results were collected with [OpenAI gym](https://github.com/openai/gym). Networks are trained using [PyTorch 1.2.0](https://github.com/pytorch/pytorch) and Python 3.7. 

Example command:
```
python main.py --policy "LAP_DDQN" --env "PongNoFrameskip-v0"
```

Hyper-parameters can be modified with different arguments to main.py and the parameter dicts in main.py. Code is set up to potentially run non-Atari environments, but the performance is mostly untested.
