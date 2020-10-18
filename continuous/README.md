# LAP/PAL with TD3 for Continuous Control

Code for Loss-Adjusted Prioritized (LAP) experience replay and Prioritized Approximation Loss (PAL) with TD3. 

Paper results were collected with [MuJoCo 2.0.2.9](http://www.mujoco.org/) on [OpenAI gym](https://github.com/openai/gym). Networks are trained using [PyTorch 1.2.0](https://github.com/pytorch/pytorch) and Python 3.7. 

Example command:
```
python main.py --policy "LAP_TD3" --env "HalfCheetah-v3"
```

Hyper-parameters can be modified with different arguments to main.py. OpenAI gym now defaults to MuJoCo 1.50. We found the performance of Humanoid-v3 is lower on this version, although relative order between algorithms is unchanged. 
