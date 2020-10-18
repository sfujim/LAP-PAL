import argparse
import copy
import importlib
import json
import os

import numpy as np
import torch

import DDQN
import PER_DDQN
import LAP_DDQN
import PAL_DDQN
import utils


def main(env, replay_buffer, is_atari, state_dim, num_actions, args, parameters, device):
	# Initialize and load policy
	kwargs = {
		"is_atari": is_atari,
		"num_actions": num_actions,
		"state_dim": state_dim,
		"device": device,
		"discount": parameters["discount"],
		"optimizer": parameters["optimizer"],
		"optimizer_parameters": parameters["optimizer_parameters"],
		"polyak_target_update": parameters["polyak_target_update"],
		"target_update_frequency": parameters["target_update_freq"],
		"tau": parameters["tau"],
		"initial_eps": parameters["initial_eps"],
		"end_eps": parameters["end_eps"],
		"eps_decay_period": parameters["eps_decay_period"],
		"eval_eps": parameters["eval_eps"]
	}

	if args.algorithm == "DDQN":
		policy = DDQN.DDQN(**kwargs)
	elif args.algorithm == "PER_DDQN":
		policy = PER_DDQN.PER_DDQN(**kwargs)

	kwargs["alpha"] = parameters["alpha"]
	kwargs["min_priority"] = parameters["min_priority"]

	if args.algorithm == "LAP_DDQN":
		policy = LAP_DDQN.LAP_DDQN(**kwargs)
	elif args.algorithm == "PAL_DDQN":
		policy = PAL_DDQN.PAL_DDQN(**kwargs)

	evaluations = []

	state, done = env.reset(), False
	episode_start = True
	episode_reward = 0
	episode_timesteps = 0
	episode_num = 0

	# Interact with the environment for max_timesteps
	for t in range(int(args.max_timesteps)):

		episode_timesteps += 1

		#if args.train_behavioral:
		if t < parameters["start_timesteps"]:
			action = env.action_space.sample()
		else:
			action = policy.select_action(np.array(state))

		# Perform action and log results
		next_state, reward, done, info = env.step(action)
		episode_reward += reward

		# Only consider "done" if episode terminates due to failure condition
		done_float = float(done) if episode_timesteps < env._max_episode_steps else 0

		# For atari, info[0] = clipped reward, info[1] = done_float
		if is_atari:
			reward = info[0]
			done_float = info[1]
			
		# Store data in replay buffer
		replay_buffer.add(state, action, next_state, reward, done_float, done, episode_start)
		state = copy.copy(next_state)
		episode_start = False

		# Train agent after collecting sufficient data
		if t >= parameters["start_timesteps"] and (t + 1) % parameters["train_freq"] == 0:
			policy.train(replay_buffer)

		if done:
			# +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
			print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
			# Reset environment
			state, done = env.reset(), False
			episode_start = True
			episode_reward = 0
			episode_timesteps = 0
			episode_num += 1

		# Evaluate episode
		if (t + 1) % parameters["eval_freq"] == 0:
			evaluations.append(eval_policy(policy, args.env, args.seed))
			np.save(f"./results/{setting}.npy", evaluations)


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, eval_episodes=10):
	eval_env, _, _, _ = utils.make_env(env_name, atari_preprocessing)
	eval_env.seed(seed + 100)

	avg_reward = 0.
	for _ in range(eval_episodes):
		state, done = eval_env.reset(), False
		while not done:
			action = policy.select_action(np.array(state), eval=True)
			state, reward, done, _ = eval_env.step(action)
			avg_reward += reward

	avg_reward /= eval_episodes

	print("---------------------------------------")
	print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
	print("---------------------------------------")
	return avg_reward


if __name__ == "__main__":

	# Atari Specific
	atari_preprocessing = {
		"frame_skip": 4,
		"frame_size": 84,
		"state_history": 4,
		"done_on_life_loss": False,
		"reward_clipping": True,
		"max_episode_timesteps": 27e3
	}

	atari_parameters = {
		# LAP/PAL
		"alpha": 0.6,
		"min_priority": 1e-2,
		# Exploration
		"start_timesteps": 2e4,
		"initial_eps": 1,
		"end_eps": 1e-2,
		"eps_decay_period": 25e4,
		# Evaluation
		"eval_freq": 5e4,
		"eval_eps": 1e-3,
		# Learning
		"discount": 0.99,
		"buffer_size": 1e6,
		"batch_size": 32,
		"optimizer": "RMSprop",
		"optimizer_parameters": {
			"lr": 0.0000625,
			"alpha": 0.95,
			"centered": True,
			"eps": 0.00001
		},
		"train_freq": 4,
		"polyak_target_update": False,
		"target_update_freq": 8e3,
		"tau": 1
	}

	regular_parameters = {
		# LAP/PAL
		"alpha": 0.4,
		"min_priority": 1,
		# Exploration
		"start_timesteps": 1e3,
		"initial_eps": 0.1,
		"end_eps": 0.1,
		"eps_decay_period": 1,
		# Evaluation
		"eval_freq": 5e3,
		"eval_eps": 0,
		# Learning
		"discount": 0.99,
		"buffer_size": 1e6,
		"batch_size": 64,
		"optimizer": "Adam",
		"optimizer_parameters": {
			"lr": 3e-4
		},
		"train_freq": 1,
		"polyak_target_update": True,
		"target_update_freq": 1,
		"tau": 0.005
	}

	# Load parameters
	parser = argparse.ArgumentParser()
	parser.add_argument("--algorithm", default="LAP_DDQN")				# OpenAI gym environment name
	parser.add_argument("--env", default="PongNoFrameskip-v0")		# OpenAI gym environment name #PongNoFrameskip-v0
	parser.add_argument("--seed", default=0, type=int)				# Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--buffer_name", default="Default")			# Prepends name to filename
	parser.add_argument("--max_timesteps", default=50e6, type=int)	# Max time steps to run environment or train for
	args = parser.parse_args()

	print("---------------------------------------")	
	print(f"Setting: Algorithm: {args.algorithm}, Env: {args.env}, Seed: {args.seed}")
	print("---------------------------------------")

	setting = f"{args.algorithm}_{args.env}_{args.seed}"

	if not os.path.exists("./results"):
		os.makedirs("./results")

	# Make env and determine properties
	env, is_atari, state_dim, num_actions = utils.make_env(args.env, atari_preprocessing)
	parameters = atari_parameters if is_atari else regular_parameters

	env.seed(args.seed)
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	# Initialize buffer
	prioritized = True if args.algorithm == "PER_DDQN" or args.algorithm == "LAP_DDQN" else False
	replay_buffer = utils.ReplayBuffer(
		state_dim, 
		prioritized, 
		is_atari, 
		atari_preprocessing, 
		parameters["batch_size"], 
		parameters["buffer_size"], 
		device
	)

	main(env, replay_buffer, is_atari, state_dim, num_actions, args, parameters, device)
