import numpy as np
import torch


class ReplayBuffer(object):
	def __init__(self, state_dim, action_dim, max_size=int(1e6)):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.next_state = np.zeros((max_size, state_dim))
		self.reward = np.zeros((max_size, 1))
		self.not_done = np.zeros((max_size, 1))

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


	def add(self, state, action, next_state, reward, done):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.not_done[self.ptr] = 1. - done

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)


	def sample(self, batch_size):
		ind = np.random.randint(self.size, size=batch_size)

		return (
			torch.FloatTensor(self.state[ind]).to(self.device),
			torch.FloatTensor(self.action[ind]).to(self.device),
			torch.FloatTensor(self.next_state[ind]).to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			torch.FloatTensor(self.not_done[ind]).to(self.device)
		)


class PrioritizedReplayBuffer():
	def __init__(self, state_dim, action_dim, max_size=int(1e6)):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.next_state = np.zeros((max_size, state_dim))
		self.reward = np.zeros((max_size, 1))
		self.not_done = np.zeros((max_size, 1))

		self.tree = SumTree(max_size)
		self.max_priority = 1.0
		self.beta = 0.4

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


	def add(self, state, action, next_state, reward, done):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.not_done[self.ptr] = 1. - done

		self.tree.set(self.ptr, self.max_priority)

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)


	def sample(self, batch_size):
		ind = self.tree.sample(batch_size)

		weights = self.tree.levels[-1][ind] ** -self.beta
		weights /= weights.max()

		self.beta = min(self.beta + 2e-7, 1) # Hardcoded: 0.4 + 2e-7 * 3e6 = 1.0. Only used by PER.

		return (
			torch.FloatTensor(self.state[ind]).to(self.device),
			torch.FloatTensor(self.action[ind]).to(self.device),
			torch.FloatTensor(self.next_state[ind]).to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			torch.FloatTensor(self.not_done[ind]).to(self.device),
			ind,
			torch.FloatTensor(weights).to(self.device).reshape(-1, 1)
		)


	def update_priority(self, ind, priority):
		self.max_priority = max(priority.max(), self.max_priority)
		self.tree.batch_set(ind, priority)


class SumTree(object):
	def __init__(self, max_size):
		self.levels = [np.zeros(1)]
		# Tree construction
		# Double the number of nodes at each level
		level_size = 1
		while level_size < max_size:
			level_size *= 2
			self.levels.append(np.zeros(level_size))
			

	# Batch binary search through sum tree
	# Sample a priority between 0 and the max priority
	# and then search the tree for the corresponding index
	def sample(self, batch_size):
		value = np.random.uniform(0, self.levels[0][0], size=batch_size)
		ind = np.zeros(batch_size, dtype=int)
		
		for nodes in self.levels[1:]:
			ind *= 2
			left_sum = nodes[ind]
			
			is_greater = np.greater(value, left_sum)
			# If value > left_sum -> go right (+1), else go left (+0)
			ind += is_greater
			# If we go right, we only need to consider the values in the right tree
			# so we subtract the sum of values in the left tree
			value -= left_sum * is_greater
		
		return ind


	def set(self, ind, new_priority):
		priority_diff = new_priority - self.levels[-1][ind]

		for nodes in self.levels[::-1]:
			np.add.at(nodes, ind, priority_diff)
			ind //= 2


	def batch_set(self, ind, new_priority):
		# Confirm we don't increment a node twice
		ind, unique_ind = np.unique(ind, return_index=True)
		priority_diff = new_priority[unique_ind] - self.levels[-1][ind]
		
		for nodes in self.levels[::-1]:
			np.add.at(nodes, ind, priority_diff)
			ind //= 2