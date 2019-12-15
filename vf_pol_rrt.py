import numpy as np
import cv2
import tqdm

class Tree():
	def __init__(self, start_node):
		self.root = start_node
		self.nodes = [start_node]

	def get_leaves(self):
		return list(filter(lambda node: not node.children, self.nodes))

class Node():
	def __init__(self, state, action, parent=None):
		self.parent = parent
		if parent is None:
			self.tree = Tree(self)
			self.depth = 0
		else:
			self.parent.children.append(self)
			self.tree = parent.tree
			self.tree.nodes.append(self)
			self.depth = self.parent.depth + 1
		self.children = []

		self.f_connection = None
		self.b_connection = None

		self.dist_to_start = None
		self.dist_to_goal = None

		self.state = state
		self.action = action

	def get_descendants_list(self):
		l = [self] + self.children
		for child in self.children:
			l.extend(child.get_descendants_list())
		return l

class VfPolRrt:
	def __init__(self,
		env,
		start_state, goal_state,
		f_policy, f_dynamics,
		b_policy, b_dynamics,
		value_fn,

		n_target_samples=1000,
		target_temperature=1.0,

		value_fn_uncertainty=0.05,

		policy_temperature=2.0,
	):
		self.env = env

		self.f_policy = f_policy
		self.b_policy = b_policy
		self.f_dynamics = f_dynamics
		self.b_dynamics = b_dynamics
		self.value_fn = value_fn

		self.n_target_samples = n_target_samples
		self.target_temperature = target_temperature

		self.value_fn_uncertainty = value_fn_uncertainty

		self.policy_temperature = policy_temperature

		self.start_node = Node(start_state, None, None)
		self.goal_node = Node(goal_state, None, None)
		self.start_node.f_connection = None
		self.start_node.dist_to_start = 0.
		self.start_node.b_connection = self.goal_node
		self.start_node.dist_to_goal = -(1 + self.value_fn_uncertainty) * \
			value_fn(np.array([start_state]), np.array([goal_state]))[0]
		self.goal_node.f_connection = self.start_node
		self.goal_node.dist_to_start = self.start_node.dist_to_goal
		self.goal_node.b_connection = None
		self.goal_node.dist_to_goal = 0.

		self.f_rrts = [self.start_node.tree]
		self.b_rrts = [self.goal_node.tree]

	def clear(self):
		self.f_rrts = self.f_rrts[:1]
		self.b_rrts = self.b_rrts[:1]
		self.f_rrts[0].nodes = self.f_rrts[0].nodes[:1]
		self.b_rrts[0].nodes = self.b_rrts[0].nodes[:1]
		self.f_rrts[0].root.children = []
		self.b_rrts[0].root.children = []

	def build_rrt(self, n_samples=1, display=False):
		for sample_n in range(n_samples):
			target_state, f_connection, b_connection = self.get_target_state(display and (sample_n==0))
			print("target_state:", target_state)
			print("f_connection:", f_connection.state)
			print("b_connection:", b_connection.state)

			f_target = Node(target_state, None, None)
			f_target.f_connection = f_connection
			f_target.dist_to_start = f_connection.dist_to_start + \
				-(1 + self.value_fn_uncertainty) * \
					value_fn(np.array([f_connection.state]), np.array([target_state]))[0]
			f_target.b_connection = b_connection
			f_target.dist_to_goal = b_connection.dist_to_goal + \
				-(1 + self.value_fn_uncertainty) * \
					value_fn(np.array([target_state]), np.array([b_connection.state]))[0]

			b_target = Node(target_state, None, None)
			b_target.f_connection = f_connection
			b_target.dist_to_start = f_target.dist_to_start
			b_target.b_connection = b_connection
			b_target.dist_to_goal = f_target.dist_to_goal

			self.b_rrts.append(f_target.tree)
			self.f_rrts.append(b_target.tree)

			f_dist = -1 * self.value_fn(np.array([f_connection.state]), np.array([target_state]))[0]
			b_dist = -1 * self.value_fn(np.array([target_state]), np.array([b_connection.state]))[0]
			print("f_dist", f_dist)
			print("b_dist", b_dist)

			f_rrt_node = f_connection
			target_node = f_target
			for t in range(max(1, int(np.ceil(f_dist/2)))):
				f_action = self.f_policy(np.array([f_rrt_node.state]), np.array([target_state]))[0]
				f_rrt_next_state = self.f_dynamics(np.array([f_rrt_node.state]), np.array([f_action]))[0]
				f_rrt_node = Node(f_rrt_next_state, f_action, f_rrt_node)
				f_rrt_node.f_connection = f_rrt_node.parent
				f_rrt_node.dist_to_start = f_rrt_node.parent.dist_to_start + 1
				f_rrt_node.b_connection = f_target
				f_rrt_node.dist_to_goal = f_target.dist_to_goal + \
					-(1 + self.value_fn_uncertainty) * \
						value_fn(np.array([f_rrt_node.state]), np.array([target_state]))[0]

				target_action = self.b_policy(np.array([f_connection.state]), np.array([target_node.state]))[0]
				target_rrt_next_state = self.b_dynamics(np.array([target_node.state]), np.array([target_action]))[0]
				target_node = Node(target_rrt_next_state, target_action, target_node)
				target_node.f_connection = f_connection
				target_node.dist_to_start = f_connection.dist_to_start + \
					-(1 + self.value_fn_uncertainty) * \
						value_fn(np.array([f_connection.state]), np.array([target_node.state]))[0]
				target_node.b_connection = target_node.parent
				target_node.dist_to_goal = target_node.parent.dist_to_goal + 1

			b_rrt_node = b_connection
			target_node = b_target
			for t in range(max(1, int(np.ceil(b_dist/2)))):
				b_action = self.b_policy(np.array([target_state]), np.array([b_rrt_node.state]))[0]
				b_rrt_next_state = self.b_dynamics(np.array([b_rrt_node.state]), np.array([b_action]))[0]
				b_rrt_node = Node(b_rrt_next_state, b_action, b_rrt_node)
				b_rrt_node.f_connection = b_target
				b_rrt_node.dist_to_start = b_target.dist_to_start + \
					-(1 + self.value_fn_uncertainty) * \
						value_fn(np.array([target_state]), np.array([b_rrt_node.state]))[0]
				b_rrt_node.b_connection = b_rrt_node.parent
				b_rrt_node.dist_to_goal = b_rrt_node.parent.dist_to_goal + 1

				target_action = self.f_policy(np.array([target_node.state]), np.array([b_connection.state]))[0]
				target_rrt_next_state = self.f_dynamics(np.array([target_node.state]), np.array([target_action]))[0]
				target_node = Node(target_rrt_next_state, target_action, target_node)
				target_node.f_connection = target_node.parent
				target_node.dist_to_start = target_node.parent.dist_to_start + 1
				target_node.b_connection = b_connection
				target_node.dist_to_goal = b_connection.dist_to_goal + \
					-(1 + self.value_fn_uncertainty) * \
						value_fn(np.array([target_node.state]), np.array([b_connection.state]))[0]

			self.get_node_distances()
			if display and (sample_n == n_samples - 1):
				self.plot_rrt()

		if display:
			cv2.waitKey(0)
			cv2.destroyWindow("rrt_plot")

	def get_node_distances(self):
		f_rrt_nodes, b_rrt_nodes = self.get_all_nodes()
		f_states = [node.state for node in f_rrt_nodes]
		b_states = [node.state for node in b_rrt_nodes]

		# get dist_to_goals
		all_nodes = list(sorted(f_rrt_nodes + b_rrt_nodes, key=lambda node: node.dist_to_goal))
		goal_node_idx = all_nodes.index(self.goal_node)
		all_nodes.pop(goal_node_idx)
		all_nodes.insert(0, self.goal_node)
		for (idx, node) in enumerate(all_nodes):
			if node is self.goal_node:
				continue
			proposed_connections = all_nodes[:idx]
			proposed_connection_states = np.array([node.state for node in proposed_connections])
			proposed_connection_dists = -(1 + self.value_fn_uncertainty) * \
				value_fn(np.array([node.state]), proposed_connection_states)
			proposed_connection_dists = np.maximum(1 + self.value_fn_uncertainty, proposed_connection_dists)
			if node.tree in self.b_rrts:
				try:
					node_parent_idx = proposed_connections.index(node.parent)
					proposed_connection_dists[node_parent_idx] = 1.
				except ValueError:
					pass
			else:
				for child in node.children:
					try:
						node_child_idx = proposed_connections.index(child)
						proposed_connection_dists[node_child_idx] = 1.
					except ValueError:
						pass
			proposed_connection_dist_to_goals = np.array([node.dist_to_goal for node in proposed_connections])
			dist_to_goals = proposed_connection_dists + proposed_connection_dist_to_goals

			min_dist_to_goal = np.min(dist_to_goals) # get min dist to goal
			candidate_idxs = list(filter(lambda idx: dist_to_goals[idx] <= np.ceil(min_dist_to_goal), range(idx))) # get connection_nodes within 1 of min_dist_to_goal
			min_connection_dist = np.min(proposed_connection_dists[candidate_idxs]) # of those connection_nodes, get the min connection_dist
			candidate_idxs = filter(lambda idx: proposed_connection_dists[idx] <= min_connection_dist, candidate_idxs) # get connection_nodes with min_connection_dist
			final_idx = min(candidate_idxs, key=lambda idx: dist_to_goals[idx]) # of those connection_nodes, select the minimum dist_to_goal

			node.b_connection = proposed_connections[final_idx]
			node.dist_to_goal = dist_to_goals[final_idx]

		# get dist_to_starts
		all_nodes = sorted(f_rrt_nodes + b_rrt_nodes, key=lambda node: node.dist_to_start)
		start_node_idx = all_nodes.index(self.start_node)
		all_nodes.pop(start_node_idx)
		all_nodes.insert(0, self.start_node)
		for (idx, node) in enumerate(all_nodes):
			if node is self.start_node:
				continue
			proposed_connections = all_nodes[:idx]
			proposed_connection_states = np.array([node.state for node in proposed_connections])
			proposed_connection_dists = -(1 + self.value_fn_uncertainty) * \
				value_fn(proposed_connection_states, np.array([node.state]))
			proposed_connection_dists = np.maximum(1 + self.value_fn_uncertainty, proposed_connection_dists)
			if node.tree in self.f_rrts:
				try:
					node_parent_idx = proposed_connections.index(node.parent)
					proposed_connection_dists[node_parent_idx] = 1.
				except ValueError:
					pass
			else:
				for child in node.children:
					try:
						node_child_idx = proposed_connections.index(child)
						proposed_connection_dists[node_child_idx] = 1.
					except ValueError:
						pass
			proposed_connection_dist_to_starts = np.array([node.dist_to_start for node in proposed_connections])
			dist_to_starts = proposed_connection_dists + proposed_connection_dist_to_starts

			min_dist_to_start = np.min(dist_to_starts) # get min dist to start
			candidate_idxs = list(filter(lambda idx: dist_to_starts[idx] <= np.ceil(min_dist_to_start), range(idx))) # get connection_nodes within 1 of min_dist_to_start
			min_connection_dist = np.min(proposed_connection_dists[candidate_idxs]) # of those connection_nodes, get the min connection_dist
			candidate_idxs = filter(lambda idx: proposed_connection_dists[idx] <= min_connection_dist, candidate_idxs) # get connection_nodes with min_connection_dist
			final_idx = min(candidate_idxs, key=lambda idx: dist_to_starts[idx]) # of those connection_nodes, select the minimum dist_to_start

			node.f_connection = proposed_connections[final_idx]
			node.dist_to_start = dist_to_starts[final_idx]

	def plot_rrt(self, dim1=0, dim2=1):
		imsize = 500
		im = np.zeros((imsize, imsize, 3))
		x1 = int(im.shape[0] \
			* (self.start_node.state[dim1] - self.env.ob_bounds[0][dim1]) \
			/ (self.env.ob_bounds[1][dim1] - self.env.ob_bounds[0][dim1]))
		y1 = int(im.shape[1] \
			* (self.env.ob_bounds[1][dim2] - self.start_node.state[dim2]) \
			/ (self.env.ob_bounds[1][dim2] - self.env.ob_bounds[0][dim2]))
		im = cv2.circle(im, (x1, y1), imsize//50, [1, 0, 0], -1)
		x1 = int(im.shape[0] \
			* (self.goal_node.state[dim1] - self.env.ob_bounds[0][dim1]) \
			/ (self.env.ob_bounds[1][dim1] - self.env.ob_bounds[0][dim1]))
		y1 = int(im.shape[1] \
			* (self.env.ob_bounds[1][dim2] - self.goal_node.state[dim2]) \
			/ (self.env.ob_bounds[1][dim2] - self.env.ob_bounds[0][dim2]))
		im = cv2.circle(im, (x1, y1), imsize//50, [0, 0, 1], -1)
		for rrt in self.f_rrts:
			x1 = int(im.shape[0] \
				* (rrt.root.state[dim1] - self.env.ob_bounds[0][dim1]) \
				/ (self.env.ob_bounds[1][dim1] - self.env.ob_bounds[0][dim1]))
			y1 = int(im.shape[1] \
				* (self.env.ob_bounds[1][dim2] - rrt.root.state[dim2]) \
				/ (self.env.ob_bounds[1][dim2] - self.env.ob_bounds[0][dim2]))
			im = cv2.circle(im, (x1, y1), imsize//100, [0, 1, 0], -1)
			im = self.plot_tree(rrt.root, im, dim1, dim2, c=[1, 0, 0])
		for rrt in self.b_rrts:
			x1 = int(im.shape[0] \
				* (rrt.root.state[dim1] - self.env.ob_bounds[0][dim1]) \
				/ (self.env.ob_bounds[1][dim1] - self.env.ob_bounds[0][dim1]))
			y1 = int(im.shape[1] \
				* (self.env.ob_bounds[1][dim2] - rrt.root.state[dim2]) \
				/ (self.env.ob_bounds[1][dim2] - self.env.ob_bounds[0][dim2]))
			im = cv2.circle(im, (x1, y1), imsize//100, [0, 1, 0], -1)
			im = self.plot_tree(rrt.root, im, dim1, dim2, c=[0, 0, 1])

		cv2.imshow("rrt_plot", im)
		cv2.waitKey(1)

	def plot_tree(self, node, im, dim1=0, dim2=1, c=[1, 1, 1]):
		x1 = int(im.shape[0] \
			* (node.state[dim1] - self.env.ob_bounds[0][dim1]) \
			/ (self.env.ob_bounds[1][dim1] - self.env.ob_bounds[0][dim1]))
		y1 = int(im.shape[1] \
			* (self.env.ob_bounds[1][dim2] - node.state[dim2]) \
			/ (self.env.ob_bounds[1][dim2] - self.env.ob_bounds[0][dim2]))
		im = cv2.circle(im, (x1, y1), im.shape[0]//200, [0, 1, 0], -1)

		if node.f_connection not in node.children and node.f_connection is not node.parent:
			x2 = int(im.shape[0] \
				* (node.f_connection.state[dim1] - self.env.ob_bounds[0][dim1]) \
				/ (self.env.ob_bounds[1][dim1] - self.env.ob_bounds[0][dim1]))
			y2 = int(im.shape[1] \
				* (self.env.ob_bounds[1][dim2] - node.f_connection.state[dim2]) \
				/ (self.env.ob_bounds[1][dim2] - self.env.ob_bounds[0][dim2]))
			im = cv2.line(im, (x1, y1), (x2, y2), (1, 0.5, 0.5))
		if node.b_connection not in node.children and node.b_connection is not node.parent:
			x2 = int(im.shape[0] \
				* (node.b_connection.state[dim1] - self.env.ob_bounds[0][dim1]) \
				/ (self.env.ob_bounds[1][dim1] - self.env.ob_bounds[0][dim1]))
			y2 = int(im.shape[1] \
				* (self.env.ob_bounds[1][dim2] - node.b_connection.state[dim2]) \
				/ (self.env.ob_bounds[1][dim2] - self.env.ob_bounds[0][dim2]))
			im = cv2.line(im, (x1, y1), (x2, y2), (0.5, 0.5, 1))

		for child in node.children:
			x2 = int(im.shape[0] \
				* (child.state[dim1] - self.env.ob_bounds[0][dim1]) \
				/ (self.env.ob_bounds[1][dim1] - self.env.ob_bounds[0][dim1]))
			y2 = int(im.shape[1] \
				* (self.env.ob_bounds[1][dim2] - child.state[dim2]) \
				/ (self.env.ob_bounds[1][dim2] - self.env.ob_bounds[0][dim2]))
			im = cv2.line(im, (x1, y1), (x2, y2), c)

			im = self.plot_tree(child, im, dim1, dim2, c)

		return im

	def get_all_nodes(self):
		f_rrt_nodes = []
		for f_rrt in self.f_rrts:
			f_rrt_nodes.extend(f_rrt.nodes)

		b_rrt_nodes = []
		for b_rrt in self.b_rrts:
			b_rrt_nodes.extend(b_rrt.nodes)

		return f_rrt_nodes, b_rrt_nodes

	def get_all_leaves(self):
		f_rrt_leaves = []
		for f_rrt in self.f_rrts:
			f_rrt_leaves.extend(f_rrt.get_leaves())

		b_rrt_leaves = []
		for b_rrt in self.b_rrts:
			b_rrt_leaves.extend(b_rrt.get_leaves())

		return f_rrt_leaves, b_rrt_leaves

	def get_target_state(self, display):
		# proposed_states = np.random.uniform(
		# 	self.env.ob_bounds[0], self.env.ob_bounds[1],
		# 	size=(self.n_target_samples, self.env.ob_bounds[0].shape[0]))
		proposed_states = np.array([env.random_state() for _ in range(self.n_target_samples)])

		f_rrt_nodes, b_rrt_nodes = self.get_all_nodes()

		f_rrt_states = np.array([node.state for node in f_rrt_nodes])
		f_rrt_states_dist_to_start = np.array([node.dist_to_start for node in f_rrt_nodes])
		b_rrt_states = np.array([node.state for node in b_rrt_nodes])
		b_rrt_states_dist_to_goal = np.array([node.dist_to_goal for node in b_rrt_nodes])

		proposed_state_scores = np.zeros(self.n_target_samples)
		proposed_f_connections = []
		proposed_b_connections = []

		for (idx, proposed_state) in enumerate(proposed_states):
			dist_to_f_rrt_states = -(1 + self.value_fn_uncertainty) * \
				value_fn(f_rrt_states, np.array([proposed_state]))
			dist_to_b_rrt_states = -(1 + self.value_fn_uncertainty) * \
				value_fn(np.array([proposed_state]), b_rrt_states)

			dist_to_start = dist_to_f_rrt_states + f_rrt_states_dist_to_start
			dist_to_goal = dist_to_b_rrt_states + b_rrt_states_dist_to_goal

			f_connection_idx = np.argmin(dist_to_start)
			b_connection_idx = np.argmin(dist_to_goal)

			proposed_f_connections.append(f_rrt_nodes[f_connection_idx])
			proposed_b_connections.append(b_rrt_nodes[b_connection_idx])

			proposed_state_scores[idx] = -(dist_to_start[f_connection_idx] + dist_to_goal[b_connection_idx]) + \
				min(dist_to_f_rrt_states[f_connection_idx], dist_to_b_rrt_states[b_connection_idx])

		proposed_state_probs = np.exp(
			self.target_temperature * (proposed_state_scores - np.max(proposed_state_scores)))
		proposed_state_probs /= np.sum(proposed_state_probs)

		if display:
			import matplotlib.pyplot as plt
			from mpl_toolkits.mplot3d import Axes3D
			fig = plt.figure()
			ax = fig.add_subplot(111, projection='3d')
			ax.scatter3D(proposed_states[:, 0], proposed_states[:, 1], proposed_state_probs)
			plt.show()

		target_state_idx = np.random.choice(self.n_target_samples, p=proposed_state_probs)
		target_state = proposed_states[target_state_idx]
		f_connection = proposed_f_connections[target_state_idx]
		b_connection = proposed_b_connections[target_state_idx]

		return target_state, f_connection, b_connection

	def execute_plan(self, max_horizon=100):
		self.env.reset(self.start_node.state)
		current_node = self.start_node
		t = 0
		self.env.render()
		while -1 * value_fn(np.array([current_node.state]), np.array([self.goal_node.state]))[0] >= 1 and \
			t < 100:

			action = self.f_policy(np.array([current_node.state]), np.array([current_node.b_connection.state]))[0]
			# print(action)
			next_state = env.step(action)[0]
			# print(next_state)
			self.env.render()

			current_node = Node(next_state, action, current_node)

			f_rrt_nodes, b_rrt_nodes = self.get_all_nodes()
			all_nodes = f_rrt_nodes + b_rrt_nodes
			all_nodes.remove(current_node)

			all_node_states = np.array([node.state for node in all_nodes])
			b_connection_dists = -(1 + self.value_fn_uncertainty) * \
				value_fn(np.array([current_node.state]), all_node_states)
			b_connection_dist_to_goals = np.array([node.dist_to_goal for node in all_nodes])
			dist_to_goals = b_connection_dists + b_connection_dist_to_goals
			min_dist_to_goal = np.min(dist_to_goals) # get min dist to start
			candidate_idxs = list(filter(lambda idx: dist_to_goals[idx] < min_dist_to_goal + 1., range(len(all_nodes)))) # get connection_nodes within 1 of min_dist_to_goal
			min_connection_dist = max(1., np.min(b_connection_dists[candidate_idxs])) # of those connection_nodes, get the min connection_dist
			candidate_idxs = list(filter(lambda idx: b_connection_dists[idx] <= min_connection_dist, candidate_idxs)) # get connection_nodes with min_connection_dist
			b_connection_idx = min(candidate_idxs, key=lambda idx: dist_to_goals[idx]) # of those connection_nodes, select the minimum dist_to_goal
			b_connection = all_nodes[b_connection_idx]
			dist_to_goal = dist_to_goals[b_connection_idx]

			current_node.b_connection = b_connection
			current_node.dist_to_goal = dist_to_goal

			t += 1

		self.env.close()

	def get_value_training_data(self):
		states = []
		goals = []
		values = []
		# look at trees
		def add_node(node, forward):
			descendants_list = node.get_descendants_list()
			if forward:
				states.extend([node.state for _ in range(len(descendants_list))])
				goals.extend([descendant.state for descendant in descendants_list])
			else:
				goals.extend([node.state for _ in range(len(descendants_list))])
				states.extend([descendant.state for descendant in descendants_list])
			values.extend([node.depth - descendant.depth for descendant in descendants_list])

			for child in node.children:
				add_node(child, forward)

		for rrt in self.f_rrts:
			add_node(rrt.root, forward=True)
		for rrt in self.b_rrts:
			add_node(rrt.root, forward=False)

		# look at paths to goal
		f_nodes, b_nodes = self.get_all_nodes()
		all_nodes = f_nodes + b_nodes

		def get_path_to_goal(node):
			path = [node]
			dists = [0.]
			current_node = node
			while current_node is not self.goal_node:
				next_node = current_node.b_connection
				path.append(next_node)
				if (next_node.tree in self.f_rrts and next_node in current_node.children) or \
					(next_node.tree in self.b_rrts and next_node is current_node.parent):
					dists.append(dists[-1] + 1.)
				else:
					dists.append(dists[-1] + (current_node.dist_to_goal - next_node.dist_to_goal) / (1 + self.value_fn_uncertainty))
				current_node = next_node
			return path, dists

		def get_path_to_start(node):
			path = [node]
			dists = [0.]
			current_node = node
			while current_node is not self.start_node:
				next_node = current_node.f_connection
				path.append(next_node)
				if (next_node.tree in self.b_rrts and next_node in current_node.children) or \
					(next_node.tree in self.f_rrts and next_node is current_node.parent):
					dists.append(dists[-1] + 1.)
				else:
					dists.append(dists[-1] + (current_node.dist_to_start - next_node.dist_to_start) / (1 + self.value_fn_uncertainty))
				current_node = next_node
			return path, dists

		for node in all_nodes:
			f_path, f_dists = get_path_to_goal(node)
			# print([node.state for node in f_path])
			# print(f_dists)
			states.append(node.state)
			idx = np.random.randint(len(path))
			goals.append(path[idx].state)
			values.append(-dists[idx])

			b_path, b_dists = get_path_to_start(node)
			idx = np.random.randint(len(path))
			states.append(path[idx].state)
			goals.append(node.state)
			values.append(-dists[idx])

		return np.array(states), np.array(goals), np.array(values)

	def get_policy_training_data(self):
		f_states = []
		f_goals = []
		f_actions = []
		f_action_weights = []

		b_states = []
		b_goals = []
		b_actions = []
		b_action_weights = []

		def add_node(node, forward):
			for child in node.children:
				descendants_list = child.get_descendants_list()
				descendant_states = np.array([descendant.state for descendant in descendants_list])

				if forward:
					baseline_dists = self.value_fn(np.array([node.state]), descendant_states)
				else:
					baseline_dists = self.value_fn(descendant_states, np.array([node.state]))
				descendant_depths = [descendant.depth for descendant in descendants_list]
				descendant_dists = np.array(descendant_depths) - node.depth
				action_weights = np.exp(self.policy_temperature * (baseline_dists - descendant_dists))

				if forward:
					f_states.extend([node.state for _ in range(len(descendants_list))])
					f_goals.extend([descendant.state for descendant in descendants_list])
					f_actions.extend([child.action for _ in range(len(descendants_list))])
					f_action_weights.extend(action_weights.tolist())

					b_states.extend([node.state for _ in range(len(descendants_list))])
					b_goals.extend([descendant.state for descendant in descendants_list])
					b_actions.extend([descendant.action for descendant in descendants_list])
					b_action_weights.extend(action_weights.tolist())
				else:
					b_states.extend([descendant.state for descendant in descendants_list])
					b_goals.extend([node.state for _ in range(len(descendants_list))])
					b_actions.extend([node.action for _ in range(len(descendants_list))])
					b_action_weights.extend(action_weights.tolist())

					f_states.extend([descendant.state for descendant in descendants_list])
					f_goals.extend([node.state for _ in range(len(descendants_list))])
					f_actions.extend([descendant.action for descendant in descendants_list])
					f_action_weights.extend(action_weights.tolist())

				add_node(child, forward)

		for rrt in self.f_rrts:
			add_node(rrt.root, forward=True)
		for rrt in self.b_rrts:
			add_node(rrt.root, forward=False)

		for (s, f, a) in zip(f_states, f_goals, f_actions):
			print(s, f, a)

		f_nodes, b_nodes = self.get_all_nodes()
		all_nodes = f_nodes + b_nodes

		def get_path_to_goal(node):
			path = []
			current_node = node
			while current_node is not self.goal_node:
				next_node = current_node.b_connection
				path.append(next_node)
			return path

		def get_path_to_start(node):
			path = []
			current_node = node
			while current_node is not self.start_node:
				next_node = current_node.f_connection
				path.append(next_node)
			return path

		for node in all_nodes:
			path_to_goal = get_path_to_goal(node)
			path_to_start = get_path_to_start(node)

			if path_to_goal:
				local_goal = np.random.choice(path_to_goal)

				direct_action = self.f_policy(np.array([node.state]), np.array([local_goal.state]))[0]
				direct_next_state = self.f_dynamics(np.array([node.state]), np.array([direct_action]))[0]

				if node.tree in self.f_rrts:
					actions = [child.action for child in node.children] + [direct_action]
					next_states = [child.state for child in node.children] + [direct_next_state]
				else:
					if node.parent:
						actions = [node.parent.action, direct_action]
						next_states = [node.parent.state, direct_next_state]
					else:
						actions = [direct_action]
						next_states = [direct_next_state]

				next_dist_to_local_goals = -self.value_fn(np.array(next_states), np.array([local_goal]))
				baseline_dist_to_local_goal = -self.value_fn(np.array([node.state]), np.array([local_goal]))[0]
				#action_weights = np.exp(self.policy_temperature * (np.min(next_dist_to_local_goals) - next_dist_to_local_goals))
				#action_weights /= np.sum(action_weights)
				action_weights = np.exp(self.policy_temperature * (baseline_dist_to_local_goal - (1 + next_dist_to_local_goals)))

				f_states.extend([node.state for _ in range(len(actions))])
				f_actions.extend(actions)
				f_goals.extend([local_goal.state for _ in range(len(actions))])
				f_action_weights.extend(action_weights.tolist())

			if path_to_start:
				local_start = np.random.choice(path_to_start)

				direct_action = self.b_policy(np.array([local_start.state]), np.array([node.state]))[0]
				direct_next_state = self.b_dynamics(np.array([node.state]), np.array([direct_action]))[0]

				if node.tree in self.b_rrts:
					actions = [child.action for child in node.children] + [direct_action]
					next_states = [child.state for child in node.children] + [direct_next_state]
				else:
					if node.parent:
						actions = [node.parent.action, direct_action]
						next_states = [node.parent.state, direct_next_state]
					else:
						actions = [direct_action]
						next_states = [direct_next_state]

				next_dist_to_local_starts = -self.value_fn(np.array([local_start]), np.array(next_states))
				baseline_dist_to_local_start = -self.value_fn(np.array([local_start]), np.array([node.state]))[0]
				#action_weights = np.exp(self.policy_temperature * (np.min(next_dist_to_local_goals) - next_dist_to_local_goals))
				#action_weights /= np.sum(action_weights)
				action_weights = np.exp(self.policy_temperature * (baseline_dist_to_local_start - (1 + next_dist_to_local_starts)))

				b_states.extend([local_start.state for _ in range(len(actions))])
				b_actions.extend(actions)
				b_goals.extend([node.state for _ in range(len(actions))])
				b_action_weights.extend(action_weights.tolist())

		return (f_states, f_goals, f_actions, f_action_weights), (b_states, b_goals, b_actions, b_action_weights)

if __name__ == "__main__":
	from envs.pointmass import WallPointEnv
	env = WallPointEnv()

	start_state = np.array([-2., -2.])
	goal_state = np.array([2., 2.])

	def f_policy(s, g):
		a = g - s
		a /= np.maximum(np.ones(a.shape[0]), (np.linalg.norm(a, axis=-1) + 1e-8))
		return a

	def b_policy(s, g):
		a = g - s
		a /= np.maximum(np.ones(a.shape[0]), (np.linalg.norm(a, axis=-1) + 1e-8))
		return a

	def f_dynamics(s, a):
		res = []
		for i in range(len(s)):
			env.reset(s[i])
			res.append(env.step(a[i])[0])
		return np.array(res)

	def b_dynamics(s, a):
		res = []
		for i in range(len(s)):
			env.reset(s[i])
			res.append(env.step(-a[i])[0])
		return np.array(res)

	#def value_fn(s, s_prime):
	#	return -np.linalg.norm(s - s_prime, axis=-1)
	from nn import ValueFn
	value_fn = ValueFn([4, 100, 100, 1])

	rrt = VfPolRrt(
		env,
		start_state, goal_state,
		f_policy, f_dynamics,
		b_policy, b_dynamics,
		value_fn,

		n_target_samples=1000,
		target_temperature=1.0,

		value_fn_uncertainty=0.05,

		policy_temperature=2.0,
	)

	for itr in tqdm.tqdm(range(101)):
		display = (itr % 10 == 0)
		rrt.build_rrt(10, display)
		for inner in range(10):
			rrt.get_node_distances()
			states, goals, values = rrt.get_value_training_data()
			# print(states, goals, values)
			value_fn.optimize(states, goals, values)
		(f_states, f_goals, f_actions, f_action_weights), (b_states, b_goals, b_actions, b_action_weights) = \
			rrt.get_policy_training_data()
		rrt.clear()

	#rrt.execute_plan(100)
    # needed to train dynamics, however, use GT dynamics for now...
