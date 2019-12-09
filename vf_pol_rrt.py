import numpy as np
import cv2

class Tree():
	def __init__(self, start_node):
		self.root = start_node
		self.nodes = [start_node]

class Node():
	def __init__(self, state, action):
		self.parent = None
		self.children = []
		self.depth = 0

		self.best_child = None
		self.dist_to_start = 0
		self.dist_to_goal = 0

		self.state = state
		self.action = action

class VfPolRrt:
	def __init__(self,
		env,
		start_state, goal_state,
		f_policy, f_dynamics,
		b_policy, b_dynamics,
		value_fn,

		n_target_samples=1000,
		temperature=0.1,

		value_fn_uncertainty=0.05,
	):
		self.env = env

		self.f_policy = f_policy
		self.b_policy = b_policy
		self.f_dynamics = f_dynamics
		self.b_dynamics = b_dynamics
		self.value_fn = value_fn

		# TODO: keep track of many forward/backward trees
		start_node = Node(start_state, None)
		# self.f_rrts = [Tree(start_node)]
		self.f_rrt = Tree(start_node)
		goal_node = Node(goal_state, None)
		# self.b_rrts = [Tree(goal_node)]
		self.b_rrt = Tree(goal_node)

		self.n_target_samples = n_target_samples
		self.temperature = temperature

		self.value_fn_uncertainty = value_fn_uncertainty

	def find_path(self):
		pass

	def execute_path(self):
		pass

	def get_node_values(self):
		pass

	def build_rrt(self, n_samples=1):
		for sample_n in range(n_samples):
			target_state = self.get_target_state()
			target = Node(target_state, None)

			self.connect_target(self.f_rrt, target)
			self.connect_target(self.b_rrt, target)

		cv2.waitKey(0)
		cv2.destroyWindow("rrt_plot")

		return self.f_rrt, self.b_rrt

	def plot_rrt(self, dim1=0, dim2=1):
		imsize = 500
		im = np.zeros((imsize, imsize))
		im = self.plot_tree(self.f_rrt.root, im, dim1, dim2)
		# im = self.plot_tree(self.b_rrt.root, im, dim1, dim2)
		cv2.imshow("rrt_plot", im)
		cv2.waitKey(1)

	def plot_tree(self, node, im, dim1=0, dim2=1):
		for child in node.children:
			x1 = int(im.shape[0] \
				* (node.state[dim1] - self.env.ob_bounds[0][dim1]) \
				/ (self.env.ob_bounds[1][dim1] - self.env.ob_bounds[0][dim1]))
			y1 = int(im.shape[1] \
				* (self.env.ob_bounds[1][dim2] - node.state[dim2]) \
				/ (self.env.ob_bounds[1][dim2] - self.env.ob_bounds[0][dim2]))

			x2 = int(im.shape[0] \
				* (child.state[dim1] - self.env.ob_bounds[0][dim1]) \
				/ (self.env.ob_bounds[1][dim1] - self.env.ob_bounds[0][dim1]))
			y2 = int(im.shape[1] \
				* (self.env.ob_bounds[1][dim2] - child.state[dim2]) \
				/ (self.env.ob_bounds[1][dim2] - self.env.ob_bounds[0][dim2]))

			im = cv2.line(im, (x1, y1), (x2, y2), 1)

			im = self.plot_tree(child, im, dim1, dim2)

		return im

	def connect_target(self, rrt, target, connection=None, connection_dist=0):
		# get connection if not pre-specified
		if connection is None:
			connection, connection_dist, dist = \
				self.get_connection(rrt, target)
			connection = connection
			connection_dist = connection_dist

		# attempt to connect by rolling out policy
		current_node = connection
		for t in range(int(np.ceil(connection_dist))):
			# get new node on path from connection to target
			if rrt is self.f_rrt:
				action = self.f_policy(np.array([current_node.state]), np.array([target.state]))[0]
				next_state = self.f_dynamics(np.array([current_node.state]), np.array([action]))[0]
			else:
				action = self.b_policy(np.array([target.state]), np.array([current_node.state]))[0]
				next_state = self.b_dynamics(np.array([current_node.state]), np.array([action]))[0]
			new_node = Node(next_state, action)

			# new_node.parent = current_node
			# new_node.depth = current_node.depth + 1
			# current_node.children.append(new_node)
			# rrt.nodes.append(new_node)

			# decide if the new node should be connected to the current node or, if there is a shortcut, rewire
			new_node_connection, new_node_connection_dist, new_node_total_dist = \
				self.get_connection(rrt, new_node)
			if (new_node_connection is current_node) or \
			not (new_node_total_dist + 1 < current_node.depth + 1):
				new_node.parent = current_node
				new_node.depth = current_node.depth + 1
				current_node.children.append(new_node)
				rrt.nodes.append(new_node)
			else:
				print("new node", new_node.state)
				print("current node", current_node.state, current_node.depth, 1)
				print("new node connection", new_node_connection.state, new_node_connection.depth, new_node_connection_dist)
				self.connect_target(rrt, new_node, new_node_connection, new_node_connection_dist)
			print("connected new node", new_node.state)
			self.plot_rrt()

			# # check if other nodes should be rewired through the new one, and rewire if so
			# for node in rrt.nodes:
			# 	if new_node.depth + 1 < node.depth:
			# 		if rrt is self.f_rrt:
			# 			node_thru_new_connection_dist = \
			# 				-(1. + self.value_fn_uncertainty) * value_fn(np.array([new_node.state]), np.array([node.state]))[0]
			# 			node_thru_new_dist = node_thru_new_connection_dist + new_node.depth
			# 		else:
			# 			node_thru_new_connection_dist = \
			# 				-(1. + self.value_fn_uncertainty) * value_fn(np.array([node.state]), np.array([new_node.state]))[0]
			# 			node_thru_new_dist = node_thru_new_connection_dist + new_node.depth

			# 		if node_thru_new_dist + 1 < node.depth:
			# 			print("rewiring:")
			# 			print("node", node.state, node.depth)
			# 			print("thru new node", new_node.state, new_node.depth, node_thru_new_connection_dist)
			# 			try:
			# 				node.parent.children.remove(node)
			# 				self.connect_target(rrt, node, new_node, node_thru_new_connection_dist)
			# 			except:
			# 				pass
			# 			self.plot_rrt()

			current_node = new_node
			# self.plot_rrt()

		# # connect target's children, if any
		# for child in target.children:
		# 	target.children.remove(child)
		# 	self.connect_target(rrt, child)

	def get_connection(self, rrt, target):
		rrt_states = np.array([node.state for node in rrt.nodes])

		if rrt is self.f_rrt:
			values = value_fn(rrt_states, np.array([target.state]))
		else:
			values = value_fn(np.array([target.state]), rrt_states)

		dists = -(1. + self.value_fn_uncertainty) * values + \
			np.array([node.depth for node in rrt.nodes])

		best_idx = np.argmin(dists)
		connection = rrt.nodes[best_idx]
		connection_dist = -(1. + self.value_fn_uncertainty) * values[best_idx]
		dist = dists[best_idx]

		return connection, connection_dist, dist

	def get_target_state(self):
		# alternatively, use CEM to improve, but probably not necessary
		proposed_states = np.random.uniform(
			self.env.ob_bounds[0], self.env.ob_bounds[1],
			size=(self.n_target_samples, self.env.ob_bounds[0].shape[0]))

		f_rrt_states = np.array([node.state for node in self.f_rrt.nodes])
		b_rrt_states = np.array([node.state for node in self.b_rrt.nodes])

		proposed_state_scores = np.zeros(self.n_target_samples)
		for (idx, proposed_state) in enumerate(proposed_states):
			f_rrt_max_value = np.max(value_fn(f_rrt_states, np.array([proposed_state])))
			b_rrt_max_value = np.max(value_fn(np.array([proposed_state]), b_rrt_states))

			proposed_state_scores[idx] = min(f_rrt_max_value, b_rrt_max_value)

		proposed_state_probs = np.exp(
			self.temperature * (proposed_state_scores - np.max(proposed_state_scores)))
		proposed_state_probs /= np.sum(proposed_state_probs)

		target_state_idx = np.random.choice(self.n_target_samples, p=proposed_state_probs)
		target_state = proposed_states[target_state_idx]

		return target_state

if __name__ == "__main__":
	from envs.pointmass import WallPointEnv
	env = WallPointEnv()

	start_state = np.array([-2., -2.])
	goal_state = np.array([2., 2.])

	def f_policy(s, g):
		a = g - s
		a /= (np.linalg.norm(a, axis=-1) + 1e-8)
		return a

	def b_policy(s, g):
		a = g - s
		a /= (np.linalg.norm(a, axis=-1) + 1e-8)
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

	def value_fn(s, s_prime):
		return -np.linalg.norm(s - s_prime, axis=-1)

	rrt = VfPolRrt(
		env,
		start_state, goal_state,
		f_policy, f_dynamics,
		b_policy, b_dynamics,
		value_fn,

		n_target_samples=1000,
		temperature=1.0,

		value_fn_uncertainty=0.05,
	)

	rrt.build_rrt(100)