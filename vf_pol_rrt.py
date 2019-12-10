import numpy as np
import cv2

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
		start_node = Node(start_state, None, None)
		self.f_rrts = [start_node.tree]
		goal_node = Node(goal_state, None, None)
		self.b_rrts = [goal_node.tree]

		self.n_target_samples = n_target_samples
		self.temperature = temperature

		self.value_fn_uncertainty = value_fn_uncertainty

	def build_rrt(self, n_samples=1):
		for sample_n in range(n_samples):
			target_state, f_connection, b_connection = self.get_target_state()
			print("target_state:", target_state)
			print("f_connection:", f_connection.state)
			print("b_connection:", b_connection.state)
			f_target = Node(target_state, None, None)
			f_target.dist_to_start = -1 * value_fn(np.array([self.f_rrts[0].root.state]), np.array([f_target.state]))[0]
			f_target.dist_to_goal = -1 * value_fn(np.array([f_target.state]), np.array([self.b_rrts[0].root.state]))[0]
			b_target = Node(target_state, None, None)
			b_target.dist_to_start = -1 * value_fn(np.array([self.f_rrts[0].root.state]), np.array([b_target.state]))[0]
			b_target.dist_to_goal = -1 * value_fn(np.array([b_target.state]), np.array([self.b_rrts[0].root.state]))[0]

			self.f_rrts.append(f_target.tree)
			self.b_rrts.append(b_target.tree)

			f_dist = -1 * self.value_fn(np.array([f_connection.state]), np.array([target_state]))[0]
			b_dist = -1 * self.value_fn(np.array([target_state]), np.array([b_connection.state]))[0]

			f_rrt_node = f_connection
			target_node = b_target
			print("f_dist:", f_dist)
			for t in range(int(np.ceil(f_dist/2))):
				f_action = self.f_policy(np.array([f_rrt_node.state]), np.array([target_state]))[0]
				f_rrt_next_state = self.f_dynamics(np.array([f_rrt_node.state]), np.array([f_action]))[0]
				f_rrt_node = Node(f_rrt_next_state, f_action, f_rrt_node)
				f_rrt_node.dist_to_start = -1 * value_fn(np.array([self.f_rrts[0].root.state]), np.array([f_rrt_node.state]))[0]
				f_rrt_node.dist_to_goal = -1 * value_fn(np.array([f_rrt_node.state]), np.array([self.b_rrts[0].root.state]))[0]
				# print(f_rrt_node.dist_to_start, f_rrt_node.dist_to_goal)

				target_action = self.b_policy(np.array([f_connection.state]), np.array([target_node.state]))[0]
				target_rrt_next_state = self.b_dynamics(np.array([target_node.state]), np.array([target_action]))[0]
				target_node = Node(target_rrt_next_state, target_action, target_node)
				target_node.dist_to_start = -1 * value_fn(np.array([self.f_rrts[0].root.state]), np.array([target_node.state]))[0]
				target_node.dist_to_goal = -1 * value_fn(np.array([target_node.state]), np.array([self.b_rrts[0].root.state]))[0]
				# print(target_node.dist_to_start, target_node.dist_to_goal)

				self.plot_rrt()
			b_rrt_node = b_connection
			target_node = f_target
			print("b_dist:", b_dist)
			for t in range(int(np.ceil(b_dist/2))):
				b_action = self.b_policy(np.array([target_state]), np.array([b_rrt_node.state]))[0]
				b_rrt_next_state = self.b_dynamics(np.array([b_rrt_node.state]), np.array([b_action]))[0]
				b_rrt_node = Node(b_rrt_next_state, b_action, b_rrt_node)
				b_rrt_node.dist_to_start = -1 * value_fn(np.array([self.f_rrts[0].root.state]), np.array([b_rrt_node.state]))[0]
				b_rrt_node.dist_to_goal = -1 * value_fn(np.array([b_rrt_node.state]), np.array([self.b_rrts[0].root.state]))[0]
				# print(b_rrt_node.dist_to_start, b_rrt_node.dist_to_goal)

				target_action = self.f_policy(np.array([target_node.state]), np.array([b_connection.state]))[0]
				target_rrt_next_state = self.f_dynamics(np.array([target_node.state]), np.array([target_action]))[0]
				target_node = Node(target_rrt_next_state, target_action, target_node)
				target_node.dist_to_start = -1 * value_fn(np.array([self.f_rrts[0].root.state]), np.array([target_node.state]))[0]
				target_node.dist_to_goal = -1 * value_fn(np.array([target_node.state]), np.array([self.b_rrts[0].root.state]))[0]
				# print(target_node.dist_to_start, target_node.dist_to_goal)

				self.plot_rrt()
				
			# for f_rrt in self.f_rrts:
			# 	print([node.state for node in f_rrt.nodes])
			# for b_rrt in self.b_rrts:
			# 	print([node.state for node in b_rrt.nodes])
			# self.plot_rrt()

		cv2.waitKey(1)
		cv2.destroyWindow("rrt_plot")

	def plot_rrt(self, dim1=0, dim2=1):
		imsize = 500
		im = np.zeros((imsize, imsize, 3))
		for rrt in self.f_rrts:
			im = self.plot_tree(rrt.root, im, dim1, dim2, c=[255, 0, 0])
		for rrt in self.b_rrts:
			im = self.plot_tree(rrt.root, im, dim1, dim2, c=[0, 0, 255])

		cv2.imshow("rrt_plot", im)
		cv2.waitKey(0)

	def plot_tree(self, node, im, dim1=0, dim2=1, c=[255, 255, 255]):
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

	def get_target_state(self):
		proposed_states = np.random.uniform(
			self.env.ob_bounds[0], self.env.ob_bounds[1],
			size=(self.n_target_samples, self.env.ob_bounds[0].shape[0]))

		f_rrt_nodes, b_rrt_nodes = self.get_all_nodes()

		f_rrt_states = np.array([node.state for node in f_rrt_nodes])
		f_rrt_states_dist_to_start = np.array([node.dist_to_start for node in f_rrt_nodes])
		b_rrt_states = np.array([node.state for node in b_rrt_nodes])
		b_rrt_states_dist_to_goal = np.array([node.dist_to_goal for node in b_rrt_nodes])

		proposed_state_scores = np.zeros(self.n_target_samples)
		proposed_f_connections = []
		proposed_b_connections = []

		for (idx, proposed_state) in enumerate(proposed_states):
			dist_to_f_rrt_states = -(1 + self.value_fn_uncertainty) * value_fn(f_rrt_states, np.array([proposed_state]))
			dist_to_b_rrt_states = -(1 + self.value_fn_uncertainty) * value_fn(np.array([proposed_state]), b_rrt_states)

			dist_to_start = dist_to_f_rrt_states + f_rrt_states_dist_to_start
			dist_to_goal = dist_to_b_rrt_states + b_rrt_states_dist_to_goal

			f_connection_idx = np.argmin(dist_to_start)
			b_connection_idx = np.argmin(dist_to_goal)

			proposed_f_connections.append(f_rrt_nodes[f_connection_idx])
			proposed_b_connections.append(b_rrt_nodes[b_connection_idx])

			proposed_state_scores[idx] = -(dist_to_start[f_connection_idx] + dist_to_goal[b_connection_idx]) + \
				min(dist_to_f_rrt_states[f_connection_idx], dist_to_b_rrt_states[b_connection_idx])

		proposed_state_probs = np.exp(
			self.temperature * (proposed_state_scores - np.max(proposed_state_scores)))
		proposed_state_probs /= np.sum(proposed_state_probs)

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