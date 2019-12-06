import numpy as np

class Tree():
	def __init__(self, start_node):
		self.root = start_node
		self.nodes = [start_node]
		self.best_node = start_node

class Node():
	def __init__(self, state, action_to):
		self.parent = None
		self.children = []
		self.state = state
		self.action_to = action_to

def naive_rrt(env, max_samples, goal, to_stop_fn):
	# returns tree and path from current state to goal
	assert len(goal) == env.ob_dim

	start_state = env.get_obs()

	start_node = Node(start_state, None)
	rrt = Tree(start_node)

	for sample_n in range(max_samples):
		random_state = np.random.uniform(env.ob_bounds[0], env.ob_bounds[1])

		nn_node = None
		nn_dist = None

		for node in rrt.nodes:
			dist = np.linalg.norm(random_state - node.state)
			if (nn_dist is None) or (dist < nn_dist):
				nn_node = node
				nn_dist = dist

		# TODO: ONLY WORKS IN POINTMASS!!!
		new_action = random_state - nn_node.state
		new_action /= (np.linalg.norm(new_action) + 1e-8)

		old_state = env.get_obs()
		env.reset(nn_node.state)
		new_state = env.step(new_action)[0]

		new_node = Node(new_state, new_action)
		new_node.parent = nn_node
		nn_node.children.append(new_node)
		rrt.nodes.append(new_node)

		if np.linalg.norm(new_node.state - goal) < np.linalg.norm(rrt.best_node.state - goal):
			rrt.best_node = new_node

		if to_stop_fn(goal, new_node.state):
			break

	path_states = []
	path_actions = []
	backtrack_node = rrt.best_node
	while backtrack_node is not rrt.root:
		path_states.append(backtrack_node.state)
		path_actions.append(backtrack_node.action_to)
		backtrack_node = backtrack_node.parent

	return rrt, list(reversed(path_states)), list(reversed(path_actions))

if __name__ == "__main__":
	import time

	from envs.pointmass import WallPointEnv
	env = WallPointEnv()

	start_state = np.array([-2.0, -2.0])
	env.reset(start_state)

	goal = np.array([2.0, 2.0])

	def stop_criterion(goal, state):
		return np.linalg.norm(goal - state) < 0.2

	rrt, path_states, path_actions = naive_rrt(env, 1000, goal, stop_criterion)

	# import matplotlib.pyplot as plt
	# plt.plot(np.array(path_states).T[0], np.array(path_states).T[1])
	# plt.show()

	env.reset(start_state)
	env.render()
	time.sleep(0.1)
	for action in path_actions:
		env.step(action)
		env.render()
		time.sleep(0.1)
	env.close()