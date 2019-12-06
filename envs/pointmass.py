import numpy as np
import cv2
from scipy.stats import truncnorm
from gym.envs.mujoco import mujoco_env
from gym import utils

class PointEnv:
    def __init__(self, ob_dim=2):
        self.ob_dim = ob_dim
        self.action_dim = self.ob_dim
        self.action_bounds = [-np.ones(self.ob_dim), np.ones(self.ob_dim)]
        self.reset()

    def reset(self, args=None):
        if args is None:
            self.point = np.random.normal(size=2)
            return self.get_obs()
        else:
            self.point = args.astype(np.float32)
            return self.get_obs()

    def step(self, action):
        self.point += action
        return self.get_obs(), 0, False, {}

    def get_obs(self):
        return self.point.copy()

    def render(self):
        if not hasattr(self, 'window'):
            self.window = 'render'
        image = np.zeros((101, 101, 3), dtype=np.uint8)
        center = np.array([self.point[0]+51, -self.point[1]+51]).astype(np.int32)
        image = cv2.circle(image, tuple(center.tolist()), 5, [255, 255, 255], -1)
        cv2.imshow(self.window, image)
        cv2.waitKey(1)

    def close(self):
        if not hasattr(self, 'window'):
            pass
        cv2.destroyWindow(self.window)

class WallPointEnv:
    def __init__(self, ob_dim=2):
        self.ob_dim = ob_dim
        self.ob_bounds = [-2.5*np.ones(self.ob_dim), 2.5*np.ones(self.ob_dim)]
        self.action_dim = self.ob_dim
        self.action_bounds = [-np.ones(self.ob_dim), np.ones(self.ob_dim)]
        self.reset()

    def reset(self, args=None):
        if args is None:
            inside_wall = True
            while inside_wall:
                self.point = truncnorm.rvs(-2.5, 2.5, size=2)
                inside_wall = self.point[0] >= -0.5 and self.point[0] <= 0.5 \
                    and self.point[1] >= -1.5 and self.point[1] <= 1.5
            return self.get_obs()
        else:
            self.point = args.astype(np.float32)
            return self.get_obs()

    def step(self, action):
        next_point = self.point + np.clip(action, self.action_bounds[0], self.action_bounds[1])
        
        line_distances = 10*np.ones(4)
        line_intersections = 10*np.zeros([4, 2])
        if action[0] != 0:
            action_mult = (-0.5-self.point[0])/action[0]
            intersection_point = self.point + action*action_mult
            if action_mult >= 0 and action_mult <= 1 \
            and intersection_point[1] >= -1.5 and intersection_point[1] <= 1.5 \
            and action[0] > 0:
                line_distances[0] = np.linalg.norm(intersection_point - self.point)
                line_intersections[0] = intersection_point

            action_mult = (0.5-self.point[0])/action[0]
            intersection_point = self.point + action*action_mult
            if action_mult >= 0 and action_mult <= 1 \
            and intersection_point[1] >= -1.5 and intersection_point[1] <= 1.5 \
            and action[0] < 0:
                line_distances[2] = np.linalg.norm(intersection_point - self.point)
                line_intersections[2] = intersection_point

        if action[1] != 0:
            action_mult = (1.5-self.point[1])/action[1]
            intersection_point = self.point + action*action_mult
            if action_mult >= 0 and action_mult <= 1 \
            and intersection_point[0] >= -0.5 and intersection_point[0] <= 0.5 \
            and action[1] < 0:
                line_distances[1] = np.linalg.norm(intersection_point - self.point)
                line_intersections[1] = intersection_point

            action_mult = (-1.5-self.point[1])/action[1]
            intersection_point = self.point + action*action_mult
            if action_mult >= 0 and action_mult <= 1 \
            and intersection_point[0] >= -0.5 and intersection_point[0] <= 0.5 \
            and action[1] > 0:
                line_distances[3] = np.linalg.norm(intersection_point - self.point)
                line_intersections[3] = intersection_point
        
        if np.min(line_distances) < 10:
            self.point = line_intersections[np.argmin(line_distances)]
        else:
            self.point = np.clip(next_point, -2.5, 2.5)

        return self.get_obs(), 0, False, {}

    def get_obs(self):
        return self.point.copy()

    def render(self):
        if not hasattr(self, 'window'):
            self.window = 'render'
        image = np.zeros((101, 101, 3), dtype=np.uint8)
        center = np.array([20*self.point[0]+51, -20*self.point[1]+51]).astype(np.int32)
        image = cv2.circle(image, tuple(center.tolist()), 5, [255, 255, 255], -1)
        image = cv2.rectangle(image, (40, 20), (60, 80), [255, 63, 63], -1)
        cv2.imshow(self.window, image)
        cv2.waitKey(1)

    def close(self):
        if not hasattr(self, 'window'):
            pass
        cv2.destroyWindow(self.window)

class RampPointEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, '/home/ubuntu/distance_representation/envs/assets/ramp_pointmass.xml', 12)
        utils.EzPickle.__init__(self)
        self.ob_dim = self.observation_space.shape[0]
        self.action_dim = self.action_space.shape[0]
        # self.init_qpos = np.array([0,0,0,1,0,0,0])
        # self.init_qpos = np.array([0,0,0.5])

    def reset(self, args=None):
        if args is None:
            return self.reset_model()
        else:
            self.set_state(args[:2], args[2:])
            return self.get_obs()

    def step(self, a):
        a = 10*a
        self.do_simulation(a, self.frame_skip)
        done = False
        ob = self.get_obs()
        qpos = self.sim.data.qpos
        self.set_state(qpos, np.zeros_like(self.sim.data.qvel))
        return ob, 0, done, {}

    def get_obs(self):
        # qpos = self.sim.data.qpos.flatten()
        # return np.concatenate([qpos, self.sim.data.qvel.flatten()])
        return self.sim.data.qpos.flatten()[:2] #+ np.random.normal(size=2)

    def reset_model(self):
        qpos = self.init_qpos
        qvel = self.init_qvel
        self.set_state(qpos, qvel)
        return self.get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = 10

    def render(self):
        im = self.sim.render(64, 64, camera_name='maincam')[::-1, :, ::-1]
        cv2.imshow('render', im)
        cv2.waitKey(1)
        return im

    def close(self):
        cv2.destroyWindow('render')

if __name__ == '__main__':
    env = RampPointEnv()
    env.reset()
    actions = np.array([
        [-1,0],
        [-1,0],
        [-1,0],
        [-1,0],
        [-1,0],
        [-1,0],
        [-1,0],
        [-1,0],
        [-1,1],
        [-1,1],
        [-1,1],
        [-1,1],
        [-1,1],
        [-1,1],
        [-1,1],
        [-1,1],
        [0,-1],
        [0,-1],
        [0,-1],
        [0,-1],
        [0,-1],
        [0,-1],
        [0,-1],
        # [0,0],
        # [0,0],
        # [0,0],
        # [0,0],
        # [0,0],
        # [0,0],
        # [0,0],
        # [0,0],
        [1,0],
        [1,0],
        [1,0],
        # [0,0],
        # [1,0],
        # [-1,0],
        # [-1,0],
        [-1,0],
        [-1,0],
        [-1,0],
        [0,-1],
        [0,-1],
        [0,-1],
        [0,-1],
        [0,-1],
        [0,-1],
        [0,-1],
        [0,-1],
        [0,-1],
        [0,-1],
        [1,0],
        [1,0],
        [1,0],
        [1,0],
        [1,0],
        [1,0],
        [1,0],
        [1,0],
        [1,0],
        [1,0],
        [1,0],
        [1,0],
        [1,0],
        [1,0],
        [1,0],
        [1,0],
        [0,1],
        [0,1],
        [0,1],
        [0,-1],
        [0,-1],
        [0,-1],
        [0,1],
        [0,1],
        [0,1],
        [0,-1],
        [0,-1],
        [0,-1],
        [0,1],
        [0,1],
        [0,1],
        [0,-1],
        [0,-1],
        [0,-1],
    ])
    # actions = np.random.normal(size=(1000, 2))
    import time
    for action in actions:
        env.render()
        print(env.step(action)[0])
        time.sleep(0.1)
    time.sleep(1)
    env.close()
