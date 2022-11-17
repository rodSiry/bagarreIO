import mujoco as mj
import time
import pybullet as p
import pybullet_data
import random
import numpy as np
import gym
         
class FightingEnv(gym.Env):
    def __init__(self):
        self.model = mj.MjModel.from_xml_path('tatakAI/assets/humanoids.xml')
        self.data = mj.MjData(self.model)

        o1, o2 = self.get_observation()
        self.n_actions = len(self.data.ctrl) // 2 
        self.n_observations = len(o1) + len(o2)
        self.n_joints = len(self.data.qpos) // 2

        self.action_space = gym.spaces.box.Box(
            low = -np.ones((self.n_actions)),
            high = np.ones((self.n_actions)))

        self.observation_space = gym.spaces.box.Box(
            low = -np.ones((self.n_observations)),
            high = np.ones((self.n_observations)))

        self.dist = np.sum((self.data.body('torso').xpos - self.data.body('2torso').xpos) ** 2) ** 0.5
        self.render_init = False

    def step(self, action1, action2):
        self.timeCount += 1
        end = 0
        r = 0

        full_action = np.concatenate([action1, action2], 0)
        self.data.ctrl = full_action

        mj.mj_step(self.model, self.data)

        o1, o2 = self.get_observation()
        
        new_dist = np.sum((self.data.body('torso').xpos - self.data.body('2torso').xpos) ** 2) ** 0.5
        r = new_dist - self.dist

        self.dist = new_dist

        if self.timeCount == 1000:
            end = 1

        return np.concatenate([o1, o2], 0), r, end, None

    def get_observation(self):
        obs1 = np.concatenate([self.data.qpos[:len(self.data.qpos)//2] , self.data.qvel[:len(self.data.qvel)//2], self.data.body('torso').xpos, self.data.body('torso').xquat], axis=0)
        obs2 = np.concatenate([self.data.qpos[len(self.data.qpos)//2:] , self.data.qvel[len(self.data.qvel)//2:], self.data.body('2torso').xpos, self.data.body('2torso').xquat], axis=0)
        return obs1, obs2

    def reset(self):
        self.model = mj.MjModel.from_xml_path('tatakAI/assets/humanoids.xml')
        self.data = mj.MjData(self.model)
        self.timeCount = 0
        o1, o2 = self.get_observation()
        self.dist = np.sum((self.data.body('torso').xpos - self.data.body('2torso').xpos) ** 2) ** 0.5
        return np.concatenate([o1, o2], 0)

    def render(self):
        if not self.render_init:
            self.cam = mj.MjvCamera()
            self.opt = mj.MjvOption()
            mj.glfw.glfw.init()
            self.window = mj.glfw.glfw.create_window(1200, 900, "Demo", None, None)
            mj.glfw.glfw.make_context_current(self.window)
            mj.glfw.glfw.swap_interval(1)
            mj.mjv_defaultCamera(self.cam)
            mj.mjv_defaultOption(self.opt)
            self.cam.distance = 10
            self.cam.lookat[0] = 0
            self.cam.lookat[1] = 0
            self.cam.lookat[2] = 0
            self.scene = mj.MjvScene(self.model, maxgeom=10000)
            self.context = mj.MjrContext(self.model, mj.mjtFontScale.mjFONTSCALE_100)

            self.render_init = True

        self.viewport = mj.MjrRect(0, 0, 1200, 900)
        mj.mjv_updateScene(self.model, self.data, self.opt, None, self.cam, mj.mjtCatBit.mjCAT_ALL.value, self.scene)
        mj.mjr_render(self.viewport, self.scene, self.context)
        mj.glfw.glfw.swap_buffers(self.window)
        mj.glfw.glfw.poll_events()

    def close(self, action):
        mj.glfw.glfw.terminate()

    def seed(self, seed=None):
        pass


"""
env = FightingEnv()
env.reset()
for _ in range(3000):
    a1 = np.random.normal(size=(21))
    a2 = np.random.normal(size=(21))
    o, r, t, _ = env.step(a1, a2)
    env.render()
    if t:
        env.reset()
"""
