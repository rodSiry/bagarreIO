import mujoco as mj
import time
import pybullet as p
import pybullet_data
import random
import numpy as np
import gym
         
# hit scale : scaling factor for punching reward

class FightingEnv(gym.Env):
    def __init__(self, hit_scale=0.001):
        self.model = mj.MjModel.from_xml_path('bagarreIO/assets/humanoids.xml')
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

        self.hit_scale = hit_scale

        self.dist = np.sum((self.data.body('torso').xpos - self.data.body('2torso').xpos) ** 2) ** 0.5
        self.render_init = False

        #legal player1 hitting parts
        p1_hit = ['hand_left', 'hand_right', 'foot1_left', 'foot2_left', 'foot1_right', 'foot2_right']
        self.p1_hit = [self.data.geom(p).id for p in p1_hit]

        #legal player2 hitting parts
        p2_hit = ['2hand_left', '2hand_right', '2foot1_left', '2foot2_left', '2foot1_right', '2foot2_right']
        self.p2_hit = [self.data.geom(p).id for p in p2_hit]

        #legal player1 target parts
        p1_targets = ['2torso', '2head', '2butt', '2waist_upper', '2waist_lower', '2foot1_right', '2foot2_right', '2foot1_left', '2foot2_left', '2upper_arm_right', '2upper_arm_left', '2lower_arm_right', '2lower_arm_left', '2hand_left', '2hand_right', '2shin_left', '2shin_right', '2thigh_left', '2thigh_right']
        self.p1_targets = [self.data.geom(p).id for p in p1_targets]

        #legal player2 target parts
        p2_targets = ['torso', 'head', 'butt', 'waist_upper', 'waist_lower', 'foot1_right', 'foot2_right', 'foot1_left', 'foot2_left', 'upper_arm_right', 'upper_arm_left', 'lower_arm_right', 'lower_arm_left', 'hand_left', 'hand_right', 'shin_left', 'shin_right', 'thigh_left', 'thigh_right']
        self.p2_targets = [self.data.geom(p).id for p in p2_targets]

        #non-target parts
        ['wall1', 'wall2', 'wall3', 'wall4', 'floor']

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

        rc1, rc2 = self.get_contacts_rewards()

        if self.timeCount == 1000:
            end = 1

        return np.concatenate([o1, o2], 0), r, end, None

    def get_contacts_rewards(self):

        #cumulated hit rewards
        r1, r2 = 0, 0


        #parse detected collisions
        for k in range(len(self.data.contact)):

            #if p1 hits p2
            if (self.data.contact[k].geom1 in self.p1_hit and self.data.contact[k].geom2 in self.p1_targets) or (self.data.contact[k].geom2 in self.p1_hit and self.data.contact[k].geom1 in self.p1_targets):
                force_index = self.data.contact[k].efc_address
                if force_index >= 0:
                    force = self.data.efc_force[force_index]
                    r1 += force
                    r2 -= force

            #if p2 hits p1
            if (self.data.contact[k].geom1 in self.p2_hit and self.data.contact[k].geom2 in self.p2_targets) or (self.data.contact[k].geom2 in self.p2_hit and self.data.contact[k].geom1 in self.p2_targets):
                force_index = self.data.contact[k].efc_address
                if force_index >= 0:
                    force = self.data.efc_force[force_index]
                    r2 += force
                    r1 -= force

        r1, r2 = r1 * self.hit_scale, r2 * self.hit_scale
        return r1, r2

    def get_observation(self):
        obs1 = np.concatenate([self.data.qpos[:len(self.data.qpos)//2] , self.data.qvel[:len(self.data.qvel)//2], self.data.body('torso').xpos, self.data.body('torso').xquat], axis=0)
        obs2 = np.concatenate([self.data.qpos[len(self.data.qpos)//2:] , self.data.qvel[len(self.data.qvel)//2:], self.data.body('2torso').xpos, self.data.body('2torso').xquat], axis=0)
        return obs1, obs2

    def reset(self):
        self.model = mj.MjModel.from_xml_path('bagarreIO/assets/humanoids.xml')
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
            self.scene = mj.MjvScene(self.model, maxgeom=10000)
            self.context = mj.MjrContext(self.model, mj.mjtFontScale.mjFONTSCALE_100)

            self.render_init = True

        poi = (self.data.body('torso').xpos + self.data.body('2torso').xpos) * .5
        inter_distance = (((self.data.body('torso').xpos - self.data.body('2torso').xpos) ** 2).sum()) ** 0.5
        self.cam.distance = max(10 + 0.1 * (inter_distance - 6), 2)

        poi[2] = 0
        self.cam.lookat = self.cam.lookat * 0.99 + 0.01 * poi
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
