import sys, os
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))  
PARENT_DIR = os.path.dirname(CURRENT_DIR)                 
sys.path.append(PARENT_DIR)

import gymnasium as gym
import scipy.ndimage as nd
import numpy as np
import math
import cv2
import matplotlib.pyplot as plt
from loguru import logger                       
from collections import defaultdict
from collections import deque
from interactions.env_DS import Lidar
from interactions.env_DS import interface2RL
from interactions.attachments import param
from interactions.attachments import Render
from interactions.attachments import Planner
from interactions.attachments import Frontier
from gymnasium import spaces
from gymnasium.utils import seeding

class DM_env(gym.Env):
    def __init__(self, exec_mode = "common", obs_mode = "full", map_id_set = None, rank = 0):
        super(DM_env, self).__init__()

        # self.dem_id = param.dem_id
        self.map_id_set = map_id_set

        # seeding for multi-env
        self.rank = int(rank)
        self.episode_num = 0
        self.base_seed = 0

        self.exec_mode = exec_mode           # common, noplan
        self.obs_mode =  obs_mode             # full, no_uncertainty
        self.EXPLORE_WEIGHT = param.EXPLORE_GAIN_WEIGHT
        self.disable_u_reward = (self.obs_mode == "no_uncertainty")

        self.planner = Planner()
        self.render_obj = Render()

        self._seed = int(0)
        self.np_random, _ = seeding.np_random(self._seed)  

        # 参数
        self.micro_step = 0
        self.step_count_inepisode = 0
        self.timestep = 0
        self.max_steps = 1000
        self.replan_interval = 3                                    
        self.visit_count = defaultdict(int)
        self.global_map = np.zeros((param.global_size_height, param.global_size_width))
        self.reward = 0.0

        # vehicle state
        self.v = 0.0
        self.agent_x = 0.0
        self.agent_ix = 0
        self.agent_y = 0.0
        self.agent_iy = 0
        self.agent_yaw = 0.0
        self.goal_x = None
        self.goal_ix = None
        self.goal_y = None
        self.goal_iy = None
        self.goal_yaw = None
        self.goal_distance = None
        self.goal_angle = None
        self.global_mask = np.zeros((param.global_size_height, param.global_size_width))
        self.local_mask = np.zeros((param.local_size_height, param.local_size_width))
        # self.current_sub_goal_astar = (None, None)              # astar goal point
        self.path = deque()                                     # path without yaw

        # map old
        self.local_m = np.ones((param.local_size_height, param.local_size_width))
        self.local_m_uncertainty = np.ones((param.local_size_height, param.local_size_width))
        
        # map
        self.local_m_occ = np.zeros((param.local_size_height, param.local_size_width))              # DS证据理论
        self.local_m_free = np.zeros((param.local_size_height, param.local_size_width))             # DS证据理论
        self.local_m_unk = np.ones((param.local_size_height, param.local_size_width))               # DS证据理论
        self.m_occ = np.zeros((param.global_size_height, param.global_size_width))                  # DS证据理论
        self.m_free = np.zeros((param.global_size_height, param.global_size_width))                 # DS证据理论
        self.m_unk =np.zeros((param.global_size_height, param.global_size_width))                   # DS证据理论
        self.belief_map = defaultdict(lambda: {"occ": 0.0, "free": 0.0, "unk": 1.0})
        self.risk_map = np.zeros((param.local_size_height, param.local_size_width))                 # risk map: local

        # 评价指标
        self.episode_reward = 0.0
        self.episode_length = 0

        # 状态空间通道数
        self.global_map_channel = 4         
        self.local_map_channel = 4         
        self.pose_channel = 6                           

        # 定义状态空间：
        '''
        global_mask
        m_occ
        m_free
        m_unk
        local_mask
        local_m_occ
        local_m_free
        local_m_unk
        x, y, sin(yaw), cos(yaw), g_distance, g_angle
        '''
        self.observation_space = spaces.Dict({
            "global_map": spaces.Box(
                low=0.0,
                high=1.0,
                shape=(self.global_map_channel, 
                       param.global_size_height, 
                       param.global_size_width),  
                dtype=np.float32
            ),

            "local_map": spaces.Box(
                low=0.0,
                high=1.0,
                shape=(self.local_map_channel, 
                       param.local_size_height, 
                       param.local_size_width),  
                dtype=np.float32
            ),

            "pose": spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(self.pose_channel,), 
                dtype=np.float32
            )
        })

        # 定义动作空间：
        
        if self.exec_mode == "common":
            self._init_actionspace_common()
        elif self.exec_mode == "noplan":
            self._init_actionspace_noplan()
        else:
            raise ValueError(self.exec_mode)

    def reset(self, *, seed=None, options=None):
        '''
        -重置环境状态
        -返回值
        observation	    初始观察
        info	        额外信息（一般空字典）
        '''
        super().reset(seed=seed)
        if seed is not None:
            self.base_seed = int(seed)
            self.episode_num = 0
        
        ep_seed = self._make_episode_seed(self.episode_num)
        self.episode_num += 1

        self._seed = ep_seed
        self.np_random, _ = seeding.np_random(self._seed)

        self._reset_world_()

        # ---observation---
        if self.obs_mode == "full":
            observation = self._build_obs_common()
        elif self.obs_mode == "no_uncertainty":
            observation = self._build_obs_no_uncertainty()
        else:
            raise ValueError(self.obs_mode)
        
        # ---info---
        info = {
            "ep_seed": int(ep_seed)
        }

        # DEBUG
        # logger.debug("reach_weight = {}", param.REACH_GOAL_WEIGHT)

        return observation, info

    def step(self, action): 
        '''
        返回值
        observation	    observation	        下一时刻的观测
        reward	        reward	            这一步的奖励
        done	        terminated	        任务是否自然结束（成功/失败）
        False	        truncated	        是否被强制中断（超时等）
        info	        info	            额外信息（调试/分析用）
        '''
        self.timestep += 1
        self.reward = 0.0
        self.macro_reward = 0.0

        # ---forward---
        if self.exec_mode == "common":
            reach, collision = self._step_forward_common(action) 
        elif self.exec_mode == "noplan":     
            reach, collision = self._step_forward_noplan(action)
        else:
            raise ValueError(self.exec_mode)

        # ---observation---
        if self.obs_mode == "full":
            obs = self._build_obs_common()
        elif self.obs_mode == "no_uncertainty":
            obs = self._build_obs_no_uncertainty()
        else:
            raise ValueError(self.obs_mode)

        # ---输出info---
        self.episode_reward += self.macro_reward
        self.episode_length += 1

        terminated = bool(reach or collision)
        truncated = bool((not terminated) and (self.micro_step >= self.max_steps))


        if reach:
            done_reason = "reach"
        elif collision:
            done_reason = "collision"
        elif truncated:
            done_reason = "max step"
        else:
            done_reason = "running"

        info = {
            "is_success": bool(reach),
            "collision": bool(collision),
            "done_reason": done_reason,
        }

        info.update({
            "agent_ix": int(self.agent_ix),
            "agent_iy": int(self.agent_iy),
            "goal_ix": int(self.goal_ix),
            "goal_iy": int(self.goal_iy),
            "agent_x": float(self.agent_x),
            "agent_y": float(self.agent_y),
            "goal_x": float(self.goal_x),
            "goal_y": float(self.goal_y),
            "agent_yaw": float(self.agent_yaw),
            "goal_angle": float(self.goal_angle),
        })
        
        if terminated or truncated:
            info["episode"] = {
                "r": self.episode_reward,   
                "l": self.episode_length    
            }
            info["progress_map"] = {
                "m_occ": self.m_occ,
                "m_free": self.m_free,
                "m_unk": self.m_unk,
                "global_mask": self.global_mask,
            }

        
        return obs, self.macro_reward, terminated, truncated, info

    def render(self, mode='human'):
        '''
        渲染环境（可选）
        mode: 渲染模式
        'human'：渲染到屏幕
        'rgb_array'：返回RGB图像数组
        '''
        
        self.render_obj._draw_local_map(self.local_m, self.agent_x, self.agent_y, self.agent_yaw)
        self.render_obj._draw_local_map_uncertainty(self.local_m_unk, self.agent_x, self.agent_y, self.agent_yaw)
        self.render_obj._draw_belief_map(self.belief_map)
        self.render_obj.flush()
    
        plt.pause(0.10)   

    def _make_episode_seed(self, episode_num: int) -> int:
        return self.base_seed + self.rank * 100000 + int(episode_num)

    def _init_actionspace_common(self):
        '''
        alpha_size;
        alpha_dist;
        alpha_risk;
        '''
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0]), 
            high=np.array([1.0, 1.0, 1.0]), 
            shape=(3,),
            dtype=np.float32
        ) 

    def _init_actionspace_noplan(self):
        '''
        accelerate;
        steer;
        '''
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0]), 
            high=np.array([1.0, 1.0]), 
            shape=(2,),
            dtype=np.float32
        )

    def _reset_world_(self):
        self.disable_u_reward = (self.obs_mode == "no_uncertainty")
        self.episode_reward = 0.0
        self.episode_length = 0
        self.step_count_inepisode = 0
        self.micro_step = 0
        self.visit_count.clear()
        self.goal_x = None
        self.goal_y = None
        self.goal_yaw = None
        self.path = deque()
        self.global_mask.fill(0)
        self.local_mask.fill(0)
        self.macro_reward = 0.0

        # 生成地图
        # self.generate_random_map(seed = seed)         # 随机生成地图 
        # self.get_easy_map()                           # 生成简单测试地图
        self.map_sampling()                             # 从map_set中采样一个map,并用作环境map

        # 随机生成智能体和目标点位置
        self.goal_ix,self.goal_iy,self.goal_yaw = self.get_random_free_position()
        self.goal_y, self.goal_x = self.Transform_index_to_world(self.goal_ix, self.goal_iy, "global")
        self.agent_ix,self.agent_iy,self.agent_yaw = self.get_random_free_position()  
        self.agent_y, self.agent_x = self.Transform_index_to_world(self.agent_ix, self.agent_iy, "global")

        # 生成固定目标点位置（debug）
        # self.goal_ix, self.goal_iy, self.goal_yaw = 80, 230, 0.0
        # self.goal_y, self.goal_x = self.Transform_index_to_world(self.goal_ix, self.goal_iy, "global")
        # self.agent_ix, self.agent_iy, self.agent_yaw = 10, 230, 0.0
        # self.agent_y, self.agent_x = self.Transform_index_to_world(self.agent_ix, self.agent_iy, "global")

        logger.debug("World reset: agent at ({:.1f},{:.1f}), goal at ({:.1f},{:.1f}), distance {:.1f}", self.agent_x, self.agent_y, self.goal_x, self.goal_y, self.get_distance2goal())

        # 得到 position mask
        local_mask_ix = int(param.local_size_width // 2)
        local_mask_iy = int(param.local_size_height // 2)
        self.global_mask[self.agent_iy, self.agent_ix] += 1
        self.local_mask[local_mask_iy, local_mask_ix] = 1

        # 初始化创建接口对象
        self.interface = interface2RL(self.global_map, 
                                      [self.agent_ix, self.agent_iy, self.agent_yaw])

        # 获取初始观测值（map_uncertainty、map_occupancy）
        m_occ, m_free, m_unk, local_m, local_m_occ, local_m_free, local_m_unk, _ , self.belief_map = self.interface.ToSAC_reset()

        self.local_m_occ = local_m_occ
        self.local_m_free = local_m_free
        self.local_m_unk = local_m_unk
        self.local_m = local_m
        self.m_occ = m_occ
        self.m_free = m_free
        self.m_unk = m_unk

        self.goal_distance = self.get_distance2goal()
        self.goal_angle = self.get_angle2goal()

        self.risk_map = self.get_risk_map()

    def _build_obs_common(self):
        '''return: obs'''
        observation = {
            "global_map": np.stack([self.global_mask,
                                    self.m_occ,
                                    self.m_free,
                                    self.m_unk], axis=0).astype(np.float32),

            "local_map": np.stack([self.local_mask,
                                   self.local_m_occ,
                                   self.local_m_free,
                                   self.local_m_unk], axis=0).astype(np.float32),

            "pose": np.array([
                self.agent_ix / param.global_size_width,
                self.agent_iy / param.global_size_height,
                math.sin(self.agent_yaw),
                math.cos(self.agent_yaw),
                self.goal_distance / param.max_dist,
                self.goal_angle / math.pi
            ], dtype=np.float32)
        }
        return observation

    def _build_obs_no_uncertainty(self):
        '''
        return: obs   ---unk is 0
        '''

        m_unk = np.zeros_like(self.m_unk)
        local_m_unk = np.zeros_like(self.local_m_unk)

        observation = {
            "global_map": np.stack([self.global_mask,
                                    self.m_occ,
                                    self.m_free,
                                    m_unk], axis=0).astype(np.float32),

            "local_map": np.stack([self.local_mask,
                                   self.local_m_occ,
                                   self.local_m_free,
                                   local_m_unk], axis=0).astype(np.float32),

            "pose": np.array([
                self.agent_ix / param.global_size_width,
                self.agent_iy / param.global_size_height,
                math.sin(self.agent_yaw),
                math.cos(self.agent_yaw),
                self.goal_distance / param.max_dist,
                self.goal_angle / math.pi
            ], dtype=np.float32)
        }
        return observation

    def _step_forward_common(self, action):
        '''
        action is weight;
        return: reach, collision;
        '''
        self.step_count_inepisode += 1
        used_backup = False
        self.back_up_count = 0
        reach = False
        collision = False
        micro_executed = 0

        # ---action to weight---    
        # weight = self.action2weight_softmax(action)
        weight = np.clip(action, -1.0, 1.0) * 2.0
        # weight = np.exp(action)

        # 如果 goal 在 localmap 内
        goal_reachable, path2goal = self.goal_reachable()

        # ---计算 frontier , astar planning---

        if goal_reachable:
            # 直接规划到 goal
            self.path.clear()
            path_global = self.path_to_world(path2goal, self.agent_x, self.agent_y)
            self.path.extend(path_global)
        else:
            back_up = self.planning_strategy(weight)
            used_backup |= back_up
            self.back_up_count += int(back_up)
            self.reward -= 2.0

        # logger.debug("self.path:{}", self.path)

        # micro steps
        for i in range(self.replan_interval):
            self.micro_step += 1
            micro_executed +=1

            # ---获取 sub_goal---
            sub_goal = self.pop_sub_goal()
            # when path is used up
            if sub_goal is None:
                back_up = self.planning_strategy(weight)
                used_backup |= back_up
                self.back_up_count += int(back_up)
                sub_goal = self.pop_sub_goal()
                self.reward -= 2.0
                if sub_goal is None:
                    # logger.debug("No sub_goal obtained after backup policy!")
                    self.reward -= 10.0
                    break

            # last reward params
            _, _, uncertain_gain, risk = self.calculate_reward_param()

            # ---更新局部map与pose---
            sub_goal_yi, sub_goal_xi = self.Transform_world_to_index(sub_goal[0], sub_goal[1], "global")
            sub_goal_i = [sub_goal_xi, sub_goal_yi, sub_goal[2]]

            m_occ, m_free, m_unk, local_m, local_m_occ, local_m_free, local_m_unk, collision, current_pose, self.belief_map = self.interface.ToSAC_step(sub_goal_i)

            self.agent_y, self.agent_x = self.Transform_index_to_world(current_pose[0], current_pose[1], "global")
            self.agent_iy, self.agent_ix = self.Transform_world_to_index(self.agent_x, self.agent_y, "global")

            self.agent_yaw = current_pose[2]
            self.local_m_occ = local_m_occ
            self.local_m_free = local_m_free
            self.local_m_unk = local_m_unk
            self.local_m = local_m
            self.m_occ = m_occ
            self.m_free = m_free
            self.m_unk = m_unk

            # 得到 position mask
            self.global_mask[self.agent_iy, self.agent_ix] += 1
            
            self.goal_distance = self.get_distance2goal()
            self.goal_angle = self.get_angle2goal()

            # risk map update
            self.risk_map = self.get_risk_map()

            # ---reward---
            # reward params
            distance_new, _, uncertain_gain_new, risk_new = self.calculate_reward_param()
            uncertainty_err = -(uncertain_gain_new - uncertain_gain)
            risk_err = risk_new - risk
            reach = self.reach_goal()
            _ = self.revisit_penalty_func()

            # reward计算
            reward = self.calculate_reward(
                distance_new,
                collision,
                reach,
                self.micro_step,
                uncertainty_err,
                risk_err,
                self.visit_count
            )

            self.reward += reward
            
            #---terminated collsion reach---
            if (reach or collision): 
                break
            if self.micro_step >= self.max_steps:
                self.reward -= 10
                break

        self.macro_reward = self.reward / max(1, micro_executed)
            
        return reach, collision

    def _step_forward_noplan(self, action):
        '''
        action is acc, steer
        return: reach, collision
        '''
        self.step_count_inepisode += 1
        self.micro_step = self.step_count_inepisode
        reach = False
        collision = False

        # last reward params
        _, _, uncertain_gain, risk = self.calculate_reward_param()

        accel = param.RATIO_throttle * action[0]
        steer = param.RATIO_steer * action[1]
        self.v += param.dt * accel
        self.v = np.clip(self.v, -10.0, 10.0)

        desired_x = self.v * np.cos(self.agent_yaw) * param.dt + self.agent_x
        desired_y = self.v * np.sin(self.agent_yaw) * param.dt + self.agent_y
        desired_steer = self.v / param.L * np.tan(steer) * param.dt + self.agent_yaw

        sub_x = np.clip(desired_x, 0, (param.global_size_width - 1)*param.XY_RESO)
        sub_y = np.clip(desired_y, 0, (param.global_size_height - 1)*param.XY_RESO)
        sub_steer = desired_steer
        sub_goal = [sub_x, sub_y, sub_steer]
        sub_goal_yi, sub_goal_xi = self.Transform_world_to_index(sub_goal[0], sub_goal[1], "global")
        sub_goal_i = [sub_goal_xi, sub_goal_yi, sub_steer]

        m_occ, m_free, m_unk, local_m, local_m_occ, local_m_free, local_m_unk, collision, current_pose, self.belief_map = self.interface.ToSAC_step(sub_goal_i)

        self.agent_y, self.agent_x = self.Transform_index_to_world(current_pose[0], current_pose[1], "global")
        self.agent_iy, self.agent_ix = self.Transform_world_to_index(self.agent_x, self.agent_y, "global")

        self.agent_yaw = current_pose[2]
        self.local_m_occ = local_m_occ
        self.local_m_free = local_m_free
        self.local_m_unk = local_m_unk
        self.local_m = local_m
        self.m_occ = m_occ
        self.m_free = m_free
        self.m_unk = m_unk

        # 得到 position mask
        self.global_mask[self.agent_iy, self.agent_ix] += 1
            
        self.goal_distance = self.get_distance2goal()
        self.goal_angle = self.get_angle2goal()

        # risk map update
        self.risk_map = self.get_risk_map()

        # ---reward---
        # reward params
        distance_new, _, uncertain_gain_new, risk_new = self.calculate_reward_param()
        uncertainty_err = -(uncertain_gain_new - uncertain_gain)
        risk_err = risk_new - risk
        reach = self.reach_goal()
        _ = self.revisit_penalty_func()

        # reward计算
        self.reward = self.calculate_reward(
            distance_new,
            collision,
            reach,
            self.micro_step,
            uncertainty_err,
            risk_err,
            self.visit_count)
        
        if self.step_count_inepisode >= self.max_steps:
            self.reward -= 10

        self.macro_reward = self.reward
        
        return reach, collision

    def read_map(self, dem_id = None):
        '''
        
        '''
        dem_id = dem_id

        DATASET_DIR = os.path.join(PARENT_DIR, "Dataset", "output", "map", "hard_obstacle" )
        npy_path = os.path.join(DATASET_DIR, f"tile_{dem_id}.npy")

        if not os.path.exists(npy_path):
            raise FileNotFoundError(f"DEM npy not found: {npy_path}")

        tile = np.load(npy_path, mmap_mode="r")

        lo = np.percentile(tile, 2)
        hi = np.percentile(tile, 98)
        tile_clip = np.clip(tile, lo, hi)

        m01 = (tile_clip - lo) / (hi - lo + 1e-6)   # 0~1 float32

        self.global_map = m01

    def map_sampling(self):
        '''
        从map_set中采样一个map,并用作环境map
        '''
        map_ids = self.map_id_set
        map_sampled_id_random = np.random.choice(map_ids)
        map_sampled_id_sequence = map_ids[self.episode_num % len(map_ids)]
        map_sampled_id = map_sampled_id_sequence
        self.read_map(map_sampled_id)

    def calculate_reward(self, distance_togoal, collision, reach, step, uncertainty_err, risk_err, visit_count):
        distance_reward = param.DISTANCE_WEIGHT * (1.0 - ((distance_togoal / param.max_dist) ** 0.4))
        collision_penalty = param.COLLISION_PENALTY if collision else 0.0
        step_penalty = (step / self.max_steps) * param.STEP_PENALTY
        reach_reward = param.REACH_GOAL_WEIGHT if reach else 0.0
        explore_reward = 0.0 if self.disable_u_reward else self.EXPLORE_WEIGHT * (uncertainty_err)
        risk_penalty = param.RISK_PENALTY * risk_err

        ix, iy = int(self.agent_ix), int(self.agent_iy)
        revisit_ratio = 1 / np.log(visit_count[(ix, iy)] + np.e)
        
        reward = revisit_ratio * (
            distance_reward +
            reach_reward +
            explore_reward
        ) + collision_penalty + step_penalty + risk_penalty

        return reward

    def get_risk_map(self, beta_occ = 1.0, beta_unk = 0.5, occ_threshold = 0.6,
                     safe_radius = 3.0,
                     w_inflate = 1.0,
                     mapping = "linear"):
        '''
        return: risk map
        '''
        r = beta_occ * self.local_m_occ + beta_unk * self.local_m_unk

        occ_mask = (self.local_m_occ > occ_threshold).astype(np.uint8)
        # logger.debug("local_m_occ{}", self.local_m_occ)

        free_mask = (1 - occ_mask).astype(np.uint8)

        # 每个格到最近障碍的距离（单位：格）
        dist = cv2.distanceTransform(free_mask, distanceType=cv2.DIST_L2, maskSize=5).astype(np.float32)

        if mapping == "linear":
            # dist=0 -> 1, dist>=R -> 0
            r_inflate = np.clip((safe_radius - dist) / (safe_radius + 1e-6), 0.0, 1.0)
        elif mapping == "exp":
            # exp(-d/sigma)：d=0 -> 1, d大 -> 0
            sigma = max(1.0, safe_radius / 2.0)
            r_inflate = np.exp(-dist / sigma).astype(np.float32)
        else:
            raise ValueError("mapping must be 'linear' or 'exp'")

        risk_map = w_inflate * r_inflate + r
        risk_map = np.clip(risk_map, 0.0, 1.0)
        
        return risk_map

    def get_frontier_clusters(self, weight, risk_map=None): 
        '''
        获取 top k frontier clusters
        '''
        local_goal_iy, local_goal_ix = self.Transform_World_global_to_local(self.goal_ix, self.goal_iy, "index")
        frontier = Frontier(self.local_m_occ, self.local_m_unk, 
                            self.local_m_free,(param.local_size_height // 2, param.local_size_width // 2)
                            , weight, (local_goal_ix, local_goal_iy), k=param.frontier_k)
        frontier_mask = frontier.compute_frontier_mask()
        clusters = frontier.cluster_frontiers(frontier_mask)
        new_clusters = frontier.splite_large_clusters(clusters)
        infos = frontier.summarize_clusters_rep_points(new_clusters)
        topk = frontier.select_topk(infos, risk_map) 
        return topk, frontier_mask

    def goal_reachable(self):
        goal_local_iy, goal_local_ix = self.Transform_World_global_to_local(self.goal_ix, self.goal_iy, "index")

        if goal_local_ix < 0 or goal_local_ix >= param.local_size_width or \
           goal_local_iy < 0 or goal_local_iy >= param.local_size_height:
            return False, None
        path = self.get_path_astar(self.local_m, (goal_local_iy, goal_local_ix))
        if path is None or len(path) == 0:
            return False, None
        else:
            return True, path

    def action2weight_softmax(self, action, temperature = 6.0):
        action = np.clip(action, -1.0, 1.0)
        logits = action / temperature
        weight = np.exp(logits - logits.max())
        weight = weight / (weight.sum() + 1e-8)
        
        return weight

    def planning_strategy(self, weight):
        '''
        return nopath: bool
        '''
        self.path.clear()
        topk, _ = self.get_frontier_clusters(weight, self.risk_map)
        best = topk[0] if topk and topk[0] is not None else None
        current_sub_goal_astar = best.get("rep_ij") if best else None     

        # logger.debug("current_sub_goal_astar:{}", current_sub_goal_astar)

        if current_sub_goal_astar is None:
            # logger.debug("No frontier found!")
            path_global = self.back_up_poloicy()
            # logger.debug("path_global backup:{}", path_global)
            
            self.path.clear()
            self.path.extend(path_global)
            return True

        path_local = self.get_path_astar(self.local_m, current_sub_goal_astar)
        # logger.debug("path_local frontier:{}", path_local)
        path_global = self.path_to_world(path_local, self.agent_x, self.agent_y)
        # logger.debug("path_global frontier:{}", path_global)

        self.path.extend(path_global)

        if not path_local:
            # logger.debug("can not obtain path from astar!")
            path_global = self.back_up_poloicy()
            # logger.debug("path_global backup:{}", path_global)
            self.path.clear()
            self.path.extend(path_global)
            return True
        
        return False
            
    def pop_sub_goal(self):
        '''
        返回 sub_goal = [x, y, yaw]
        如果 path 不足，返回 None（触发重规划或 fallback）
        '''
        if len(self.path) == 0:
            return None
        
        if len(self.path) == 1:
            p_ = self.path.popleft()
            sub_y, sub_x = p_
            sub_yaw = self.agent_yaw
            sub_goal = [sub_x, sub_y, sub_yaw]
            return sub_goal
        else:
            p_ = self.path.popleft()
            p_next = self.path[0]
            sub_y, sub_x = p_
            sub_y_next, sub_x_next = p_next
            sub_yaw = math.atan2((sub_y_next - sub_y),(sub_x_next - sub_x))
            sub_goal = [sub_x, sub_y, sub_yaw]
            return sub_goal

    def back_up_poloicy(self):
        H, W = param.local_size_height, param.local_size_width
        ci, cj = H // 2, W // 2
        res = param.XY_RESO

        # goal 在 local 坐标系中的偏移（格）
        dx = (self.goal_x - self.agent_x) / res
        dy = (self.goal_y - self.agent_y) / res

        # local 索引 (y, x)
        y = int(round(ci - dy))
        x = int(round(cj + dx))

        # clamp 到 local 边界
        y = int(np.clip(y, 0, H - 1))
        x = int(np.clip(x, 0, W - 1))

        current_sub_goal_ = (y, x)
        path_local = self.get_path_astar(self.local_m, current_sub_goal_)
        path_global = self.path_to_world(path_local, self.agent_x, self.agent_y)

        return path_global

    # def point_global_to_local(self, x_in, y_in):
    #     '''
    #     return local 索引 (y, x)
    #     '''
    #     H, W = param.local_size_height, param.local_size_width
    #     ci, cj = H // 2, W // 2
    #     res = param.XY_RESO

    #     # goal 在 local 坐标系中的偏移（格）
    #     dx = (x_in - self.agent_x) / res
    #     dy = (y_in - self.agent_y) / res

    #     # local 坐标 (y, x)
    #     y = int(round(ci + dy))
    #     x = int(round(cj + dx))

    #     return (y, x)

    def path_to_world(self, path_local, agent_x, agent_y):
        if path_local is None or len(path_local)==0:
            return []
        H, W = param.local_size_height, param.local_size_width
        ci, cj = H // 2, W // 2
        res = param.XY_RESO
        
        path_global = []
        for (y, x) in path_local:   # y, x
            xg = agent_x + (x - cj) * res
            yg = agent_y + (ci - y) * res
            # 
            xg = np.clip(xg, 0, (param.global_size_width - 1)*param.XY_RESO)
            yg = np.clip(yg, 0, (param.global_size_height - 1)*param.XY_RESO)
            path_global.append((yg, xg))

        return path_global

    # Astar interface
    def get_path_astar(self, local_map, goal):
        path = self.planner.main_workflow(local_map, goal)
        return path

    def calculate_reward_param(self):

        # distance and yaw
        distance_MH = abs(self.agent_x - self.goal_x) + abs(self.agent_y - self.goal_y)
        distance_Euclidean = math.sqrt((self.agent_x - self.goal_x)**2 + (self.agent_y - self.goal_y)**2)
        yaw2goal = self.get_angle2goal()
        angle_yaw_goal = yaw2goal - self.agent_yaw
        angle_yaw_goal = (angle_yaw_goal + math.pi) % (2 * math.pi) - math.pi

        # 不确定性增益
        uncertain_gain = np.mean(self.local_m_unk)
        # uncertain_gain = np.max(self.local_m_unk)

        # risk gain
        ci = param.local_size_height // 2
        cj = param.local_size_width // 2
        risk_gain = self.local_m_occ[ci, cj] + self.local_m_unk[ci, cj] 
        
        return distance_Euclidean, angle_yaw_goal, uncertain_gain, risk_gain
        
    def revisit_penalty_func(self):
        ix, iy = int(self.agent_ix), int(self.agent_iy)
        c = self.visit_count[(ix, iy)]
        visit_penalty = math.log1p(c) 
        self.visit_count[(ix, iy)] = c + 1
        return visit_penalty

    def get_easy_map(self, width=param.global_size_width, height=param.global_size_height):
        map_ = np.zeros((height, width))

        # 1. 四周边框
        map_[0, :] = 1
        map_[-1, :] = 1
        map_[:, 0] = 1
        map_[:, -1] = 1

        # 2. 中间横墙，高 8 像素，留左右两个大缺口
        wall_y = height // 2
        gap = width // 4          # 每个缺口宽度
        left_gap_start = gap
        right_gap_start = width - gap

        # 横墙主体
        map_[wall_y - 4 : wall_y + 4, left_gap_start + gap : right_gap_start] = 1
        self.global_map = map_

    def get_random_free_position(self):
        '''
        生成 RL 可学习的随机出生点：保证不贴墙、车头朝向大致目标、距离合适
        '''
        free_thr = 0.35   
        obs_thr  = 0.7

        min_dist_to_obs = 5
        min_goal_dist = 64
        max_goal_dist = 180
        free_cells = np.argwhere(self.global_map < free_thr)

        gm = self.global_map
        # print("[DBG] dem_id=", getattr(self, "dem_id", None),
        #     "shape=", gm.shape, "dtype=", gm.dtype,
        #     "min/max/mean=", float(np.nanmin(gm)), float(np.nanmax(gm)), float(np.nanmean(gm)),
        #     "nan%=", float(np.isnan(gm).mean()),
        #     "free_thr=", free_thr,
        #     "count(gm<free_thr)=", int(np.sum(gm < free_thr)),
        #     "count(gm==0)=", int(np.sum(gm == 0)))

        max_attempts = 300
        for _ in range(max_attempts):
            # 随机取一个 free cell（确保不是边界）
            y, x = free_cells[self.np_random.integers(0, len(free_cells))]

            # 边界过滤
            if x < 5 or y < 5 or x > param.global_size_width - 5 or y > param.global_size_height - 5:
                continue

            # 保证离障碍物有安全距离
            # 使用局部窗口检查障碍
            xmin = max(0, x - min_dist_to_obs)
            xmax = min(param.global_size_width, x + min_dist_to_obs)
            ymin = max(0, y - min_dist_to_obs)
            ymax = min(param.global_size_height, y + min_dist_to_obs)

            if np.any(self.global_map[ymin:ymax, xmin:xmax] >= obs_thr):
                continue

            # 如果这是 goal 的生成逻辑，需要只返回位置
            if self.goal_x is None:
                yaw = 0.0
                return x, y, yaw

            # 保证起点与目标距离合理
            dx = self.goal_x - x
            dy = self.goal_y - y
            dist = math.sqrt(dx*dx + dy*dy)
            if dist < min_goal_dist or dist > max_goal_dist:
                continue

            # Yaw 朝向目标 ±30°
            yaw2goal = math.atan2(dy, dx)
            yaw = yaw2goal + self.np_random.uniform(-math.pi/6, math.pi/6)

            return x, y, yaw

        # 如果实在找不到，就返回一个最安全的点
        return free_cells[0][1], free_cells[0][0], 0.0

    def reach_goal(self):
        distance = math.sqrt((self.agent_x - self.goal_x)**2 + (self.agent_y - self.goal_y)**2)
        yaw_diff = abs(self.agent_yaw - self.goal_yaw)
        yaw_diff = min(yaw_diff, 2*math.pi - yaw_diff)

        if distance < 5.0: # and yaw_diff < (30.0 * math.pi / 180.0):
            return True
        else:
            return False

    def get_distance2goal(self):
        dx = self.goal_ix - self.agent_ix
        dy = self.goal_iy - self.agent_iy
        goal_distance = math.sqrt(dx*dx + dy*dy)
        
        return goal_distance
    
    def get_angle2goal(self):
        goal_angle = math.atan2(self.goal_iy - self.agent_iy, self.goal_ix - self.agent_ix)
        # 归一化到 [-pi, pi]
        goal_angle = (goal_angle + math.pi) % (2 * math.pi) - math.pi
        
        return goal_angle

    # 坐标变换接口
    def Transform_World_global_to_local(self, x_global, y_global, xy_format="index"):
        '''
        global坐标转local坐标
        返回 local index
        '''
        H, W = param.local_size_height, param.local_size_width
        ci, cj = H // 2, W // 2

        if xy_format == "index":
            dx = (x_global - self.agent_ix)
            dy = (y_global - self.agent_iy)
        elif xy_format == "world":
            logger.error("remain some bug here, be careful when using world format!")
            dx = (x_global - self.agent_x) / param.XY_RESO
            dy = (y_global - self.agent_y) / param.XY_RESO
        else:
            raise ValueError("xy_format must be 'index' or 'world'")

        y_local = int(round(ci + dy))
        x_local = int(round(cj + dx))

        return y_local, x_local
    
    # def Transform_World_local_to_global(self, x_local, y_local, xy_format="index"):
    #     '''
    #     local坐标转global坐标
    #     return global index or world coordinate
    #     '''
    #     H, W = param.local_size_height, param.local_size_width
    #     ci, cj = H // 2, W // 2

    #     if xy_format == "index":
    #         dx = (x_local - cj)
    #         dy = (y_local - ci)
    #     elif xy_format == "world":
    #         dx = (x_local - cj) * param.XY_RESO
    #         dy = (y_local - ci) * param.XY_RESO
    #     else:
    #         raise ValueError("xy_format must be 'index' or 'world'")

    #     x_global = self.agent_x + dx
    #     y_global = self.agent_y + dy

    #     return y_global, x_global
    
    def Transform_world_to_index(self, x_world, y_world, map_format="local"):
        '''
        world坐标转index坐标
        '''
        
        res = param.XY_RESO

        if map_format == "local":
            H, W = param.local_size_height, param.local_size_width
            ci, cj = H // 2, W // 2
            dx = (x_world - self.agent_x) / res
            dy = (y_world - self.agent_y) / res
            ix = int(round(cj + dx))
            iy = int(round(ci - dy))
        elif map_format == "global":
            H, W = param.global_size_height, param.global_size_width
            ix = int(round(x_world / res))
            iy = int((H-1) - round(y_world / res))
        else:
            raise ValueError("map_format must be 'local' or 'global'")

        return iy, ix
    
    def Transform_index_to_world(self, ix, iy, map_format="local"):
        '''
        index坐标转world坐标
        '''
        
        res = param.XY_RESO

        if map_format == "local":
            H, W = param.local_size_height, param.local_size_width
            ci, cj = H // 2, W // 2
            x_world = self.agent_x + (ix - cj) * res
            y_world = self.agent_y + (ci - iy) * res
        
        elif map_format == "global":
            H, W = param.global_size_height, param.global_size_width
            x_world = ix * res
            y_world = (H - 1 - iy) * res
        else:
            raise ValueError("map_format must be 'local' or 'global'")

        return y_world, x_world

if __name__ == "__main__":
    '''
    fmt's code
    '''
    pass
   

