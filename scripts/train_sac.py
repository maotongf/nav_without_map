import os, sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(ROOT_DIR)

# MODEL_PATH = os.path.join(ROOT_DIR, "wandb_models", "model.zip")
# RB_PATH = os.path.join(ROOT_DIR, "wandb_models", "buffer.pkl")
print("PYTHONPATH updated:", ROOT_DIR)

import multiprocessing as mp
import gymnasium as gym
import utils.register_env as register_env
import wandb
import time
import numpy as np
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3 import SAC
from stable_baselines3 import TD3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CallbackList, EvalCallback
from stable_baselines3.common.monitor import Monitor
from interactions.custom_encoder import CustomEncoder
from wandb.integration.sb3 import WandbCallback
from utils.wandb_callback import WandbCustomCallback
from loguru import logger
from interactions.attachments import param

def train():
    run = wandb.init(
        project="fmt-without_prior_map-dm",
        # name=run_name,
        sync_tensorboard=True,
        dir="./",
        config={
            "master_seed": 0,
            "exec_mode": "common",
            "obs_mode": "full",

            "map_id_set": ["00000"],

            "learning_rate": 3e-4,
            "buffer_size": 50000,
            "batch_size": 512,
            "gamma": 0.99,
            "tau": 0.005,
            "total_timesteps": 50_0000,
            "n_envs": 8,
            "gradient_steps": 4,
            "train_freq": 1,
            "learning_starts": 5000,
            "features_dim": 256,

            "reach_w": 10.0,
            "collision_penalty": -10.0,
            "step_penalty": -1.0,
            "distance_w": 5.0,
            "explore_gain_w": 0.5,
            "risk_penalty": -0.8,
            "nopath_penalty": -3.0,
        }
    )

    cfg = wandb.config

    logger.info(
        f"cfg.buffer_size={cfg.buffer_size}, cfg.batch_size={cfg.batch_size}, "
        f"cfg.n_envs={cfg.n_envs}, cfg.learning_rate={cfg.learning_rate}"
    )
    logger.info(f"cfg.total_timesteps={cfg.total_timesteps}, cfg.features_dim={cfg.features_dim}")


    run_name = (
        f"fmt"
        f"_run_id_{wandb.run.id}_"
        # f"_dem{cfg.map_id_set}_"
        f"_seed{cfg.master_seed}"
        f"_env{cfg.n_envs}"
        f"_{time.strftime('%m%d-%H%M')}"
    )
    wandb.run.name = run_name

    EXEC_MODE = str(cfg.exec_mode)
    OBS_MODE = str(cfg.obs_mode)
    MASTER_SEED = int(cfg.master_seed)

    # ---dir----
    sweep_dir = os.path.join(ROOT_DIR, "sweep_runs")
    run_dir = os.path.join(sweep_dir, "_run_", wandb.run.id)
    os.makedirs(run_dir, exist_ok=True)

    MODEL_PATH = os.path.join(ROOT_DIR, "WandB_models", "model.zip")

    # ---create multi-env---
    def make_env(rank: int):
        '''
        return: init()
        '''
        def _init():
            import gymnasium as gym    
            import utils.register_env
            from stable_baselines3.common.monitor import Monitor
            from interactions.attachments import param

            param.REACH_GOAL_WEIGHT = float(env_cfg["reach_w"])
            param.COLLISION_PENALTY = float(env_cfg["collision_penalty"])
            param.STEP_PENALTY = float(env_cfg["step_penalty"])
            param.DISTANCE_WEIGHT = float(env_cfg["distance_w"])
            param.EXPLORE_GAIN_WEIGHT = float(env_cfg["explore_gain_w"])
            param.RISK_PENALTY = float(env_cfg["risk_penalty"])
            param.NOPATH_PENALTY = float(env_cfg["nopath_penalty"])
            
            param.map_id_set = env_cfg["map_id_set"]
            # env.unwrapped.EXPLORE_GAIN_WEIGHT = float(env_cfg["explore_gain_w"])

            env = gym.make("DM-v1", exec_mode = EXEC_MODE, obs_mode = OBS_MODE, map_id_set=env_cfg["map_id_set"], rank = rank)
            env = Monitor(env)

            return env
        return _init
    
    env_cfg = {
        "exec_mode": EXEC_MODE,
        "obs_mode": OBS_MODE,

        "map_id_set": list(cfg.map_id_set),

        "reach_w": cfg.reach_w,
        "collision_penalty": cfg.collision_penalty,
        "step_penalty": cfg.step_penalty,
        "distance_w": cfg.distance_w,
        "explore_gain_w": cfg.explore_gain_w,
        "risk_penalty": cfg.risk_penalty,
        "nopath_penalty": cfg.nopath_penalty,
    }

    env = SubprocVecEnv([make_env(i) for i in range(cfg.n_envs)], start_method="forkserver")
    env.seed(MASTER_SEED)

    # ---model create/read---
    policy_kwargs = dict(
        features_extractor_class=CustomEncoder,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=dict(pi=[256, 256], qf=[256, 256])
        )
        
    if os.path.exists(MODEL_PATH):
        logger.info(f"Loading weights from: {MODEL_PATH}")
        model = SAC.load(MODEL_PATH, env=env, device="cuda")            
        # model = TD3.load(MODEL_PATH, env=env, device="cuda")           
        
    else:
        model = SAC(
            "MultiInputPolicy",
            env,
            device="cuda",
            learning_rate=float(cfg.learning_rate),
            buffer_size=int(cfg.buffer_size),
            batch_size=int(cfg.batch_size),
            tau=float(cfg.tau),
            gamma=float(cfg.gamma),
            train_freq=(int(cfg.train_freq), "step"),
            gradient_steps=int(cfg.gradient_steps),
            ent_coef="auto",
            target_entropy="auto",
            policy_kwargs=policy_kwargs,
            verbose=2,
            tensorboard_log="./tb_logs/",
            learning_starts=int(cfg.learning_starts),
        )


        # n_actions = env.action_space.shape[-1]
        # action_noise = NormalActionNoise(
        #     mean=np.zeros(n_actions),
        #     sigma=0.1 * np.ones(n_actions)
        # )

        # model = TD3(
        #     "MultiInputPolicy",
        #     env,
        #     device="cuda",
        #     learning_rate=float(cfg.learning_rate),
        #     buffer_size=int(cfg.buffer_size),
        #     learning_starts=int(cfg.learning_starts),
        #     batch_size=int(cfg.batch_size),
        #     tau=float(cfg.tau),
        #     gamma=float(cfg.gamma),
        #     train_freq=(int(cfg.train_freq), "step"),
        #     gradient_steps=int(cfg.gradient_steps),
        #     action_noise=None,
        #     replay_buffer_class=None,                   # 使用默认的ReplayBuffer, need promote
        #     policy_kwargs=policy_kwargs,
        #     verbose=2,
        #     tensorboard_log="./tb_logs/",
        # )
    
    # ---callback---
    param.REACH_GOAL_WEIGHT = float(env_cfg["reach_w"])
    param.COLLISION_PENALTY = float(env_cfg["collision_penalty"])
    param.STEP_PENALTY = float(env_cfg["step_penalty"])
    param.DISTANCE_WEIGHT = float(env_cfg["distance_w"])
    param.EXPLORE_GAIN_WEIGHT = float(env_cfg["explore_gain_w"])
    param.RISK_PENALTY = float(env_cfg["risk_penalty"])
    param.NOPATH_PENALTY = float(env_cfg["nopath_penalty"])

    # param.map_id_set = eval(env_cfg["map_id_set"])
    n_envs = env.num_envs
    eval_freq = max(5000 // n_envs, 1)
    best_dir = os.path.join(run_dir, "best_model")
    eval_dir  = os.path.join(run_dir, "eval")
    os.makedirs(best_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)

    eval_env = gym.make("DM-v1", exec_mode = EXEC_MODE, obs_mode = OBS_MODE, map_id_set=env_cfg["map_id_set"])
    eval_env = Monitor(eval_env)
    _ = eval_env.reset(seed=MASTER_SEED)

    custom_cb = WandbCustomCallback(save_freq=0, 
                                    log_freq=5000)
    
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=best_dir,  # 只保存 best
        log_path=eval_dir,
        eval_freq=eval_freq,
        n_eval_episodes=100,
        deterministic=True,
    )

    # ---learning---
    callback = CallbackList([custom_cb, eval_cb])

    model.learn(
        total_timesteps=int(cfg.total_timesteps),
        log_interval=4,
        callback=callback
    )

    # ---save---
    # model.save(MODEL_PATH)
    # model.save_replay_buffer(RB_PATH)
    env.close()
    run.finish()

if __name__ == "__main__":
    train()
