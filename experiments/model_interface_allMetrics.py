

import os
import sys
import logging
logging.basicConfig(level=logging.DEBUG)

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import simpler_env
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
import sapien.core as sapien
import time
from pathlib import Path
import uncertainty.utils as uncerUtils
import torch

from uncertainty.token_based_metrics_fast import TokenMetricsFast
import uncertainty.uncertainty_metrics as uncerMetrics


SEED = 2024
TIME_RANGE = 8

PACKAGE_DIR = Path(__file__).parent.resolve()

RT_1_CHECKPOINTS = {
    "rt_1_x": "rt_1_x_tf_trained_for_002272480_step",
    "rt_1_400k": "rt_1_tf_trained_for_000400120",
    "rt_1_58k": "rt_1_tf_trained_for_000058240",
    "rt_1_1k": "rt_1_tf_trained_for_000001120",
}

TASKS = [
    "google_robot_pick_customizable",
    "google_robot_pick_customizable_ycb",
    "google_robot_pick_customizable_no_overlay",
    "google_robot_move_near_customizable",
    "google_robot_move_near_customizable_ycb",
    "google_robot_move_near_customizable_no_overlay",
    "widowx_put_on_customizable",
    "widowx_put_on_customizable_ycb",
    "widowx_put_on_customizable_no_overlay",
    "widowx_put_in_customizable",
    "widowx_put_in_customizable_ycb",
    "widowx_put_in_customizable_no_overlay",
]

ckpt_dir = str(PACKAGE_DIR) + '/../checkpoints'

sapien.render_config.rt_use_denoiser = True


class VLAInterface:
    def __init__(self, task, model_name):

        self.model_name = model_name
        if task in TASKS:
            self.task = task
        else:
            raise ValueError(task)
        if "google" in self.task:
            self.policy_setup = "google_robot"
        else:
            self.policy_setup = "widowx_bridge"
        if "openvla" in model_name:
            from simpler_env.policies.openvla.openvla_model import OpenVLAInference
            self.model = OpenVLAInference(model_type='../checkpoints/openvla-7b', policy_setup=self.policy_setup)
            #self.followUpModel = OpenVLAInference(model_type=model_name, policy_setup=self.policy_setup)
           # self.variability_models=[OpenVLAInference(model_type='../checkpoints/openvla-7b' , policy_setup=self.policy_setup) for i in range(0,uncerMetrics.VARIABILITY)]
            
        elif "pi0" in model_name:
            from simpler_env.policies.lerobotpi.pi0_or_fast import LerobotPiFastInference
            if self.policy_setup == "widowx_bridge":
                model_path = "../checkpoints/lerobot-pi0-bridge"
            else:
                model_path = "../checkpoints/lerobot-pi0-fractal"

            self.model = LerobotPiFastInference(saved_model_path=model_path, policy_setup=self.policy_setup)
           # self.variability_models=[LerobotPiFastInference(saved_model_path=model_path, policy_setup=self.policy_setup) for i in range(0,uncerMetrics.VARIABILITY)]
            #self.followUpModel = LerobotPiFastInference(saved_model_path=model_path, policy_setup=self.policy_setup)
        elif "spatialvla" in model_name:
            from simpler_env.policies.spatialvla.spatialvla_model import SpatialVLAInference

            self.model = SpatialVLAInference(saved_model_path="../checkpoints/spatialvla-4b-mix-224-pt",policy_setup=self.policy_setup)
           # self.variability_models=[SpatialVLAInference(saved_model_path="../checkpoints/spatialvla-4b-mix-224-pt",policy_setup=self.policy_setup) for i in range(0,uncerMetrics.VARIABILITY)]
            
            #self.followUpModel = SpatialVLAInference(model_type=model_name, policy_setup=self.policy_setup)
        elif "gr00t" in model_name:
                from simpler_env.policies.gr00t.gr00t_model import Gr00tInference
                if self.policy_setup == "widowx_bridge":
                    model_path = "../checkpoints/gr00t-n1.5-bridge-posttrain"
                else:
                    model_path = "../checkpoints/gr00t-n1.5-fractal-posttrain"
                self.model = Gr00tInference(saved_model_path=model_path, policy_setup=self.policy_setup)
               # self.variability_models=[Gr00tInference(saved_model_path=model_path, policy_setup=self.policy_setup) for i in range(0,uncerMetrics.VARIABILITY)]
        elif "eo1" in model_name:
                from simpler_env.policies.eo1.eo1_model import EOInference
                if self.policy_setup == "widowx_bridge":
                    model_path = "../checkpoints/eo1-qwen25_vl-bridge"
                else:
                    model_path = "../checkpoints/eo1-qwen25_vl-fractal"
                self.model = EOInference(saved_model_path=model_path, policy_setup=self.policy_setup)
               # self.variability_models=[Gr00tInference(saved_model_path=model_path, policy_setup=self.policy_setup) for i in range(0,uncerMetrics.VARIABILITY)]
                
        else:
            raise ValueError(model_name)

    def run_interface(self, seed=None, options=None, task_type=None, prompt=None):

        print("Step 1: Environment Make...")
        env = simpler_env.make(self.task)
        print("Step 2: Environment Reset...")
        obs, reset_info = env.reset(seed=seed, options=options)
        instruction = env.unwrapped.get_language_instruction()
        print("Step 3: Model Reset...")
        self.model.reset(instruction)

       # [self.variability_models[i].reset(instruction) for i in range(0, len(self.variability_models))]
        image = get_image_from_maniskill2_obs_dict(env, obs)  # np.ndarray of shape (H, W, 3), uint8
        images = [image]
        predicted_terminated, success, truncated = False, False, False
        timestep = 0
        episode_stats = {}
        actions = []
        tcp_poses = []
        
        tcp_pose = obs['extra']['tcp_pose']

        if task_type == "grasp":
                
            object_pose = env.unwrapped.obj_pose
            total_dist = np.abs(np.linalg.norm(tcp_pose[:3] - object_pose.p))
        else:
            object_pose = env.unwrapped.source_obj_pose
            final_pose = env.unwrapped.target_obj_pose
            total_dist = np.abs(np.linalg.norm(tcp_pose[:3] - object_pose.p)) + np.abs(np.linalg.norm(final_pose.p - object_pose.p))

        while not (predicted_terminated or truncated):
            init_time = time.time()
            # step the model; "raw_action" is raw model action output; "action" is the processed action to be sent into maniskill env
            if "pi0" in self.model_name or "gr00t" in self.model_name or "eo1" in self.model_name:
                raw_action, action = self.model.step(image, instruction, eef_pos=obs["agent"]["eef_pos"])
            elif "spatialvla" in self.model_name:
                raw_action, action = self.model.step(image, instruction)
            else:
                raw_action, action = self.model.step(image)
            predicted_terminated = bool(action["terminate_episode"][0] > 0)
            obs, reward, success, truncated, info = env.step(
                np.concatenate([action["world_vector"], action["rot_axangle"], action["gripper"]])
            )

            print(timestep, info)
            tcp_pose = obs['extra']['tcp_pose']
            tcp_poses.append(tcp_pose.tolist())
            episode_stats[timestep] = info
            action_norm = uncerUtils.normalize_action(action, env.action_space)
            actions.append(action_norm)
            time_execution = time.time()
            print(f"Time to execute: {time_execution - init_time}")
            # update image observation
            image = get_image_from_maniskill2_obs_dict(env, obs)
            images.append(image)
            timestep += 1

        print(f"Episode success: {success}")
        env.close()
        del env
        return images, episode_stats, actions, tcp_poses


class VLAInterfaceLM(VLAInterface):
    def run_interface(self, seed=None, options=None, instruction=None):
        env = simpler_env.make(self.task)
        obs, reset_info = env.reset(seed=seed, options=options)
        if not instruction:
            instruction = env.get_language_instruction()
        self.model.reset(instruction)
        print(instruction)
        print("Reset info", reset_info)

        image = get_image_from_maniskill2_obs_dict(env, obs)  # np.ndarray of shape (H, W, 3), uint8
        images = [image]
        predicted_terminated, success, truncated = False, False, False
        timestep = 0
        episode_stats = {}
        while not (predicted_terminated or truncated):
            # step the model; "raw_action" is raw model action output; "action" is the processed action to be sent into maniskill env
            raw_action, action = self.model.step(image)
            predicted_terminated = bool(action["terminate_episode"][0] > 0)
            obs, reward, success, truncated, info = env.step(
                np.concatenate([action["world_vector"], action["rot_axangle"], action["gripper"]])
            )
            print(timestep, info)
            episode_stats[timestep] = info
            # update image observation
            image = get_image_from_maniskill2_obs_dict(env, obs)
            images.append(image)
            timestep += 1

        print(f"Episode success: {success}")
        env.close()
        del env
        return images, episode_stats


if __name__ == '__main__':
    task_name = "google_robot_pick_customizable"
    model = "rt_1_x"  # @param ["rt_1_x", "rt_1_400k", "rt_1_58k", "rt_1_1k", "octo-base", "octo-small", "openvla-7b]

    vla = VLAInterface(model_name=model, task=task_name)
