
import argparse
import numpy as np
from model_interface_allMetrics import VLAInterface
from pathlib import Path
from tqdm import tqdm
import json
import os
import shutil
import cv2
import re
import torch
import subprocess
import simpler_env
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
from PIL import Image
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)

#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#os.environ.setdefault("VK_ICD_FILENAMES", "/usr/share/vulkan/icd.d/nvidia_icd.json")
        # Try EGL for OpenGL offscreen if available
#os.environ.setdefault("PYOPENGL_PLATFORM", "egl")


# Setup paths
PACKAGE_DIR = Path(__file__).parent.resolve()
#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"
class StableJSONizer(json.JSONEncoder):
    def default(self, obj):
        return super().encode(bool(obj)) \
            if isinstance(obj, np.bool_) \
            else super().default(obj)


if __name__ == '__main__':
    
    task_data="../data/FollowUp/"
    mrs=["MR1","MR2","MR3","MR4","MR5",]
    tasks=["grasp","move", "put-in", "put-on"]#, "grasp","move", "put-in", "put-on"]
    model="eo1"
    
    for task_type in tasks:
        if task_type=='grasp':
            base_env="google_robot_pick_customizable"
        elif task_type=="move":
            base_env="google_robot_move_near_customizable"
        elif task_type=="put-on":
            base_env="widowx_put_on_customizable"
        elif task_type=="put-in":
            base_env="widowx_put_in_customizable"
        vla=None
        vla = VLAInterface(model_name=model, task=base_env)
        for mr in mrs:
            n_scenarios=[f.path for f in os.scandir(f"{task_data}{model}/{mr}/{task_type}")if f.is_file()]     
            for scenario in n_scenarios:
                file=scenario
               # file=f"{task_data}pi0/{mr}/{task_type}/task_{i}.json"
                with open(file, 'r') as f:
                    task = json.load(f)
                for j in range(len(task)):
                    i=scenario.split("_")[-1].split(".")[0]
                    print(f"\n\n Evaluating MR {mr} in task {task_type} for environment {i} the {j}-th time")
                    options=task[j]['task_data']
                    prompt=task[j]['prompt']
                    result_dir=f"../FollowUp_Results/{model}/{mr}/{task_type}/task_{i}/follow_up_{j}"
                    if os.path.exists(result_dir):
                        continue
                    os.makedirs(result_dir, exist_ok=True)
                    
                    images, episode_stats, actions, tcp_poses, = vla.run_interface(seed=1, options=options, task_type=task_type, prompt=prompt)
                    
                    with open(result_dir + f'/log.json', "w") as f:
                        json.dump(episode_stats, f, cls=StableJSONizer)
                    #print(actions)
                    json_ready_actions=[
                        {key: value.tolist() for key, value in entry.items()}
                        for entry in actions
                    ]
                    with open(result_dir + f'/actions.json', "w") as f:
                        json.dump(json_ready_actions, f, indent=2)
                    
                    with open(result_dir + f'/tcp_poses.json', "w") as f:
                        json.dump(tcp_poses, f, indent=2)

                    video_path = os.path.join(result_dir , "simulation_orig.mp4")
                    video_path_dest = os.path.join(result_dir , "simulation.mp4")

                    # Get frame size from the first image
                    height, width = images[0].shape[:2]

                    # Create a VideoWriter object
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID'
                    fps = 10  # Set your desired FPS
                    out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

                    for img_idx in range(len(images)):
                        frame = images[img_idx]

                        # Ensure frame is uint8 and in BGR (OpenCV uses BGR not RGB)
                        if frame.dtype != np.uint8:
                            frame = (255 * (frame - frame.min()) / (frame.ptp() + 1e-8)).astype(np.uint8)
                        
                        if frame.shape[2] == 3:  # RGB
                            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        
                        out.write(frame)
                        #im = Image.fromarray(images[img_idx])
                        #im.save(imagePath+"/Input_"+str(img_idx)+".jpg")



                    out.release()
                    command = [
                        "ffmpeg",
                        "-i", video_path,
                        "-c:v", "h264",  # Or "openh264" if needed
                        video_path_dest
                    ]

                    # Run the command
                    try:
                        subprocess.run(command, check=True)
                        print(f"Video saved to {video_path_dest}")
                    except subprocess.CalledProcessError as e:
                        print("Error during conversion:", e)
                    os.remove(video_path)
                    print(f"Video saved to {video_path}")

   

    
    