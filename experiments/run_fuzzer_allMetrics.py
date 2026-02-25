
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


PACKAGE_DIR = Path(__file__).parent.resolve()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"

class StableJSONizer(json.JSONEncoder):
    def default(self, obj):
        return super().encode(bool(obj)) \
            if isinstance(obj, np.bool_) \
            else super().default(obj)
    
def convert_to_native(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.generic):  
        return obj.item()
    elif isinstance(obj, list):
        return [convert_to_native(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: convert_to_native(v) for k, v in obj.items()}
    return obj

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="VLA Fuzzing")
    parser.add_argument('-d', '--data', type=str, help="Testing data")
    parser.add_argument('-o', '--output', type=str, default=None, help="Output path, e.g., folder")
    parser.add_argument('-io', '--image_output', type=str, default=None, help="Image output path, e.g., folder")
    parser.add_argument('-s', '--seed', type=int, default=None, help="Random Seed")
    parser.add_argument('-m', '--model', type=str,
                        choices=["openvla-7b", "pi0", "spatialvla-4b", "gr00t", "eo1"],
                        default="gr00t",
                        help="VLA model")
    parser.add_argument('-r', '--resume', type=bool, default=True, help="Resume from where we left.")

    args = parser.parse_args()

    random_seed = args.seed if args.seed else np.random.randint(0, 4294967295)  # max uint32
    
    task_data="t-grasp_n-1000_o-m3_s-2498586606.json"

    data_path = args.data if args.data else str(PACKAGE_DIR) + "/../data/"+task_data

    dataset_name = data_path.split('/')[-1]

    match = re.search(r't-(.*?)_n', dataset_name)

    if match:
        task_type = match.group(1)
    else:
        print("No match found")

    if "grasp" in dataset_name:
        vla = VLAInterface(model_name=args.model, task="google_robot_pick_customizable")
    elif "move" in dataset_name:
        vla = VLAInterface(model_name=args.model, task="google_robot_move_near_customizable")
    elif "put-on" in dataset_name:
        vla = VLAInterface(model_name=args.model, task="widowx_put_on_customizable")
    elif "put-in" in dataset_name:
        vla = VLAInterface(model_name=args.model, task="widowx_put_in_customizable")
    else:
        raise NotImplementedError

    with open(data_path, 'r') as f:
        tasks = json.load(f)

    if args.output:
        result_dir = args.output + data_path.split('/')[-1].split(".")[0]
    else:
        result_dir = str(PACKAGE_DIR) + "/../results/" + data_path.split('/')[-1].split(".")[0]
    os.makedirs(result_dir, exist_ok=True)
    result_dir += f'/{args.model}'#_{random_seed}'
    if not args.resume:
        if os.path.exists(result_dir):
            shutil.rmtree(result_dir)
    os.makedirs(result_dir, exist_ok=True)


    for idx in tqdm(range(round(tasks["num"]/2))):
        if args.resume and os.path.exists(result_dir + f"/allMetrics/{idx}/" + '/log.json'):  # if resume allowed then skip the finished runs.
            continue
        options = tasks[str(idx)]
        images, episode_stats, actions, tcp_poses = vla.run_interface(seed=random_seed, options=options, task_type=task_type)
        os.makedirs(result_dir + f"/allMetrics/{idx}", exist_ok=True)

        with open(result_dir + f"/allMetrics/{idx}/" + '/log.json', "w") as f:
            json.dump(episode_stats, f, cls=StableJSONizer)
        #print(actions)
        json_ready_actions=[
            {key: value.tolist() for key, value in entry.items()}
            for entry in actions
        ]
        with open(result_dir + f"/allMetrics/{idx}/" + '/actions.json', "w") as f:
            json.dump(json_ready_actions, f, indent=2)
        
        with open(result_dir + f"/allMetrics/{idx}/" + '/tcp_poses.json', "w") as f:
            json.dump(tcp_poses, f, indent=2)

            
        video_path = os.path.join(result_dir , f"allMetrics/{idx}", f"{idx}_simulation_orig.mp4")
        video_path_dest = os.path.join(result_dir , f"allMetrics/{idx}", f"{idx}_simulation.mp4")

        # Get frame size from the first image
        height, width = images[0].shape[:2]

        # Create a VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID'
        fps = 10  # Set your desired FPS
        out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

        imagePath=os.path.join(result_dir , f"allMetrics/{idx}","images")
        os.makedirs(imagePath)
        for img_idx in range(len(images)):
            frame = images[img_idx]

            # Ensure frame is uint8 and in BGR (OpenCV uses BGR not RGB)
            if frame.dtype != np.uint8:
                frame = (255 * (frame - frame.min()) / (frame.ptp() + 1e-8)).astype(np.uint8)
            
            if frame.shape[2] == 3:  # RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            out.write(frame)

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

