#!/usr/bin/env python3
"""
Follow_up_test_case_generator.py

Usage examples:
  python Follow_up_test_case_generator.py --mr C_MR1 --model openvla-7b --tasks 0-9
  python Follow_up_test_case_generator.py --mr V_MR2 --model openvla-7b --tasks 3
  python Follow_up_test_case_generator.py --mr C_MR2 --model openvla-7b --tasks 0-2,5,8-9
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import List
from tqdm import tqdm
from datetime import datetime
import random
import copy
import math
import re
import pandas as pd
import numpy as np
RANDOM_SEED=42
random.seed()
# Known MR codes
KNOWN_MRS = ["C_MR1", "C_MR2", "V_MR1", "V_MR2"]
OFFSET_DISTANCE=0.1
NUM_OF_TASKS=20

def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

def parse_args():
    p = argparse.ArgumentParser(description="MR tool: run per-MR function and write JSON outputs per task.")
    p.add_argument("--mr", required=True, help=f"MR code (one of: {', '.join(KNOWN_MRS)})")
    p.add_argument(
        "--tasks",
        required=False,
        help=(
            "Tasks spec: single id (e.g. 5), range (e.g. 0-9), "
            "list (e.g. 1,3,7), or any combination (e.g. 0-2,5,8-9)."
        ),
    )
    p.add_argument("--model", default="gr00t", help="VLA model for which will be generated.")
    p.add_argument("--dataset", default="t-grasp_n-1000_o-m3_s-2498586606.json", help="Dataset of tasks.")
    return p.parse_args()

def expand_task_spec(spec: str) -> List[int]:
    """
    Expand a tasks spec string into a sorted list of unique integers.
    Examples:
      "5" -> [5]
      "0-3" -> [0,1,2,3]
      "1,3,7" -> [1,3,7]
      "0-2,5,8-9" -> [0,1,2,5,8,9]
    """
    out = set()
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    for part in parts:
        if "-" in part:
            lo_str, hi_str = part.split("-", 1)
            lo, hi = int(lo_str), int(hi_str)
            if hi < lo:
                raise ValueError(f"Invalid range '{part}': end < start")
            out.update(range(lo, hi + 1))
        else:
            out.add(int(part))
    return sorted(out)

def get_negative(action: str, max_results: int) -> List[str]:
    gerund_prefixes = [
        "avoid ", "refrain from ", "stop yourself from ", "would you avoid ",
        "resist ", "cease ", "hold back from ", "desist from ",
        "abstain from ", "please refrain from ", "stop from ",
        "do not engage in ", "cease the act of "
    ]
    infinitive_prefixes = [
        "do not attempt to ", "forbid yourself to ", "do not try to ",
        "avoid the urge to ", "do not proceed to ", "do not undertake "
    ]
    bare_prefixes = [
        "do not ", "don't ", "please don't ", "let's not ",
        "never ", "under no circumstances "
    ]
    candidates=[]
    for i in range(min(len(gerund_prefixes)+len(infinitive_prefixes)+len(bare_prefixes), max_results)):
        if i<len(gerund_prefixes):
            prefix=gerund_prefixes[random.randint(0,len(gerund_prefixes)-1)]
        elif i<len(gerund_prefixes)+len(infinitive_prefixes):
            prefix=infinitive_prefixes[i-len(gerund_prefixes)]
        else:
            prefix=bare_prefixes[i-len(gerund_prefixes)- len(infinitive_prefixes)]
        # Split the action into first verb and the rest
        parts = action.strip().split(maxsplit=1)
        verb = parts[0]
        rest = parts[1] if len(parts) > 1 else ""

        # Apply correct form based on prefix
        if prefix in gerund_prefixes:
            # If verb already ends with 'ing', keep it
            if not verb.endswith("ing"):
                # Simple heuristic: add 'ing' (could be improved with a proper lemmatizer)
                if verb.endswith("e") and verb not in ["be", "see", "flee"]:  # drop 'e'
                    verb = verb[:-1] + "ing"
                else:
                    verb = verb + "ing"
        elif prefix in infinitive_prefixes:
            verb = "to " + verb
        # else: bare verb (no change)
        candidates.append(f"{prefix}{verb} {rest}".strip())
    # Construct the final prompt
    return candidates


def get_synonyms(action: str, max_results: int, rest: str) -> List[str]:
    template = {
        'grasp': [
            "grab ",
            "can you pick up ",
            "fetch ",
            "get ",
            "lift ",
            "take ",
            "retrieve ",
            "let's pick up ",
            "would you grab "
        ],
        'move': [
            "take ",
            "bring ",
            "position ",
            "move ",
            "put",
            "place ",
            "set ",
            "can you move ",
            "shift ",
            "let's move "
        ],
        'put-on': [
            "place ",
            "set ",
            "move ",
            "position",
            "put ",
            "could you put ",
            "let's put ",
            "please place ",
            "can you place ",
            "would you move "
        ],
        'put-in': [
            "take",
            "bring ",
            "place ",
            "move ",
            "put",
            "drop",
            "insert ",
            "can you put ",
            "please put",
            "let's put "
        ]
    }

    combinations=template[action]
    candidates=[]
    for i in range(0, min(len(combinations),max_results)):
        candidates.append(combinations[random.randint(0, len(combinations)-1)]+' '+rest)

    
    return candidates

def generate_valid_position(main_pos, pos_range, min_dist, max_attempts=1000):
    dist=0
    while dist<min_dist:
        # Randomly sample a position within the range
        x = random.uniform(pos_range[0][0], pos_range[0][1])
        y = random.uniform(pos_range[1][0], pos_range[1][1])

        # Compute Euclidean distance
        dist = math.sqrt((x - main_pos[0])**2 + (y - main_pos[1])**2)

    return [x, y]
def random_quaternion():
    """Generate a random unit quaternion (x, y, z, w)."""
    u1, u2, u3 = random.random(), random.random(), random.random()
    sqrt1_minus_u1 = math.sqrt(1 - u1)
    sqrt_u1 = math.sqrt(u1)
    return [
        sqrt1_minus_u1 * math.sin(2 * math.pi * u2),
        sqrt1_minus_u1 * math.cos(2 * math.pi * u2),
        sqrt_u1 * math.sin(2 * math.pi * u3),
        sqrt_u1 * math.cos(2 * math.pi * u3)
    ]
def add_to_task(task_data, selected_model: str, task_type:str, prompt:str):
    new_task_data = copy.deepcopy(task_data)
    if task_type=='grasp':
        # Ensure distractor_model_ids exists
        if 'distractor_model_ids' not in new_task_data:
            new_task_data['distractor_model_ids'] = []

        # Ensure distractor_obj_init_options exists
        if 'distractor_obj_init_options' not in new_task_data:
            new_task_data['distractor_obj_init_options'] = {}

        main_object_position = new_task_data['obj_init_options']['init_xy']
        position_range = [(-0.5, -0.05), (0, 0.4)]

        # Add selected model safely
        if selected_model not in new_task_data['distractor_model_ids']:
            new_task_data['distractor_model_ids'].append(selected_model)

        # Add object initialization data
        new_task_data['distractor_obj_init_options'][selected_model] = {
            "init_xy": generate_valid_position(main_object_position, position_range, min_dist=0.1),
            "init_rot_quat": random_quaternion()
        }
    else:
        if 'model_ids' not in new_task_data:
            new_task_data['model_ids'] = []
        if 'obj_init_options' not in new_task_data:
            new_task_data['obj_init_options'] = {}
        if selected_model not in new_task_data['obj_init_options']:
            new_task_data['obj_init_options'][selected_model]={}
        main_object=find_first_referenced_object(prompt,new_task_data['model_ids'])
        main_object_position = new_task_data['obj_init_options'][main_object]['init_xy']

        if task_type=="move":
            position_range = [(-0.5, -0.05), (0, 0.4)]
        elif task_type=="put-on":
            position_range = [(-0.3, 0), (-0.16, 0.12)]
        elif task_type=="put-in":
            position_range = [(-0.2, -0.1), (0.1, 0.2)]
        
        if selected_model not in new_task_data['model_ids']:
            new_task_data['model_ids'].append(selected_model)

        # Add object initialization data
        new_task_data['obj_init_options'][selected_model] = {
            "init_xy": generate_valid_position(main_object_position, position_range, min_dist=0.05),
            "init_rot_quat": random_quaternion()
        }


    return new_task_data


def add_confunding_object(task_data, available_objects: List[str], task_type: str, prompt:str):
    candidates = []

    # Make a copy to avoid modifying the original list
    available_objects_c = available_objects.copy()

    # Remove main object and existing distractors safely
    if task_type == "grasp":
        if 'model_id' in task_data and task_data['model_id'] in available_objects_c:
            available_objects_c.remove(task_data['model_id'])

        if 'distractor_model_ids' in task_data:
            for model in task_data['distractor_model_ids']:
                if model in available_objects_c:
                    available_objects_c.remove(model)
    else:
        if 'model_ids' in task_data:
            for model in task_data['model_ids']:
                if model in available_objects_c:
                    available_objects_c.remove(model)

    for i in range(1,4):
        new_task_data = copy.deepcopy(task_data)
        for j in range(i + 1):
            if not available_objects_c:
                break  # No objects left to choose from
            selected_model = random.choice(available_objects_c)
            new_task_data = add_to_task(new_task_data, selected_model, task_type, prompt)
        candidates.append(new_task_data)

    return candidates
def compute_new_position(curr_pos, off, position_range):
    (xmin, xmax), (ymin, ymax) = position_range
    new_x = curr_pos[0] + off[0]
    new_y = curr_pos[1] + off[1]
    if new_x < xmin:
        new_x = xmin
    elif new_x > xmax:
        new_x = xmax

    # Clamp Y
    if new_y < ymin:
        new_y = ymin
    elif new_y > ymax:
        new_y = ymax
    return new_x, new_y

def find_first_referenced_object(prompt: str, model_ids: list[str]) -> str | None:
    prompt_norm = prompt.lower()

    # Build mapping: model_id → list of readable variants
    variants = {}
    for mid in model_ids:
        words = mid.lower().split("_")
        var_list = set()

        # full phrase ("7up can")
        var_list.add(" ".join(words))

        # partial phrase ("7up", "can", "7up can")
        for i in range(len(words)):
            for j in range(i+1, len(words)+1):
                piece = " ".join(words[i:j])
                if len(piece) > 2:
                    var_list.add(piece)

        variants[mid] = sorted(list(var_list), key=len, reverse=True)

    # Track the earliest match (smallest index)
    best_mid = None
    best_index = len(prompt_norm) + 1

    for mid, var_list in variants.items():
        for v in var_list:
            idx = prompt_norm.find(v)
            if idx != -1 and idx < best_index:
                best_index = idx
                best_mid = mid

    return best_mid

def move_main_object(task_data, task_type:str, mode:str, prompt:str):
    candidates=[]
    directions=[]
    dirs = {
        "right": (0.0, OFFSET_DISTANCE),
        "left": (0.0, -OFFSET_DISTANCE),
        "up": (-OFFSET_DISTANCE, 0.0),
        "down": (OFFSET_DISTANCE, 0.0),
    }
    if mode == "all8":
        dirs.update({
            "up_right": (-OFFSET_DISTANCE, OFFSET_DISTANCE),
            "up_left": (-OFFSET_DISTANCE, -OFFSET_DISTANCE),
            "down_right": (OFFSET_DISTANCE, OFFSET_DISTANCE),
            "down_left": (OFFSET_DISTANCE, -OFFSET_DISTANCE),
        })
    
    if task_type=="grasp":
        position_range = [(-0.5, -0.05), (0, 0.4)]

        for direction, off in dirs.items():
            candidate=copy.deepcopy(task_data)
            new_x, new_y=compute_new_position(candidate['obj_init_options']['init_xy'], off,position_range)            
            candidate['obj_init_options']['init_xy'] = [new_x, new_y]
            candidates.append(candidate)
            directions.append(direction)
            
    else:
        if task_type=="move":
            position_range = [(-0.5, -0.05), (0, 0.4)]
        elif task_type=="put-on":
            position_range = [(-0.3, 0), (-0.16, 0.12)]
        elif task_type=="put-in":
            position_range = [(-0.2, -0.1), (0.1, 0.2)]
        object=find_first_referenced_object(prompt, task_data['model_ids'])
        for direction, off in dirs.items():
            candidate=copy.deepcopy(task_data)
            new_x, new_y=compute_new_position(candidate['obj_init_options'][object]['init_xy'], off,position_range)            
            candidate['obj_init_options'][object]['init_xy'] = [new_x, new_y]
            candidates.append(candidate)
            directions.append(direction)
        

    return candidates, directions
# ---------------------------
# Placeholder MR functions
# ---------------------------
# Each MR function should accept (task_id: int, out_path: Path)
# and create the JSON file for that task. Replace the bodies with
# your real generation logic later.

def create_for_C_MR1(task_id: int, out_path: Path, task_data, prompt, task_type):
    """C_MR1: Consistency pattern Synonym replace."""
    num_variants=1
    parts = prompt.strip().split(maxsplit=1)
    if not parts:
        raise ValueError("Prompt cannot be empty")

    rest = parts[1] if len(parts) > 1 else ""

    candidates = get_synonyms(task_type, num_variants, rest)

    payloads = []
    for i, new_prompt in enumerate(candidates):
        payload = {
            "mr": "C_MR1",
            "task_id": task_id,
            "follow-up": i,
            "task_data": task_data,
            "prompt": new_prompt
        }
        payloads.append(payload)

    # Write all payloads into the same file as a JSON list
    out_path.write_text(json.dumps(payloads, indent=2))

def create_for_C_MR2(task_id: int, out_path: Path, task_data, prompt, task_type):
    """C_MR2: Consistency pattern Add more confunding objects."""
    if task_type== "grasp" or task_type=="move":
        folder_path="../ManiSkill2_real2sim/data/custom/info_pick_custom_v0.json"
    elif task_type=="put-in" or task_type=="put-on":
        folder_path="../ManiSkill2_real2sim/data/custom/info_bridge_custom_v0.json"
    with open(folder_path, 'r') as f:
        objects = json.load(f)
    available_objects=list(objects.keys())
   # available_objects=[f for f in os.listdir(folder_path)
   #           if os.path.isdir(os.path.join(folder_path, f))]
    
    new_tasks_data=add_confunding_object(task_data, available_objects, task_type, prompt)
    selected=random.randint(0, len(new_tasks_data)-1)
    payloads=[]
    #for i, task in enumerate(new_tasks_data):
    #    task
    payload = {
        "mr": "C_MR2",
        "task_id": task_id,
        "follow-up": selected,
        "task_data": new_tasks_data[selected],
        "prompt": prompt
    }
    payloads.append(payload)

    # Write all payloads into the same file as a JSON list
    out_path.write_text(json.dumps(payloads, indent=2))

def create_for_V_MR1(task_id: int, out_path: Path, task_data, prompt, task_type):
    """V_MR1: Variation pattern add a negative statement in the prompt."""
    num_variants=1
    
    candidates = get_negative(prompt, num_variants)

    payloads = []
    for i, new_prompt in enumerate(candidates):
        payload = {
            "mr": "V_MR1",
            "task_id": task_id,
            "follow-up": i,
            "task_data": task_data,
            "prompt": new_prompt,
        }
        payloads.append(payload)

    # Write all payloads into the same file as a JSON list
    out_path.write_text(json.dumps(payloads, indent=2))

def create_for_V_MR2(task_id: int, out_path: Path, task_data, prompt, task_type):
    """V_MR2: Variation pattern move further the target object."""

    mode="all8"
    new_tasks_data, directions=move_main_object(task_data, task_type, mode, prompt)
    payloads=[]
    selected=random.randint(0,7)
    
    #for i, task in enumerate(new_tasks_data):
        #task
    payload = {
        "mr": "V_MR2",
        "task_id": task_id,
        "follow-up": selected,
        "mvoement_type": directions[selected],
        "task_data": new_tasks_data[selected],
        "prompt": prompt
    }
    payloads.append(payload)
    out_path.write_text(json.dumps(payloads, indent=2))

# Map MR code -> function
MR_FUNCTIONS = {
    "C_MR1": create_for_C_MR1,
    "C_MR2": create_for_C_MR2,
    "V_MR1": create_for_V_MR1,
    "V_MR2": create_for_V_MR2,
}

# ---------------------------
# Runner
# ---------------------------

def run_for_mr(mr_code: str, task_ids: List[int], outdir: Path, dataset, overwrite: bool):
    fn = MR_FUNCTIONS.get(mr_code)
    if fn is None:
        raise KeyError(f"No function defined for MR '{mr_code}'")

    # Read task data and prompts
    #task_data="data/t-put-in_n-1000_o-m3_s-2905191776.json"
    with open(dataset, 'r') as f:
        tasks = json.load(f)

    dataset_name = dataset.split('/')[-1]
    match = re.search(r't-(.*?)_n', dataset_name)
    
    prompt_data=f"../data/prompts/{dataset_name}"
    with open(prompt_data, 'r') as f:
        prompts = json.load(f)

    if match:
        task_type = match.group(1)
    mr_outdir = outdir / mr_code / task_type
    mr_outdir.mkdir(parents=True, exist_ok=True)

    for task_id in tqdm(task_ids, desc=f"Processing {mr_code}", unit="task"):
        out_path = mr_outdir / f"task_{task_id}.json"
        if out_path.exists() and not overwrite:
            logging.info("Skipping existing file (use --overwrite to replace): %s", out_path)
            continue

        try:
            fn(task_id, out_path, tasks[str(task_id)], prompts[task_id], task_type)
            logging.info("Wrote: %s", out_path)
        except Exception as e:
            logging.exception("Failed to create output for MR=%s task=%s: %s", mr_code, task_id, e)

def get_from_human_eval(model, dataset):
    result_folder="../results/human_eval/"
    dataset_name = dataset.split('/')[-1]
    match = re.search(r't-(.*?)_n', dataset_name)
    if match:
        task_type = match.group(1)
    
    file=result_folder+f"final_evaluations_{model}_{task_type}.xlsx"
    
    data= pd.read_excel(file)

    high_samples = data[data['final_evaluation'] == 'High Quality']#.sample(n=np.min([20,len(data[data['final_evaluation'] == 'High Quality'])]), random_state=RANDOM_SEED)
    low_samples = data[data['final_evaluation'] == 'Low Quality']#.sample(n=np.min([20,len(data[data['final_evaluation'] == 'Low Quality'])]), random_state=RANDOM_SEED)
    meidum_samples = data[data['final_evaluation'] == 'Medium Quality']
    sampled_df = pd.concat([high_samples, low_samples, meidum_samples]).reset_index(drop=True)

    selected_indexes=list(sampled_df['simulation'].str.extract(r'/(\d+)_simulation\.mp4')[0].astype(int))
    selected_qualities=list(sampled_df['final_evaluation'])

    #existing_indices = data['simulation'].str.extract(r'/(\d+)_simulation\.mp4')[0].astype(int)
    #all_indices = set(range(500))
    #missing_indices = list(all_indices - set(existing_indices))
    #sampled_missing = random.sample(missing_indices, k=20)

    #[selected_indexes.append(i)  for i in sampled_missing]
    #[selected_qualities.append('Failure')  for i in sampled_missing]

    return selected_indexes, selected_qualities



def main():
    setup_logger()
    args = parse_args()

    mr = args.mr
    model=args.model
    dataset= args.dataset
    #mr="V_MR2"
    #model="pi0"
    #dataset="data/t-grasp_n-1000_o-m3_s-2498586606.json"

    if mr not in KNOWN_MRS:
        logging.error("Unknown MR '%s'. Known MRs: %s", mr, ", ".join(KNOWN_MRS))
        raise SystemExit(2)

    try:
        #task_ids=[0]
        #task_ids = expand_task_spec(args.tasks)
        task_ids, quality_of_tasks = get_from_human_eval(model,dataset)
    except Exception as e:
        logging.error("Invalid --tasks spec: %s", e)
        raise SystemExit(3)
    outdir = Path(f"../data/FollowUp/{model}")
    overwrite= False
    run_for_mr(mr, task_ids, outdir,dataset, overwrite=overwrite)
    logging.info("Finished MR %s. Outputs in: %s", mr, outdir / mr)

if __name__ == "__main__":
    main()


    
    