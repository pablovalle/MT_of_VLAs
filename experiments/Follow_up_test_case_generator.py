#!/usr/bin/env python3
"""
Follow_up_test_case_generator.py

Usage examples:
  python Follow_up_test_case_generator.py --mr C_MR1 --tasks 0-9
  python Follow_up_test_case_generator.py --mr V_MR2 --tasks 3
  python Follow_up_test_case_generator.py --mr C_MR2 --tasks 0-2,5,8-9
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

random.seed(42)
# Known MR codes
KNOWN_MRS = ["C_MR1", "C_MR2", "V_MR1", "V_MR2"]
MAX_CONFUNDING_OBJECTS=4


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
        required=True,
        help=(
            "Tasks spec: single id (e.g. 5), range (e.g. 0-9), "
            "list (e.g. 1,3,7), or any combination (e.g. 0-2,5,8-9)."
        ),
    )
    p.add_argument("--outdir", default="results_environment", help="Base output directory.")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing task files if present.")
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
            prefix=gerund_prefixes[i]
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
        'pick': [
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
        candidates.append(combinations[i]+' '+rest)

    
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
def add_to_task(task_data, selected_model: str):
    new_task_data = copy.deepcopy(task_data)

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
        "init_xy": generate_valid_position(main_object_position, position_range, min_dist=0.15),
        "init_rot_quat": random_quaternion()
    }

    return new_task_data


def add_confunding_object(task_data, available_objects: List[str], task_type: str):
    candidates = []

    # Make a copy to avoid modifying the original list
    available_objects_c = available_objects.copy()

    # Remove main object and existing distractors safely
    if 'model_id' in task_data and task_data['model_id'] in available_objects_c:
        available_objects_c.remove(task_data['model_id'])

    if 'distractor_model_ids' in task_data:
        for model in task_data['distractor_model_ids']:
            if model in available_objects_c:
                available_objects_c.remove(model)

    print(f"Available objects for distractors: {len(available_objects_c)}")

    for i in range(MAX_CONFUNDING_OBJECTS):
        new_task_data = task_data
        for j in range(i + 1):
            if not available_objects_c:
                break  # No objects left to choose from
            selected_model = random.choice(available_objects_c)
            new_task_data = add_to_task(new_task_data, selected_model)
        candidates.append(new_task_data)

    return candidates
# ---------------------------
# Placeholder MR functions
# ---------------------------
# Each MR function should accept (task_id: int, out_path: Path)
# and create the JSON file for that task. Replace the bodies with
# your real generation logic later.

def create_for_C_MR1(task_id: int, out_path: Path, task_data, prompt, task_type="pick"):
    """C_MR1: Consistency pattern Synonym replace."""
    num_variants=5
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

def create_for_C_MR2(task_id: int, out_path: Path, task_data, prompt):
    """C_MR2: Consistency pattern Add more confunding objects."""

    task_type="grasp"
    folder_path="ManiSkill2_real2sim/data/custom/models"
    available_objects=[f for f in os.listdir(folder_path)
              if os.path.isdir(os.path.join(folder_path, f))]
    
    new_tasks_data=add_confunding_object(task_data, available_objects, task_type)
    payloads=[]
    for i, task in enumerate(new_tasks_data):
        task
        payload = {
            "mr": "C_MR1",
            "task_id": task_id,
            "follow-up": i,
            "task_data": task,
            "prompt": prompt
        }
        payloads.append(payload)

    # Write all payloads into the same file as a JSON list
    out_path.write_text(json.dumps(payloads, indent=2))

def create_for_V_MR1(task_id: int, out_path: Path, task_data, prompt):
    """V_MR1: Variation pattern add a negative statement in the prompt."""
    num_variants=15
    
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

def create_for_V_MR2(task_id: int, out_path: Path, task_data, prompt):
    """V_MR2: Variation pattern move further the target object."""
    payload = {
        "mr": "V_MR2",
        "task_id": task_id,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "note": "PLACEHOLDER: implement real generator for V_MR2",
        "data": {"example": f"result for task {task_id}"}
    }
    out_path.write_text(json.dumps(payload, indent=2))

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

def run_for_mr(mr_code: str, task_ids: List[int], outdir: Path, overwrite: bool):
    fn = MR_FUNCTIONS.get(mr_code)
    if fn is None:
        raise KeyError(f"No function defined for MR '{mr_code}'")

    mr_outdir = outdir / mr_code
    mr_outdir.mkdir(parents=True, exist_ok=True)

    # Read task data and prompts
    task_data="data/t-grasp_n-1000_o-m3_s-2498586606.json"
    with open(task_data, 'r') as f:
        tasks = json.load(f)
    
    prompt_data="data/prompts/t-grasp_n-1000_o-m3_s-2498586606.json"
    with open(prompt_data, 'r') as f:
        prompts = json.load(f)

    for task_id in tqdm(task_ids, desc=f"Processing {mr_code}", unit="task"):
        out_path = mr_outdir / f"task_{task_id}.json"
        if out_path.exists() and not overwrite:
            logging.info("Skipping existing file (use --overwrite to replace): %s", out_path)
            continue

        try:
            fn(task_id, out_path, tasks[str(task_id)], prompts[task_id])
            logging.info("Wrote: %s", out_path)
        except Exception as e:
            logging.exception("Failed to create output for MR=%s task=%s: %s", mr_code, task_id, e)

def main():
    setup_logger()
    #args = parse_args()

    #mr = args.mr.strip()
    mr="C_MR2"
    if mr not in KNOWN_MRS:
        logging.error("Unknown MR '%s'. Known MRs: %s", mr, ", ".join(KNOWN_MRS))
        raise SystemExit(2)

    try:
        #task_ids = expand_task_spec(args.tasks)
        task_ids = [1,2,3]
    except Exception as e:
        logging.error("Invalid --tasks spec: %s", e)
        raise SystemExit(3)

    outdir = Path("data/FollowUp")
    overwrite= False
    run_for_mr(mr, task_ids, outdir, overwrite=overwrite)
    logging.info("Finished MR %s. Outputs in: %s", mr, outdir / mr)

if __name__ == "__main__":
    main()


    
    