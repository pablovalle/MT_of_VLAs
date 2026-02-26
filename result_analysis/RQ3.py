
import os
import json
from collections import defaultdict
from tqdm import tqdm
import pandas as pd

import numpy as np
from frechetdist import frdist
from moviepy import VideoFileClip, TextClip, CompositeVideoClip, clips_array, ColorClip
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

LOW  = np.array([-1.0, -1.0, -1.0, -1.5707964, -1.5707964, -1.5707964, -1.0])
HIGH = np.array([ 1.0,  1.0,  1.0,  1.5707964,  1.5707964,  1.5707964,  1.0])

def getTrajectory(path):
    """
    Loads a trajectory json file containing normalized 7D vectors and
    returns a list of *unnormalized* 7D vectors, each as a tuple.

    Format of each entry in the JSON:
    [
      x, y, z,
      rx, ry, rz,
      gripper
    ]
    """

    with open(path, "r") as f:
        data = json.load(f)

    trajectory = []

    raw = data * (HIGH - LOW) + LOW                      # (N, 7)

    # Extract only x,y,z
    pos = raw[:, :3]                                     # (N, 3)

    # Convert to list of tuples (if needed for frdist)
    return pos

    #return trajectory



def getVerdict(path):
    with open(path, 'r') as f:
        log_data = json.load(f)
        last_timestep_key = str(max(map(int, log_data.keys())))
        success_flag = log_data[last_timestep_key]["success"]
        label = 0 if success_flag=='false' or success_flag==False else 1
    return label

def calcualteOracleMR(verdict_orig, verdict_mt, mr):
    verdict=0
    if mr =="V_MR1":
        if verdict_orig!=verdict_mt:
            verdict=1
    else:
        if verdict_orig==verdict_mt:
            verdict=1

    return verdict
results = []

base_dir = "../results"

mt_results_dir="../FollowUp_Results"
task_mapping={"grasp":"t-grasp_n-1000_o-m3_s-2498586606","move":"t-move_n-1000_o-m3_s-2263834374",
              "put-in":"t-put-in_n-1000_o-m3_s-2905191776","put-on":"t-put-on_n-1000_o-m3_s-2593734741"}
task_mapping2={"t-grasp_n-1000_o-m3_s-2498586606": "grasp","t-move_n-1000_o-m3_s-2263834374": "move",
              "t-put-in_n-1000_o-m3_s-2905191776": "put-in","t-put-on_n-1000_o-m3_s-2593734741": "put-on"}
data_high_mr=pd.read_excel("case_mr_only_High.xlsx")
data_high_mr["threshold"]="High"
data_medium_mr=pd.read_excel("case_mr_only_Medium.xlsx")
data_medium_mr["threshold"]="Medium"
data_low_mr=pd.read_excel("case_mr_only_Low.xlsx")
data_low_mr["threshold"]="Low"

data_mr=pd.concat([data_high_mr, data_medium_mr, data_low_mr])

data_high_oracle=pd.read_excel("case_oracle_only_High.xlsx")
data_high_oracle["threshold"]="High"
data_medium_oracle=pd.read_excel("case_oracle_only_Medium.xlsx")
data_medium_oracle["threshold"]="Medium"
data_low_oracle=pd.read_excel("case_oracle_only_Low.xlsx")
data_low_oracle["threshold"]="Low"

data_oracle=pd.concat([data_high_oracle, data_medium_oracle, data_low_oracle])


sampled_mr = data_mr.sample(n=96, random_state=42)

# Select 96 random videos from the second dataframe
sampled_oracle = data_oracle.sample(n=98, random_state=42)


# ---------------------------------------------------------
# 1. HELPER FUNCTIONS FOR PROMPTS
# ---------------------------------------------------------

def load_json_file(filepath):
    """Safely loads a JSON file."""
    if not os.path.exists(filepath):
        print(f"Warning: File not found {filepath}")
        return None
    with open(filepath, 'r') as f:
        return json.load(f)

def get_prompts(row, base_data_path="../data"):
    """
    Determines the source prompt and the follow-up prompt based on logic.
    Returns: (source_prompt_text, follow_up_prompt_text)
    """
    model = row['model']
    task = row['task']
    scene_id = int(row['scene_id'])
    mr = row['mr']
    
    # --- 1. Get Source Prompt ---
    # Path: data/prompts/{task}.json
    source_json_path = os.path.join(base_data_path, "prompts", f"{task_mapping.get(task)}.json")
    
    source_prompt = "Prompt not found"
    data_source = load_json_file(source_json_path)
    
    if data_source and isinstance(data_source, list) and len(data_source) > scene_id:
        # User specified: "if scene_id is 1 then the [1] prompt is correct"
        source_prompt = data_source[scene_id]
    
    # --- 2. Get Follow-Up Prompt ---
    follow_up_prompt = source_prompt # Default behavior
    
    if mr in ['MR1', 'MR4']:
        # Exception Logic: Look in specific JSON file
        # Path: data/FollowUp/{model}/{MR}/{task}/task_{scene_id}.json
        fu_json_path = os.path.join(
            base_data_path, "FollowUp", model, mr, task, f"task_{scene_id}.json"
        )
        
        fu_data = load_json_file(fu_json_path)
        
        if fu_data:
            # Assuming the JSON structure has a 'prompt' key. 
            # If the JSON IS the prompt string itself, remove ['prompt'].
            # Adjust 'prompt' key below to match your actual JSON structure.
            follow_up_prompt = fu_data[0].get('prompt', str(fu_data))

    return source_prompt, follow_up_prompt

# ---------------------------------------------------------
# 2. VIDEO PROCESSING FUNCTION
# ---------------------------------------------------------
def create_caption_clip(text, width, duration, font_size=20):
    """
    Creates a text clip that auto-sizes height so text isn't cut.
    """
    # 1. Generate the TextClip first to see how tall it needs to be
    # method='caption' wraps the text. 
    # align='North' puts text at the top (closer to video).
    try:
        txt = TextClip(
            text="Prompt: "+text, 
            font_size=font_size, 
            color='white', 
            size=(width - 40, 80), # Width - padding, Height = Auto
            method='label',
            text_align='center',
        )
    except Exception:
        # Fallback for systems with font issues
        txt = TextClip(
            text="Prompt: "+text, 
            font="Arial",
            font_size=font_size, 
            color='white', 
            size=(width - 40, 80), 
            method='label', 
            text_align='center',
        )

    # 2. Create a background that fits the text + a little padding
    # We add 20px padding to the height
    text_height = txt.h
    box_height = text_height + 20 
    
    bg = ColorClip(size=(width, box_height), color=(0,0,0))
    
    # 3. Composite: Put text on background, aligned to TOP ('North')
    # This ensures it is close to the video border
    txt = txt.with_position(('center', 'top'))
    
    final = CompositeVideoClip([bg, txt]).with_duration(duration)
    return final

def process_single_video(args):
    """
    Independent worker function.
    args: (row_dict, output_folder, type_val, task_mapping)
    """
    row, output_folder, type_val, task_mapping = args
    
    try:
        # Extract variables
        task = row['task']
        model = row['model']
        scene_id = row['scene_id']
        mr = row['mr']
        fu_num = row['follow_up_num']
        
        # Resolve paths
        mapped_task = task_mapping.get(task, task)
        
        path_source = f"../results/{mapped_task}/{model}/allMetrics/{scene_id}/{scene_id}_simulation.mp4"
        path_source2 = f"../results/{mapped_task}/{model}/allMetrics/{scene_id}/{scene_id}_simulation_orig.mp4"
        path_followup = f"../FollowUp_Results/{model}/{mr}/{task}/task_{scene_id}/follow_up_{fu_num}/simulation.mp4"
        
        if not os.path.exists(path_source) and os.path.exists(path_source2):
            path_source = path_source2

        if not os.path.exists(path_source) or not os.path.exists(path_followup):
            return f"Missing file for Scene {scene_id}, skipping..."
        
        # --- MoviePy Processing ---
        clip_left = VideoFileClip(path_source)
        clip_right = VideoFileClip(path_followup)
        
        # Resize right to match left height
        if clip_left.h != clip_right.h:
            clip_right = clip_right.resized(height=clip_left.h)

        txt_source, txt_followup = get_prompts(row)
        dur = clip_left.duration

        # --- Create Captions (Bottom) ---
        cap_left = create_caption_clip(txt_source, clip_left.w, dur)
        cap_right = create_caption_clip(txt_followup, clip_right.w, dur)

        # Equalize Caption Heights
        max_h = max(cap_left.h, cap_right.h)
        
        def extend_caption(cap_clip, target_h, width):
            if cap_clip.h < target_h:
                bg = ColorClip(size=(width, target_h), color=(0,0,0)).with_duration(cap_clip.duration)
                return CompositeVideoClip([bg, cap_clip.with_position(('center', 'top'))])
            return cap_clip

        cap_left = extend_caption(cap_left, max_h, clip_left.w)
        cap_right = extend_caption(cap_right, max_h, clip_right.w)

        # Create Left and Right Stacks (Video + Text)
        final_left = clips_array([[clip_left], [cap_left]])
        final_right = clips_array([[clip_right], [cap_right]])

        # --- NEW: Create Middle Divider with MR Name ---
        # Height must match the full stack height (Video + Caption)
        total_height = final_left.h
        divider_width = 80  # Width of the middle bar
        
        # 1. Background for divider
        divider_bg = ColorClip(size=(divider_width, total_height), color=(0,0,0)).with_duration(dur)
        
        # 2. MR Text
        try:
            mr_txt = TextClip(
                text=str(mr), 
                font="Arial", 
                font_size=30,
                size=(None, 700), 
                color='white', 
                method='label'
            )
        except Exception:
            mr_txt = TextClip(
                text=str(mr), 
                font_size=30, 
                size=(None, 80),
                color='white', 
                method='label'
            )
            
        # 3. Composite Divider (Center the text in the black strip)
        divider_col = CompositeVideoClip([divider_bg, mr_txt.with_position('center')]).with_duration(dur)

        # --- Final Assembly: [Left] [Divider] [Right] ---
        final_composition = clips_array([[final_left, divider_col, final_right]])

        output_filename = f"{output_folder}/{model}_{task}_{scene_id}_{mr}_{type_val}.mp4"
        
        # --- WRITE FILE (Fastest Settings) ---
        final_composition.write_videofile(
            output_filename, 
            codec="libx264", 
            audio_codec="aac",
            fps=10,
            preset="ultrafast",      
            threads=1,               # Keep 1 thread per worker!
            ffmpeg_params=[
                "-crf", "30", 
                "-tune", "fastdecode",
                "-movflags", "+faststart"
            ],
            logger=None 
        )

        clip_left.close()
        clip_right.close()
        final_composition.close()
        return None

    except Exception as e:
        return f"Error processing scene {row.get('scene_id')}: {e}"

# --- 2. Main Execution Function ---

def process_videos_parallel(df, output_folder="merged_videos", type_val=None, task_mapping=None):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    if task_mapping is None:
        task_mapping = {} # Prevent crash if not provided

    # Prepare arguments for each task
    # We convert the DataFrame row to a dict for safe pickling
    tasks = []
    for index, row in df.iterrows():
        tasks.append((row.to_dict(), output_folder, type_val, task_mapping))

    # Determine number of workers (Max 60 or CPU count)
    max_workers = min(60, multiprocessing.cpu_count()-2)
    print(f"Starting parallel processing with {max_workers} workers...")

    # Execute
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_video, t): t for t in tasks}
        
        for future in tqdm(as_completed(futures), total=len(tasks), desc=f"Processing {output_folder}"):
            result = future.result()
            if result:
                tqdm.write(result)

# ---------------------------------------------------------
# 3. EXECUTION
# ---------------------------------------------------------

# Assuming sampled_df1 and sampled_df2 are the dataframes created in the previous step:
print("Processing DataFrame 1...")
process_videos_parallel(sampled_mr, output_folder="output_mr", type_val="MR", task_mapping=task_mapping)

print("Processing DataFrame 2...")
print(len(sampled_oracle))
process_videos_parallel(sampled_oracle, output_folder="output_oracle", type_val="Oracle", task_mapping=task_mapping)
