<div align="center">

# Metamorphic Testing of of Vision-Language Action-enabled Robots

> [Pablo Valle](https://scholar.google.com/citations?user=-3y0BlAAAAAJ&hl=en)<sup>1</sup>, [Sergio Segura](https://scholar.google.com/citations?user=AcMLHeEAAAAJ&hl=en)<sup>2</sup>, [Shaukat Ali](https://scholar.google.com/citations?user=S_UVLhUAAAAJ&hl=en)<sup>3</sup>, [Aitor Arrieta](https://scholar.google.com/citations?user=ft06jF4AAAAJ&hl=en)<sup>1</sup></br>
> Mondragon Unibertsitatea<sup>1</sup>, University of Seville<sup>2</sup> ,Simula Research Laboratory<sup>3</sup>

[\[📄Paper\]]()  [\[🔥Project Page\]]()

</div>

In this paper, we explore whether Metamorphic Testing (MT) can alleviate the test oracle problem in this context. To do so, we propose two metamorphic relation patterns and five metamorphic relations to assess whether changes to the test inputs impact the original trajectory of the VLA-enabled robots. An empirical study involving five VLA models, two simulated robots, and four robotic tasks shows that MT can effectively alleviate the test oracle problem by automatically detecting diverse types of failures, including, but not limited to, uncompleted tasks. More importantly, the proposed MRs are generalizable, making the proposed approach applicable across different VLA models, robots, and tasks, even in the absence of test oracles.

## Hardware and Software Requirements


## Installation

<details>
<summary><b>Using Docker (Highly recommended)</b></summary>

Using Docker handles the complex installation of robotics simulators and specific CUDA requirements.

1.  **Build the image:**
    ```bash
    docker build -t mt-vla-artifact .
    ```
2.  **Run the container with GPU support:**
    ```bash
    docker run --gpus all -it --rm -v $(pwd)/data:/app/data mt-vla-artifact
    ```
3.  **Verify the installation:**
    Inside the container, run:
    ```bash
    python3 scripts/check_env.py
    ```
</details>

<details>
<summary><b>From source</b></summary>

If you prefer to install locally, ensure you have **Python 3.9+** and **CUDA 11.8+** installed.

1.  **Create a virtual environment:**
    ```bash
    conda create -n mt_vla python=3.9
    conda activate mt_vla
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Install Simulators:**
    Follow the specific instructions in `docs/SIMULATOR_SETUP.md` to configure the physics engine.
4.  **Download Model Weights:**
    ```bash
    bash scripts/download_weights.sh
    ```
</details>


## Prerequisites:
- CUDA version >=12.
- Cuda toolkit (nvcc)
- An NVIDIA GPU.
- Python >= 3.10
- Vulkan 
- Anaconda or environment virtualization tool

Clone this repo:
```
git clone https://github.com/pablovalle/VLA_UQ.git
```

## Installation for each VLA
Each VLA needs it's own dependencies and libraries, so we opted to generate a virtual environment for each of them. First, follow the following steps that are common for all the environments execpt for Gr00t.

Create an anaconda environment:
```
conda create -n <env_name> python=3.10 (any version above 3.10 should be fine)
conda activate <env_name>
```

Install numpy<2.0 (otherwise errors in IK might occur in pinocchio):
```
pip install numpy==1.24.4
```

Install ManiSkill2 real-to-sim environments and their dependencies:
```
cd {this_repo}/ManiSkill2_real2sim
pip install -e .
```

Install this package:
```
cd {this_repo}
pip install -e .
```

```
sudo apt install ffmpeg
```

```
pip install tensorflow==2.15.0
pip install -r requirements_full_install.txt
pip install tensorflow[and-cuda]==2.15.1 # tensorflow gpu support
```

Install simulated annealing utils for system identification:
```
pip install git+https://github.com/nathanrooy/simulated-annealing
```

Install torch dependencies:
```
pip install torch==2.3.1 torchvision==0.18.1 timm==0.9.10 tokenizers==0.15.2 accelerate==0.32.1
pip install flash-attn==2.6.1 --no-build-isolation
```

Proceed with specific installation for each model in our evaluation:



### OpenVLA
Activate the conda environment:
```
conda activate <env_name>
```

Install the transformers (v.4.40.1) library in this repo:
```
cd {this_repo}/transformers-4.40.1
pip install -e .
```

Download the model:
```
cd {this_repo}/checkpoints
python download_model.py openvla/openvla-7b
```

Update the modeling_prismatic.py:
```
cd {this_repo}/checkpoints
cp modeling_prismatic openvla-7b
```



### pi0
Activate the conda environment:
```
conda activate <env_name>
```

Install the lerobot environment and its packages:
```
cd {this_repo}/lerobot
pip install -e .
```

Install the correct version of numpy, since it was changed:
```
pip install numpy==1.24.4
```

Install pytest to avoid possible errors:
```
pip install pytest
```

Install the transformers (v.4.48.1) library in this repo:
```
cd {this_repo}/transformers-4.48.1
pip install -e .
```

Download the model:
```
cd {this_repo}/checkpoints
python download_model.py HaomingSong/lerobot-pi0-bridge
python download_model.py HaomingSong/lerobot-pi0-fractal
```



### SpatialVLA
Activate the conda environment:
```
conda activate <env_name>
```

Install spatialVLA's requirements:
```
pip install -r spatialVLA_requirements
```

Install the transformers (v.4.48.1) library in this repo:
```
cd {this_repo}/transformers-4.48.1
pip install -e .
```
Download the model:
```
cd {this_repo}/checkpoints
python download_model.py IPEC-COMMUNITY/spatialvla-4b-mix-224-pt
```

Update the modeling_spatialvla.py:
```
cd {this_repo}/checkpoints
cp modeling_spatialvla spatialvla-4b
```


### Nvidia-Gr00t
Create and activate the conda environment:
```
conda create -n <env_name> python=3.10 (any version above 3.10 should be fine)
conda activate <env_name>
```

Install the lerobot environment and its packages:
```
cd {this_repo}/Isaac-GR00T
pip install --upgrade setuptools
pip install -e .[base]
pip install --no-build-isolation flash-attn==2.7.1.post4 
```

Install numpy<2.0 (otherwise errors in IK might occur in pinocchio):
```
pip install numpy==1.24.4
```

Install ManiSkill2 real-to-sim environments and their dependencies:
```
cd {this_repo}/ManiSkill2_real2sim
pip install -e .
```

Install this package:
```
cd {this_repo}
pip install -e .
```

Download the model:
```
cd {this_repo}/checkpoints
python download_model.py youliangtan/gr00t-n1.5-bridge-posttrain
python download_model.py youliangtan/gr00t-n1.5-fractal-posttrain
```

### EO-1
Create and activate the conda environment:
```
conda create -n <env_name> python=3.10 (any version above 3.10 should be fine)
conda activate <env_name>
```

Install the lerobot environment and its packages:
```
cd {this_repo}
git clone https://github.com/EO-Robotics/EO1.git
pip install --upgrade setuptools
pip install -e .
pip install flash-attn==2.8.3 --no-build-isolation
```

Install numpy<2.0 (otherwise errors in IK might occur in pinocchio):
```
pip install numpy==1.24.4
```

Install ManiSkill2 real-to-sim environments and their dependencies:
```
cd {this_repo}/ManiSkill2_real2sim
pip install -e .
```

Install this package:
```
cd {this_repo}
pip install -e .
```

Download the model:
```
cd {this_repo}/checkpoints
python download_model.py IPEC-COMMUNITY/eo1-qwen25_vl-fractal
python download_model.py IPEC-COMMUNITY/eo1-qwen25_vl-bridge
```

## Running the code
To generate the Follow-up test cases execute the following command:
```
cd {this_repo}/experiments
./run_Follow_up_generator.sh <conda_env> <mode_name>
```
To execute the Follow-up test cases on the model run the ```Follow_up_test_cases_launcher.py``` inside the file you can select the model, mrs and tasks to execute.

## Analyzing the results
The result_analysis folder is composed as follows:
- RQ1_result_analyzer.py: Generates an xlsx file with the results for RQ1 and RQ2
- RQ1_Venn.py: Generates the Venn diagram of the paper
- RQ1_heatmap_distances.py: Generates the heatmap of MR violations
- MT_threshold_estimation.py: Makes the preliminary evaluation by showing the distribution of data along with the defined thresholds
- RQ3.py: Selects the videos to be shown on the human evaluation.
