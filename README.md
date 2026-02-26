<div align="center">

# Metamorphic Testing of of Vision-Language Action-enabled Robots

> [Pablo Valle](https://scholar.google.com/citations?user=-3y0BlAAAAAJ&hl=en)<sup>1</sup>, [Sergio Segura](https://scholar.google.com/citations?user=AcMLHeEAAAAJ&hl=en)<sup>2</sup>, [Shaukat Ali](https://scholar.google.com/citations?user=S_UVLhUAAAAJ&hl=en)<sup>3</sup>, [Aitor Arrieta](https://scholar.google.com/citations?user=ft06jF4AAAAJ&hl=en)<sup>1</sup></br>
> Mondragon Unibertsitatea<sup>1</sup>, University of Seville<sup>2</sup>, Simula Research Laboratory<sup>3</sup>

[\[📄Paper\]](https://aitorarrietamarcos.github.io/assets/Metamorphic_vla.pdf)  [\[🔥Project Page\]](https://pablovalle.github.io/MT_of_VLAs_web/)

</div>

In this paper, we explore whether Metamorphic Testing (MT) can alleviate the test oracle problem in this context. To do so, we propose two metamorphic relation patterns and five metamorphic relations to assess whether changes to the test inputs impact the original trajectory of the VLA-enabled robots. An empirical study involving five VLA models, two simulated robots, and four robotic tasks shows that MT can effectively alleviate the test oracle problem by automatically detecting diverse types of failures, including, but not limited to, uncompleted tasks. More importantly, the proposed MRs are generalizable, making the proposed approach applicable across different VLA models, robots, and tasks, even in the absence of test oracles.

## Hardware and Software Requirements
The software requirements to run this are minimal:
 - Docker
> **Note:** "Minimal" refers ot the users that want to use the docker we provide. However, to run this on your local machine will require additional software requirements listed in [Building from source](#Building from source).

The hardware requirements to run this are VLA dependant since each VLA needs a specific amount of GPU RAM, overall the following requirements are needed:
 - 8-12GBs GPU (We tested the approach in a NVIDIA RTX4080Ti and in a NVIDIA RTX A6000)
 - A high amount of disk space. For each VLA model around 60GBs are needed. (We tested the approach with 250GBs of free space)

## Installation
For the installation and usage of this repository there are two alternatives:
 - Using Docker, which we strongly recommend, to avoid dependency conflicts or library discrepancies. 
 - Running in your machine building from source this repo. This option will require much more time and some depedency issues may arise between the software requirements and the environments.

Below you can find a furhter guide on how to setup for both cases:

<details>
<summary><h3><b>Using Docker (Highly recommended)</b></h3></summary>

Using Docker handles the complex installation of robotics simulators and specific CUDA requirements. For that we provide a [Dockerfile](Dockerfile) and also we provide a Docker image at [Dockerhub](https://hub.docker.com/r/pvalleentrena/mt_of_vlas). before starting ensure you have installed docker and that you can passthrough your GPU to the docker.

1.  **Build the image:**

    Download only the Docker file and build de image:
    ```bash
    docker build -t mt_4_vlas .
    ```
2.  **Run the container with GPU support:**

    If you built from image:

    ```bash
    docker run -d --gpus all -e NVIDIA_DRIVER_CAPABILITIES=all --name="mt_4_vlas" mt_4_vlas
    ```

    If you are runing from the image we provide:

    ```bash
    docker run -d --gpus all -e NVIDIA_DRIVER_CAPABILITIES=all --name="mt_4_vlas" pvalleentrena/mt_of_vlas
    ```

3.  **Enter in the container and check that you see the GPUs:**

    Inside the container, run:
    ```bash
    docker exec -it mt_4_vlas bash
    nvidia-smi
    ```
</details>

<details>
<summary name="Building from source"><h3><b>Building from source</b></h3></summary>

If you prefer to install locally, ensure you have **CUDA 12.1+** installed (we tested it using Cuda 12.1).
Install the following commands (We tested it in an Ubuntu 22.04 machine):

1. **The core graphics, audio, and utility packages:**

```bash
sudo apt-get update && sudo apt-get install -y \
    dbus-x11 git locales pavucontrol pulseaudio pulseaudio-utils \
    software-properties-common sudo vim x11-xserver-utils \
    xfce4 xfce4-goodies xorgxrdp xrdp ffmpeg libaio-dev \
    python3-pip python3-venv python3-dev \
    vulkan-tools libvulkan1 mesa-vulkan-drivers libgl1-mesa-glx \
    gnupg wget nano git-lfs xvfb ca-certificates
```
2. **NVIDIA & Vulkan Configuration:**

> **Note:** Ensure you have NVIDIA drivers installed on your host (recommended version 525+).
```bash
sudo apt-get install -y libglvnd-dev
# if it doesn't exist
sudo mkdir -p /usr/share/vulkan/icd.d/ /usr/share/glvnd/egl_vendor.d/

# Configure Vulkan ICD if it doesn't exist
echo '{ "file_format_version" : "1.0.0", "ICD": { "library_path": "libGLX_nvidia.so.0", "api_version" : "1.2.155" } }' | sudo tee /usr/share/vulkan/icd.d/nvidia_icd.json

# Configure EGL Vendor if it doesn't exist
echo '{ "file_format_version" : "1.0.0", "ICD" : { "library_path" : "libEGL_nvidia.so.0" } }' | sudo tee /usr/share/glvnd/egl_vendor.d/10_nvidia.json
```

3. **Miniconda Setup:**

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
bash ~/miniconda.sh -b -p $HOME/miniconda
source "$HOME/miniconda/bin/activate"
conda init bash
```

4. **Repository Setup:**

```bash
git clone https://github.com/pablovalle/MT_of_VLAs.git
cd MT_of_VLAs
```
</details>


## Usage:
<details>
<summary><b>Setting up the conda environment and downloading the models</summary></b>

Once everything is sat up and you can access to the repository either on your local machine or inside the docker, for each VLA one conda environment will be generated and the corresponding models will be downloaded, for that inside [environment_mount](/environment_mount/) you can find one ```.sh``` file for each model. To setup the environment and download the models:

```
cd {this_repo/environment_mount}
./{model_name}.sh   #Options: EO1, GR00T, PI0, SPATIALVLA, and OPENVLA
```

Once it finishes, you will find the model weights inside [checkpoints](/checkpoints/) folder and you will have the corresponding conda environment wiht the same name as the ```.sh``` file your launched. For example if you launched ```EO1.sh``` you will have a conda env called EO1.
</details>
<details>
<summary><b>Generating the follow-up test cases</summary></b>

To generate the follow-up test cases just a .sh file should be ran:

```
cd {this_repo/experiments}
./{follow_up_generator.sh}.sh -e <env> -m <model> [options]
```

| Flag | Status   | Description                                                                                     |
|------|----------|-------------------------------------------------------------------------------------------------|
| -e   | REQUIRED | Conda Environment: The name of the specific Conda environment you wish to activate for the run. |
| -m   | REQUIRED | Model Name: The identifier for the model being tested (e.g., gpt-4, llama-3, eo1).            |
| -r   | OPTIONAL | Metamorphic Relations: Specifies which relations to apply (MR1 through MR5). Multiple values should be comma-separated. |
| -t   | OPTIONAL | Task ID Filter: Filters the execution to specific tasks. Accepts arrays like [1,2,3] or ranges like [1-50]. |
| -d   | OPTIONAL | Dataset JSON: Allows you to target specific dataset files (e.g., t-grasp_n-1000_o-m3_s-2498586606.json). |
| -h   | OPTIONAL | Help: Displays the manual and all available options, then exits the script.                  |

Usage example:
```
./follow_up_generator.sh -e EO1 -m eo1 -r MR1,MR3 -t [1-10,15,18] -d t-grasp_n-1000_o-m3_s-2498586606.json
```
</details>
<details>


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
