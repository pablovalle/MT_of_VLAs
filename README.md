<div align="center">

# Metamorphic Testing of of Vision-Language Action-enabled Robots

> [Pablo Valle](https://scholar.google.com/citations?user=-3y0BlAAAAAJ&hl=en)<sup>1</sup>, [Sergio Segura](https://scholar.google.com/citations?user=AcMLHeEAAAAJ&hl=en)<sup>2</sup>, [Shaukat Ali](https://scholar.google.com/citations?user=S_UVLhUAAAAJ&hl=en)<sup>3</sup>, [Aitor Arrieta](https://scholar.google.com/citations?user=ft06jF4AAAAJ&hl=en)<sup>1</sup></br>
> Mondragon Unibertsitatea<sup>1</sup>, University of Seville<sup>2</sup>, Simula Research Laboratory<sup>3</sup>

[\[📄Paper\]](https://aitorarrietamarcos.github.io/assets/Metamorphic_vla.pdf)  [\[🔥Project Page\]](https://pablovalle.github.io/MT_of_VLAs_web/)

</div>

In this paper, we explore whether Metamorphic Testing (MT) can alleviate the test oracle problem in this context. To do so, we propose two metamorphic relation patterns and five metamorphic relations to assess whether changes to the test inputs impact the original trajectory of the VLA-enabled robots. An empirical study involving five VLA models, two simulated robots, and four robotic tasks shows that MT can effectively alleviate the test oracle problem by automatically detecting diverse types of failures, including, but not limited to, uncompleted tasks. More importantly, the proposed MRs are generalizable, making the proposed approach applicable across different VLA models, robots, and tasks, even in the absence of test oracles.

## Index
1. [Repo Structure](#repo-structure)
2. [Hardware and Software Requirements](#hardware-and-software-requirements)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Citation](#citation)
6. [Acknowledgment](#acknowledgment)

## Repo Structure

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


## Usage
This section details the usage of the approach in this repository. The steps described in this section are the same for both, the docker-based installation and for the building from source installation.
<details>
<summary><b>Setting up the conda environment and downloading the models</summary></b>

Once everything is sat up and you can access to the repository either on your local machine or inside the docker, for each VLA one conda environment will be generated and the corresponding models will be downloaded, for that inside [environment_mount](/environment_mount/) you can find one ```.sh``` file for each model. To setup the environment and download the models:

```
cd {this_repo/environment_mount}
./{model_name}.sh   #Options: EO1, GR00T, PI0, SPATIALVLA, and OPENVLA
```
> **Note:** The model name corresponding to each conda environment is the same name but in lowercase: ```eo1```, ```gr00t```, ```pi0``` and for spatialVLA and openVLA the number of parameters should be added: ```spatialvla-4b``` and ```openvla-7b```

Once it finishes, you will find the model weights inside [checkpoints](/checkpoints/) folder and you will have the corresponding conda environment wiht the same name as the ```.sh``` file your launched. For example if you launched ```EO1.sh``` you will have a conda env called EO1.
</details>
<details>
<summary><b>Generating the follow-up test cases</summary></b>

To generate the follow-up test cases just a ```.sh``` file should be ran:

```
cd {this_repo/experiments}
./{follow_up_generator.sh}.sh -e <env> -m <model> [options]
```
> **Note:** We already provide the Follow-up test cases in [data/FollowUp](/data/FollowUp/) so maybe there will be no test cases generated. If you want to generate another ones, you can remove or move the ```/data/FollowUp``` folder.

| Flag | Status   | Description                                                                                     |
|------|----------|-------------------------------------------------------------------------------------------------|
| -e   | REQUIRED | Conda Environment: The name of the specific Conda environment you wish to activate for the run. |
| -m   | REQUIRED | Model Name: The identifier for the model being tested (e.g., eo1, gr00t).            |
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

<summary><b>Executing Source Test Cases</summary></b>

To execute the source test cases just a ```.sh``` file should be ran:

```
cd {this_repo/experiments}
./{run_source_test_cases.sh}.sh -e <env> -m <model> [options]
```
> **Note:** We already provide the results for the source test cases in [results](/results) so no test cases will be run. If you want to run another one, you can remove or move the ```/results``` folder.

| Flag | Status   | Description |
|------|----------|------------|
| -e   | REQUIRED | Conda environment name to activate. |
| -m   | REQUIRED | Specific model name to run (e.g., eo1, gr00t3). |
| -d   | OPTIONAL | Dataset JSON filename(s). Accepts one or multiple values separated by commas (e.g.,  t-grasp_n-1000_o-m3_s-2498586606.json,t-move_n-1000_o-m3_s-2263834374.json). Default: Runs the four standard datasets (t-grasp_n-1000_o-m3_s-2498586606.json, t-move_n-1000_o-m3_s-2263834374.json, t-put-in_n-1000_o-m3_s-2905191776.json, t-put-on_n-1000_o-m3_s-2593734741.json). |
| -h   | OPTIONAL | Displays the help message and exits. |

Usage example:
```
./run_source_test_cases.sh -e EO1 -m eo1 -d t-put-on_n-1000_o-m3_s-2593734741.json
```
</details>
<details>

<summary><b>Executing Follow-Up Test Cases</summary></b>

To execute the follow-up test cases just a ```.sh``` file should be ran:

```
cd {this_repo/experiments}
./{run_follow_up_test_cases.sh}.sh -e <env> -m <model> [options]
```
> **Note:** We already provide the results for the follow-up test cases in [FollowUp_Results](/FollowUp_Results) so no test cases will be run. If you want to run another one, you can remove or move the ```/FollowUp_Results``` folder.

| Flag | Status   | Description |
|------|----------|------------|
| -e   | REQUIRED | Conda environment name to activate. |
| -m   | REQUIRED | Specific model name to run (e.g., eo1, gr00t). |
| -r   | OPTIONAL | Metamorphic Relations (MRs). Accepts single or multiple values separated by commas or spaces (e.g., "MR1,MR2,MR5"). Default: MR1, MR2, MR3, MR4, MR5. |
| -d   | OPTIONAL | Dataset JSON filename(s). Accepts one or multiple values separated by commas (e.g., grasp,move). Default: Runs the four standard datasets (grasp, move, put-in, put-on). |
| -h   | OPTIONAL | Displays the help message and exits. |

Usage example:
```
./run_source_test_cases.sh -e EO1 -m eo1 -r MR1,MR3 -d grasp,move
```
</details>
<details>

<summary><b>Result Analisis</summary></b>

To reproduce the results of our paper we provide the results for the source test cases ([results](/results)) and the results for the follow-up test cases ([FollowUp_Results](/FollowUp_Results)) along with an automated pipeline to analyze the results in the [result_analysis](/result_analysis) folder. Below we detail the steps to analyze the results we provide.
> **Note:** The scripts are already configured to calculate the results ad-hoc for our paper, in case an additional MR is addedd, the scripts must be updated to properly calculate the MR accordingly. In case any additional VLA is added or removed, the figure size must be updated accordingly.

First, ensure all the results are available in the corresponding folders. Once all the results are available the following commans should be executed:
```
cd {this_repo/result_analysis}
./{result_analysis.sh}.sh -e <env>
```
> **Note:** The environment is mandatory since all the analisis libraries are already included in each environment, you can analyze the results with any environment.

This will analyze the results for RQ1 and RQ2 by generating a set of summary ```RQ1_results_{model_name}.xlsx``` files. Each corresponds to one model. In addition in [figures](/results/figures) the 3rd and 4th figure of the paper will be generated along with additional interesting data. In addition it will generate the [output_mr](/results/output_mr) and [output_oracle](/results/output_oracle) corresponding to the selection of videos used to generate the Taxonomy in RQ3. 

To generate this taxonomy we used a questionnaire that is available at: [questionnaire](https://github.com/pablovalle/MT_questionnaire/tree/main)

</details>

## Citation
If you find this project useful in your research, please consider cite:

```BibTeX
@article{,
  title={},
  author={},
  journal={},
  year={}
}
```

## Acknowledgement
This project is buil with reference to the code of the following projects: [Isaac-Groot](https://github.com/NVIDIA/Isaac-GR00T), [Lerobot](https://github.com/huggingface/lerobot), [EO-1](https://github.com/SHAILAB-IPEC/EO1) [SimplerEnv-OpenVLA](https://github.com/DelinQu/SimplerEnv-OpenVLA), [VLATest](https://github.com/ma-labo/VLATest), and [Evaluating Uncertainty and Quality of Visual Language Action-enabled Robots](https://github.com/pablovalle/VLA_UQ)