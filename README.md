<div align="center">

# Metamorphic Testing of Vision-Language Action-enabled Robots

> [Pablo Valle](https://scholar.google.com/citations?user=-3y0BlAAAAAJ&hl=en)<sup>1</sup>, [Sergio Segura](https://scholar.google.com/citations?user=AcMLHeEAAAAJ&hl=en)<sup>2</sup>, [Shaukat Ali](https://scholar.google.com/citations?user=S_UVLhUAAAAJ&hl=en)<sup>3</sup>, [Aitor Arrieta](https://scholar.google.com/citations?user=ft06jF4AAAAJ&hl=en)<sup>1</sup></br>
> Mondragon Unibertsitatea<sup>1</sup>, University of Seville<sup>2</sup>, Simula Research Laboratory<sup>3</sup>

[\[📄Paper\]](https://aitorarrietamarcos.github.io/assets/Metamorphic_vla.pdf)  [\[🔥Project Page\]](https://pablovalle.github.io/MT_of_VLAs_web/)

</div>

In this paper, we explore whether Metamorphic Testing (MT) can alleviate the test oracle problem in this context. To do so, we propose two metamorphic relation patterns and five metamorphic relations to assess whether changes to the test inputs impact the original trajectory of the VLA-enabled robots. An empirical study involving five VLA models, two simulated robots, and four robotic tasks shows that MT can effectively alleviate the test oracle problem by automatically detecting diverse types of failures, including, but not limited to, uncompleted tasks. More importantly, the proposed MRs are generalizable, making the proposed approach applicable across different VLA models, robots, and tasks, even in the absence of test oracles.

## Index
1. [Repo Structure](#repo-structure)
2. [Provided Datasets](#provided-datasets)
2. [Hardware and Software Requirements](#hardware-and-software-requirements)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Citation](#citation)
6. [Acknowledgment](#acknowledgment)

## Repo Structure
The repository is organized as follows:
- [FollowUp_Results](https://github.com/pablovalle/MT_of_VLAs/tree/main/FollowUp_Results): Results of follow-up test cases for each VLA model, task and MR.  
- [Isaac-GR00T](https://github.com/pablovalle/MT_of_VLAs/tree/main/Isaac-GR00T): Submodule/code for the Isaac GR00T VLA model.  
- [ManiSkill2_real2sim](https://github.com/pablovalle/MT_of_VLAs/tree/main/ManiSkill2_real2sim): Submodule for ManiSkill2 real-to-sim experiments.  
- [checkpoints](https://github.com/pablovalle/MT_of_VLAs/tree/main/checkpoints): Folder where trained model weights and checkpoints are downloaded when setting up the environment.  
- [data](https://github.com/pablovalle/MT_of_VLAs/tree/main/data): Dataset containing the source test cases along with the correspoding prompts for the VLA model, Follow-up test cases used in our evaluation for each VLA model, task and MR along with the correspoding prompts for the VLA model.  
- [environment_mount](https://github.com/pablovalle/MT_of_VLAs/tree/main/environment_mount): Setup scripts (`.sh`) for Conda environments per VLA model.  
- [experiments](https://github.com/pablovalle/MT_of_VLAs/tree/main/experiments): Follow-up test case generator, source test case and follow-up test case executor scripts.  
- [lerobot](https://github.com/pablovalle/MT_of_VLAs/tree/main/lerobot): Dependency or submodule related to the “lerobot” for PI0 model.  
- [result_analysis](https://github.com/pablovalle/MT_of_VLAs/tree/main/result_analysis): Analysis and plotting scripts for experiment results.  
- [results](https://github.com/pablovalle/MT_of_VLAs/tree/main/results): Included source test case results for reproducibility.  
- [simpler_env.egg-info](https://github.com/pablovalle/MT_of_VLAs/tree/main/simpler_env.egg-info): Python package metadata.  
- [transformers-4.40.1](https://github.com/pablovalle/MT_of_VLAs/tree/main/transformers-4.40.1): Specific Transformers library version required for OpenVLA.  
- [transformers-4.48.1](https://github.com/pablovalle/MT_of_VLAs/tree/main/transformers-4.48.1): Additional Transformers library version required for SpatialVLA and Pi0.  
- [uncertainty](https://github.com/pablovalle/MT_of_VLAs/tree/main/uncertainty): Additional scripts needed for simulation taken from our [prior work](https://github.com/pablovalle/VLA_UQ).
- [Dockerfile](https://github.com/pablovalle/MT_of_VLAs/blob/main/Dockerfile): Docker configuration for building the replication environment. 

## Provided Datasets
We provide a comprehensive benchmark for evaluating Vision-Language-Action (VLA) models, including the environments and simulators required to run experiments. Specifically, the benchmark supports five VLA models, [Isaac-Groot](https://github.com/NVIDIA/Isaac-GR00T), [pi0](https://github.com/Physical-Intelligence/openpi), [EO-1](https://github.com/SHAILAB-IPEC/EO1), [OpenVLA](https://github.com/DelinQu/SimplerEnv-OpenVLA), and [SpatialVLA](https://github.com/SpatialVLA/SpatialVLA),across four manipulation tasks and two robotic platforms, corresponding to the [Bridge](https://github.com/rail-berkeley/bridge_data_robot) and [Fractal](https://github.com/google-research/robotics_transformer) datasets.

For each task (*pick up*, *move near*, *put in*, and *put on*), we include 500 distinct scenes, along with the execution results for each VLA model, enabling standardized comparisons.

Additionally, the benchmark contains 9,320 follow-up test cases generated using the five proposed MRs. These follow-up cases were derived from the source test cases that successfully passed across all models and tasks. Complete execution results for these follow-up test cases are also provided, allowing thorough evaluation of both task-level correctness and execution-level robustness.

## Hardware and Software Requirements
The software requirements to run this are minimal:
 - Docker
> **Note:** "Minimal" applies to users who choose to run the provided Docker image. Running directly on your local machine requires additional software dependencies, which are detailed in [Building from source](#building-from-source).

The hardware requirements to run this are VLA dependant since each VLA needs a specific amount of GPU RAM, overall the following requirements are needed:
- GPU: 8–12 GB (tested on NVIDIA RTX 4080 Ti and NVIDIA RTX A6000)  
- Disk space: At least 60 GB per VLA model. We tested the setup with 250 GB of free space.

## Installation
There are two ways to install and use this repository:
1. **Using Docker (recommended):** Avoids dependency conflicts and library discrepancies.  
2. **Building from source:** Installs directly on your machine. This method takes longer and may encounter dependency issues.


<details>
<summary><b>Using Docker (Highly recommended)</b></summary>

Docker simplifies the installation of robotics simulators and CUDA requirements. We provide a [Dockerfile](Dockerfile) and a prebuilt image on [Dockerhub](https://hub.docker.com/r/pvalleentrena/mt_of_vlas). Ensure Docker is installed and GPU passthrough is enabled.


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

    If using the prebuilt image:

    ```bash
    docker run -d --gpus all -e NVIDIA_DRIVER_CAPABILITIES=all --name="mt_4_vlas" pvalleentrena/mt_of_vlas
    ```

3.  **Enter the container and verify GPU access:**

    Inside the container, run:
    ```bash
    docker exec -it mt_4_vlas bash
    nvidia-smi
    ```
</details>

<details>
<summary name="Building from source"><b>Building from source</b></summary>

If you prefer a local installation, ensure **CUDA 12.1+** is installed (tested on Ubuntu 22.04 and CUDA 12.1). Then install the following:

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

> **Note:** Ensure NVIDIA drivers (recommended 525+) are installed.
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
The following steps apply to both Docker and source installations.
<details>
<summary><b>Setting up the conda environment and downloading the models</summary></b>

Once everything is sat up and you can access to the repository either on your local machine or inside the docker, for each VLA one conda environment will be generated and the corresponding models will be downloaded. To do so, inside [environment_mount](/environment_mount/) you can find one ```.sh``` file for each model. To setup the environment and download the models:

```
cd {this_repo/environment_mount}
./{model_name}.sh   #Options: EO1, GR00T, PI0, SPATIALVLA, and OPENVLA
```
> **Note:** The model name corresponding to each conda environment is the same name but in lowercase: ```eo1```, ```gr00t```, ```pi0``` and for spatialVLA and openVLA the number of parameters should be added: ```spatialvla-4b``` and ```openvla-7b```.

Once it finishes, you will find the model weights inside [checkpoints](/checkpoints/) folder and you will have the corresponding conda environment wiht the same name as the ```.sh``` file you launched. For example if you launched ```EO1.sh``` you will have a conda env called EO1.
</details>
<details>
<summary><b>Generating the follow-up test cases</summary></b>

To generate the follow-up test cases just a ```.sh``` file should be ran:

```
cd {this_repo/experiments}
./{follow_up_generator.sh}.sh -e <env> -m <model> [options]
```
> **Note:** Pre-generated follow-up cases used in our evaluation are available in [data/FollowUp](/data/FollowUp/). Remove this folder if you want to regenerate test cases.

| Flag | Status   | Description                                                                                     |
|------|----------|-------------------------------------------------------------------------------------------------|
| -e   | REQUIRED | Conda Environment: The name of the specific Conda environment you wish to activate for the run. |
| -m   | REQUIRED | Model Name: The identifier for the model being tested (e.g., eo1, gr00t).            |
| -r   | OPTIONAL | Metamorphic Relations: Specifies which relations to apply (MR1 through MR5). Multiple values should be comma-separated. |
| -t   | OPTIONAL | Task ID Filter: Filters the execution to specific tasks. Accepts arrays like [1,2,3] or ranges like [1-50]. |
| -d   | OPTIONAL | Dataset JSON: Allows you to target specific dataset files (e.g., t-grasp_n-1000_o-m3_s-2498586606.json). |
| -h   | OPTIONAL | Help: Displays the manual and all available options, then exits the script.                  |

> **Note:** If no task ID is specified, to follow our experimental setup it will take all the passing test from the source test cases, so if that the case, please first run the source test cases and provide the human verification. In our case we took this data from [Evaluating Uncertainty and Quality of Visual Language Action-enabled Robots](https://github.com/pablovalle/VLA_UQ).

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
> **Note:** Precomputed results for the source test cases are available in [results](/results). By default, the script will not re-run these test cases. To execute them again, you can rename, move, or remove the ```/results``` folder.

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
> **Note:** Precomputed results for the follow-up test cases are available in [FollowUp_Results](/FollowUp_Results).By default, the script will not re-run these follow-up test cases. To execute them again, you can rename, move, or remove the ```/FollowUp_Results``` folder.

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

<summary><b>Result Analysis</summary></b>

To reproduce the results reported in our paper, we provide precomputed outcomes for both the source test cases ([results](/results)) and the follow-up test cases([FollowUp_Results](/FollowUp_Results)), along with an automated analysis pipeline located in the [result_analysis](/result_analysis) folder. The steps below describe how to analyze the provided results.
> **Note:** The analysis scripts are preconfigured to generate results exactly as presented in the paper. If additional metamorphic relations (MRs) are added, the scripts must be updated to handle them correctly. Similarly, if any VLA models are added or removed, figure sizes may need adjustment.

Before running the analysis, ensure that all required results are present in their corresponding folders (this is in case you renamed, moved or removed any folder). Once ready, execute the following commands:
```
cd {this_repo/result_analysis}
./{result_analysis.sh}.sh -e <env>
```
> **Note:** Specifying the environment is mandatory, as all required analysis libraries are included within each environment. You can perform the analysis using any of the provided environments.

Running the analysis will process the results for RQ1 and RQ2, generating a set of summary files named ```RQ1_results_{model_name}.xlsx```, one per VLA model. Additionally, the 3rd and 4th figures from the paper, along with other relevant visualizations, will be produced in [figures](/results/figures). The analysis will also generate the folders [output_mr](/results/output_mr) and [output_oracle](/results/output_oracle), containing the selected videos used to build the taxonomy described in RQ3.

The taxonomy itself was generated using a questionnaire, which is available here: [questionnaire](https://github.com/pablovalle/MT_questionnaire/tree/main)

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
This project is built with reference to the code of the following projects: [Isaac-Groot](https://github.com/NVIDIA/Isaac-GR00T), [Lerobot](https://github.com/huggingface/lerobot), [EO-1](https://github.com/SHAILAB-IPEC/EO1), [SimplerEnv-OpenVLA](https://github.com/DelinQu/SimplerEnv-OpenVLA), [VLATest](https://github.com/ma-labo/VLATest), and [Evaluating Uncertainty and Quality of Visual Language Action-enabled Robots](https://github.com/pablovalle/VLA_UQ)