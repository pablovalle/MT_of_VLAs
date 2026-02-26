ARG TAG=22.04
FROM ubuntu:$TAG AS builder

RUN apt-get update && DEBIAN_FRONTEND="noninteractive" apt-get install -y --no-install-recommends autoconf build-essential ca-certificates dpkg-dev libpulse-dev lsb-release git libtool libltdl-dev sudo && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/neutrinolabs/pulseaudio-module-xrdp.git /pulseaudio-module-xrdp
WORKDIR /pulseaudio-module-xrdp
ENV DEBIAN_FRONTEND=noninteractive

RUN ln -fs /usr/share/zoneinfo/Etc/UTC /etc/localtime && \
    echo "Etc/UTC" > /etc/timezone && \
    apt-get update && \
    apt-get install -y tzdata
RUN DEBIAN_FRONTEND=noninteractive scripts/install_pulseaudio_sources_apt.sh && \
    ./bootstrap && \
    ./configure PULSE_DIR=$HOME/pulseaudio.src && \
    make && \
    make install DESTDIR=/tmp/install


# Build the final image
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH
RUN apt-get update && DEBIAN_FRONTEND="noninteractive" apt-get install -y --no-install-recommends dbus-x11 git locales pavucontrol pulseaudio pulseaudio-utils software-properties-common sudo vim x11-xserver-utils xfce4 xfce4-goodies xfce4-pulseaudio-plugin xorgxrdp xrdp xubuntu-icon-theme nvidia-utils-525 python3-pip python3-venv python3-dev libaio-dev ffmpeg libavcodec-dev libavformat-dev libswscale-dev && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends \
    gnupg \
    wget \
    nano \
    git-lfs \
    xvfb \
    ffmpeg \
    ca-certificates \
    && \
    wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | \
    gpg --dearmor -o /usr/share/keyrings/google-chrome.gpg && \
    echo "deb [arch=amd64 signed-by=/usr/share/keyrings/google-chrome.gpg] http://dl.google.com/linux/chrome/deb/ stable main" > /etc/apt/sources.list.d/google-chrome.list && \
    apt-get update && \
    apt-get install -y google-chrome-stable && \
    rm -rf /var/lib/apt/lists/* && \
    locale-gen es_ES.UTF-8
RUN apt update && apt install -y \
    vulkan-tools \
    libvulkan1 \
    mesa-vulkan-drivers \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*
# Ensure directories exist and create NVIDIA config files if missing
RUN apt-get update && apt-get install -y libglvnd-dev && \
    mkdir -p /usr/share/vulkan/icd.d/ /usr/share/glvnd/egl_vendor.d/ /etc/vulkan/implicit_layer.d/ && \
    # 1. Handle nvidia_icd.json
    if [ ! -f /usr/share/vulkan/icd.d/nvidia_icd.json ]; then \
        echo '{ "file_format_version" : "1.0.0", "ICD": { "library_path": "libGLX_nvidia.so.0", "api_version" : "1.2.155" } }' > /usr/share/vulkan/icd.d/nvidia_icd.json; \
    fi && \
    # 2. Handle 10_nvidia.json
    if [ ! -f /usr/share/glvnd/egl_vendor.d/10_nvidia.json ]; then \
        echo '{ "file_format_version" : "1.0.0", "ICD" : { "library_path" : "libEGL_nvidia.so.0" } }' > /usr/share/glvnd/egl_vendor.d/10_nvidia.json; \
    fi && \
    # 3. Handle nvidia_layers.json
    if [ ! -f /etc/vulkan/implicit_layer.d/nvidia_layers.json ]; then \
        echo '{ "file_format_version" : "1.0.0", "layer": { "name": "VK_LAYER_NV_optimus", "type": "INSTANCE", "library_path": "libGLX_nvidia.so.0", "api_version" : "1.2.155", "implementation_version" : "1", "description" : "NVIDIA Optimus layer", "functions": { "vkGetInstanceProcAddr": "vk_optimusGetInstanceProcAddr", "vkGetDeviceProcAddr": "vk_optimusGetDeviceProcAddr" }, "enable_environment": { "__NV_PRIME_RENDER_OFFLOAD": "1" }, "disable_environment": { "DISABLE_LAYER_NV_OPTIMUS_1": "" } } }' > /etc/vulkan/implicit_layer.d/nvidia_layers.json; \
    fi && \
    rm -rf /var/lib/apt/lists/*
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p $CONDA_DIR && \
    rm ~/miniconda.sh && \
    conda clean -afy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

# 2. Setup /home/ubuntu and Clone Repository
RUN useradd -m -s /bin/bash ubuntu || echo "User already exists"
WORKDIR /home/ubuntu
RUN git clone https://github.com/pablovalle/MT_of_VLAs.git && \
    chown -R ubuntu:ubuntu /home/ubuntu/MT_of_VLAs

COPY --from=builder /tmp/install /
RUN sed -i 's|^Exec=.*|Exec=/usr/bin/pulseaudio|' /etc/xdg/autostart/pulseaudio-xrdp.desktop

ENV LANG=es_ES.UTF-8
COPY entrypoint.sh /usr/bin/entrypoint
EXPOSE 3389/tcp
ENTRYPOINT ["/usr/bin/entrypoint"]
