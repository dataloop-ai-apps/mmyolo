FROM dataloopai/dtlpy-agent:gpu.cuda.11.8.devel.py3.8.pytorch2

USER root
# Install the required packages
RUN apt-get update \
    && apt-get install -y ffmpeg libsm6 libgl1-mesa-glx libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install MMEngine and MMCV
RUN pip install openmim


USER 1000

RUN mim install "mmengine>=0.7.1" "mmcv>=2.0.0rc4" "mmdet>=3.0.0" "mmdet3d>=1.1.0" "mmyolo>=0.6.0"

COPY /requirements.txt .

RUN pip install -r requirements.txt

WORKDIR /mmdetection