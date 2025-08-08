# Use a 'large' base container to show-case how to load pytorch (macOS)
# FROM --platform=linux/arm64 pytorch/pytorch AS example-task2-arm64

# Use a 'large' base container to show-case how to load pytorch and use the GPU (when enabled) (Linux and WSL)
FROM --platform=linux/amd64 pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime AS example-task3-amd64

# Ensures that Python output to stdout/stderr is not buffered: prevents missing information when terminating
ENV PYTHONUNBUFFERED=1

RUN groupadd -r user && useradd -m --no-log-init -r -g user user
USER user

WORKDIR /opt/app

#Set needed environment variable
ENV nnUNet_results="/opt/app/nnUNet_results"
ENV nnUNet_preprocessed="/opt/app/nnUNet_preprocessed"
ENV nnUNet_raw="/opt/app/nnUNet_raw"

#upgrade pip
RUN python -m pip install --user -U pip && python -m pip install --user pip-tools && python -m pip install --upgrade pip

# Copy files, especially nnunet_results
COPY --chown=user:user Hecktor2025 /opt/app/Hecktor2025
COPY --chown=user:user nnUNet_HN /opt/app/nnUNet_HN
COPY --chown=user:user nnUNet_results_submission /opt/app/nnUNet_results

# install all needed packages
RUN cd Hecktor2025 && pip install --user --no-cache-dir --no-color -r requirements.txt
RUN cd nnUNet_HN && pip install -e .

RUN chmod +x Hecktor2025/src/run.py

ENTRYPOINT ["python", "Hecktor2025/src/run.py"] 


### To build
# You should be located in the folder where you clone the repository
# Example tree:
# WhateverFolder
# ├── HECKTOR2025
# │   └── Hecktor2025
# │       └── src
# │           └── Dockerfile
# ├── input
# │    └──  images
# │           ├── ct
# │           |   └── HMR-030__CT.nii.gz
# │           └── pet
# │               └── HMR-030__PT.nii.gz
# ├── output
# └── nnUNet_results_submission
# Go in HECKTOR2025 or whatever you named your working folder, then you can run the following command:
# docker build -t hecktor25_ds002 -f Hecktor2025/src/Dockerfile .
### To export
# docker save hecktor25_ds002 | gzip -c > hecktor25_ds002.tar.gz
### To test put files to predict in input directory
# Output directory will contain segmentations
docker run --rm \
--gpus '"device=0"' \
--runtime=nvidia \
--memory="32g" \
--memory-swap="32g" \
--network="none" \
--cap-drop="ALL" \
--security-opt="no-new-privileges" \
--shm-size="128m" \
--pids-limit="256" \
-v ./input/:/input/ \
-v ./output/:/output/ \
hecktor25_ds002


### To run the docker interactively
# docker run -it --entrypoint /bin/bash hecktor25_ds002


### To test the docker you can simply create (../HECKTOR2025/Hecktor2025/src/tmp/convert_files.py)
# import SimpleITK as sitk
# pet = sitk.ReadImage("/Data/Seb/Hecktor2025/Task_1/HMR-030/HMR-030__PT.nii.gz")
# sitk.WriteImage(pet, "/home/sebq/HECKTOR2025/input/images/pet/HMR-030.mha", True)
# ct = sitk.ReadImage("/Data/Seb/Hecktor2025/Task_1/HMR-030/HMR-030__CT.nii.gz")
# sitk.WriteImage(ct, "/home/sebq/HECKTOR2025/input/images/ct/HMR-030.mha", True)