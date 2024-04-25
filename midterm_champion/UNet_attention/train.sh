#!/bin/bash -l

# Set SCC project
#$ -P ds598xz

# Request 4 CPUs
#$ -pe omp 1

# Request 1 GPU 
#$ -l gpus=1

# module load python3/3.8.10
# # PyTorch requires a compute capability of at least 6.0
# module load pytorch/1.11.0
module load miniconda
conda activate Lux
module load gcc/10.2.0
module load nodejs/14.16.1

# Download the data from https://www.kaggle.com/datasets/bomac1/luxai-replay-dataset and save it here midterm_champion/full_episodes/top_agents folder
# Run this script from midterm_champion
python ./UNet_attention/train.py --model_dir "models" --model "unet" --multiplier 1 > ./logs/log.txt