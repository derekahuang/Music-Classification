#!/bin/bash
#
#SBATCH --job-name=test
#
#SBATCH --time=20:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=2G
#SBATCH -p gpu
#SBATCH -c 10
#SBATCH --gres gpu:1

python lin_reg.py
