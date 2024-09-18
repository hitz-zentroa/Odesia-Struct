#!/bin/bash
#SBATCH --job-name=Odesia-llama-inference
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --output=.slurm/Odesia-llama-inference.out.txt
#SBATCH --error=.slurm/Odesia-llama-inference.err.txt


source /ikerlariak/igarcia945/envs/pytorch2/bin/activate


export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8
export LANGUAGE=en_US.UTF-8
export TOKENIZERS_PARALLELISM=true
export TRANSFORMERS_NO_ADVISORY_WARNINGS=true
export WANDB_ENTITY=igarciaf
export WANDB_PROJECT=Odesia
export OMP_NUM_THREADS=16
export WANDB__SERVICE_WAIT=300

echo CUDA_VISIBLE_DEVICES "${CUDA_VISIBLE_DEVICES}"


export PYTHONPATH="$PYTHONPATH:$PWD"
torchrun --standalone --master_port 37227 --nproc_per_node=1 src/evaluate.py --tasks all --model_name HiTZ/Hermes-3-Llama-3.1-8B_ODESIA --output_dir results/finetune/Hermes-3-Llama-3.1-8B_ODESIA



