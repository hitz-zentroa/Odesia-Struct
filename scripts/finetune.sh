#!/bin/bash
#SBATCH --job-name=Odesia-finetune
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --output=.slurm/Odesia-finetune.out.txt
#SBATCH --error=.slurm/Odesia-finetune.err.txt


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
torchrun --standalone --master_port 37227 --nproc_per_node=2 src/train.py train_configs/gemma2B.yaml
torchrun --standalone --master_port 37227 --nproc_per_node=2 src/train.py train_configs/llama8b.yaml

torchrun --standalone --master_port 37227 --nproc_per_node=2 src/evaluate.py --tasks all --model_name models/gemma-2b-it --output_dir results/finetune/gemma-2b-it
torchrun --standalone --master_port 37227 --nproc_per_node=2 src/evaluate.py --tasks all --model_name models/Llama-3.1-8B-Instruct --output_dir results/finetune/Llama-3.1-8B-Instruct 
