#!/bin/bash
#SBATCH --job-name=Odesia-qwen32B
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --mem=64G
#SBATCH --output=.slurm/Odesia-qwen32B.out.txt
#SBATCH --error=.slurm/Odesia-qwen32B.err.txt


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
accelerate launch --config_file train_configs/deepspeed_8.json src/train.py train_configs/qwen32B.yaml
torchrun --standalone --master_port 37227 --nproc_per_node=1 src/evaluate.py --tasks all --quantization --model_name models/Qwen2.5-32B-Instruct --output_dir results/finetune/Qwen2.5-32B-Instruct
torchrun --standalone --master_port 37227 --nproc_per_node=1 src/inference.py --tasks all --quantization --model_name models/Qwen2.5-32B-Instruct --output_dir results/finetune/Qwen2.5-32B-Instruct


