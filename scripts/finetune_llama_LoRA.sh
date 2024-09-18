#!/bin/bash
#SBATCH --job-name=Odesia-llama_LoRA
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --mem=64G
#SBATCH --output=.slurm/Odesia-llama_LoRA.out.txt
#SBATCH --error=.slurm/Odesia-llama_LoRA.err.txt


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
accelerate launch --config_file train_configs/deepspeed.json src/train.py train_configs/llama8b_LoRA.yaml
torchrun --standalone --master_port 37227 --nproc_per_node=1 src/evaluate.py --tasks all --model_name models/Hermes-3-Llama-3.1-8B_LoRA --output_dir results/finetune/Hermes-3-Llama-3.1-8B_LoRA
torchrun --standalone --master_port 37227 --nproc_per_node=1 src/inference.py --tasks all --model_name models/Hermes-3-Llama-3.1-8B_LoRA --output_dir results/finetune/Hermes-3-Llama-3.1-8B_LoRA



