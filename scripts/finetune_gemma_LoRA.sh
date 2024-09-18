#!/bin/bash
#SBATCH --job-name=Odesia-gemma-2b-it_LoRa
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --mem=64G
#SBATCH --output=.slurm/Odesia-gemma-2b-it_LoRa.out.txt
#SBATCH --error=.slurm/Odesia-gemma-2b-it_LoRa.err.txt


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
accelerate launch --config_file train_configs/deepspeed.json src/train.py train_configs/gemma2B_LoRA.yaml
torchrun --standalone --master_port 37227 --nproc_per_node=1 src/evaluate.py --tasks all --model_name models/gemma-2b-it_LoRa --output_dir results/finetune/gemma-2b-it_LoRa
torchrun --standalone --master_port 37227 --nproc_per_node=1 src/inference.py --tasks all --model_name models/gemma-2b-it_LoRa --output_dir results/finetune/gemma-2b-it_LoRa




