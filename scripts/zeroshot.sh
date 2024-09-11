#!/bin/bash
#SBATCH --job-name=Odesia-zero
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --output=.slurm/Odesia-zero.out.txt
#SBATCH --error=.slurm/Odesia-zero.err.txt


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

python3 -m src.evaluate --tasks all --model_name google/gemma-2-2b-it --output_dir results/zero-shot/gemma-2-2b-it 
python3 -m src.evaluate --tasks all --model_name google/gemma-2-9b-it --output_dir results/zero-shot/gemma-2-9b-it 


python3 -m src.evaluate --tasks all --model_name meta-llama/Meta-Llama-3.1-8B-Instruct --output_dir results/zero-shot/Llama-3.1-8B-Instruct 
python3 -m src.evaluate --tasks all --model_name NousResearch/Hermes-3-Llama-3.1-8B --output_dir results/zero-shot/Hermes-3-Llama-3.1-8B
python3 -m src.evaluate --tasks all --model_name meta-llama/Meta-Llama-3-70B-Instruct --output_dir results/zero-shot/Llama-3-70B-Instruct --quantization



python3 -m src.evaluate --tasks all --model_name 01-ai/Yi-1.5-9B-Chat --output_dir results/zero-shot/Yi-1.5-9B-Chat
python3 -m src.evaluate --tasks all --model_name 01-ai/Yi-1.5-34B-Chat --output_dir results/zero-shot/Yi-1.5-34B-Chat --quantization


python3 -m src.evaluate --tasks all --model_name deepseek-ai/DeepSeek-V2-Lite-Chat --output_dir results/zero-shot/DeepSeek-V2-Lite-Chat

python3 -m src.evaluate --tasks all --model_name allenai/OLMoE-1B-7B-0924 --output_dir results/zero-shot/OLMoE-1B-7B-0924

