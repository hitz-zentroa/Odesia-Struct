<p align="center">
    <br>
    <img src="ODESIA.png" style="height: 250px;">
    <br>
    <h3 align="center">Evaluation of NLP models in Spanish</h3>
    <h1 align="center">IXA Submission for the 2024 ODESIA Challenge</h1>
    


<p align="center">
    <a href="https://twitter.com/intent/tweet?text=The+IXA+Code+for+Odesia:&url=https%3A%2F%2Fgithub.com%2Fhitz-zentroa%2FOdesia-Struct"><img alt="Twitter" src="https://img.shields.io/twitter/url?style=social&url=https%3A%2F%2Fgithub.com%2Fhitz-zentroa%2FOdesia-Struct"></a>
    <a href="https://github.com/hitz-zentroa/Odesia-Struct/blob/main/LICENSE.md"><img alt="GitHub license" src="https://img.shields.io/github/license/hitz-zentroa/Odesia-Struct"></a>
    <a href="https://huggingface.co/collections/HiTZ/odesia-challenge-2024-66ea8e1731b52eaa8f3d8cd8"><img alt="Pretrained Models" src="https://img.shields.io/badge/🤗HuggingFace-Pretrained Models-green"></a>
    <a href="https://upload.wikimedia.org/wikipedia/commons/8/80/Comingsoon.png"><img alt="Paper" src="https://img.shields.io/badge/📖-Paper-orange"></a>
<br>
     <a href="http://www.hitz.eus/"><img src="https://img.shields.io/badge/HiTZ-Basque%20Center%20for%20Language%20Technology-blueviolet"></a>
    <a href="http://www.ixa.eus/?language=en"><img src="https://img.shields.io/badge/IXA-%20NLP%20Group-ff3333"></a>
    <br>
     <br>
</p>


This repository contains the IXA submission for the 2024 ODESIA Challenge.
- 📈 ODESIA Leaderboard: https://leaderboard.odesia.uned.es/leaderboard/challenge
- 📒 System Description Paper: Cooming Soon


# Explanation of the approach

Every task is converted into a text-to-text task in a format suitable for current state-of-the-art decoder-only models. 

## ⤵️ Model Input
We format every task following the same prompt schema. The prompt includes the guidelines for the task, up to 20 few-shot examples randomly sampled from the train split, and the input to analyze. 

```jinja
{{ Guidelines }}

Examples
--------

{% for example in examples %}
Input: {{ example.question }}
Output: {{ example.answer.model_dump_json() }}

{% endfor %}

--------

Now, analyze the following input:

Input: {{ question }}
```

## ➡️ Model output

Every task output is defined as a JSON schema using Pydantic. For example, for `DIPROMATS_2023`, which is a multi-label classification task, the output is defined as follows:


```python
class LabelEnum(str, Enum):
    ad_populum = "ad-populum"
    flag_waving = "flag-waving"
    absurdity_appeal = "absurdity-appeal"
    demonization = "demonization"
    doubt = "doubt"
    fear_appeals_destructive = "fear-appeals-destructive"
    name_calling = "name-calling"
    propaganda_slinging = "propaganda-slinging"
    scapegoating = "scapegoating"
    undiplomatic_assertiveness_whataboutism = (
        "undiplomatic-assertiveness-whataboutism"
    )
    loaded_language = "loaded-language"
    appeal_to_false_authority = "appeal-to-false-authority"
    bandwagoning = "bandwagoning"
    non_propaganda = "non-propaganda"

class Identification(BaseModel):
    label: List[LabelEnum]
```


Using [🗒️ Outlines](https://github.com/dottxt-ai/outlines), we use guided generation to produce the output. At inference, the model is forced to produce a valid `JSON` output that is compliant with the Pydantic specification. For example:


```python
{"label":["ad-populum","loaded-language"]}
```


The Guidelines and output specification for every task are defined in [src/tasks](src/tasks)

## Model finetuning

We finetune a decoder-only model (gemma-2b or Llama3.1) in a multi-task setting. This means we train a single model that works for every task. Our pretrained models are available on 🤗HuggingFace: https://huggingface.co/collections/HiTZ/odesia-challenge-2024-66ea8e1731b52eaa8f3d8cd8

# Reproduce our results

## Requirements

You should install the following requirements. All of them can be installed with `pip install [requirement]`


```
torch
transformers
accelerate
deepspeed
outlines
pydantic
bitsandbytes
jinja2
seqeval
```

To reproduce our environment install the `requirements.txt` file:

```bash
pip install -r requirements.txt
```
 

You should unzip the `.zip` file in [data/](data/).  
The expected data structure is 
```
data/
    diann_2023/
        DIANN_2023_T1_en.json
        DIANN_2023_T1_en.json
        ...
    dipromats_2023/
        ...
    exist_2022/
        ...
    exist_2023/
        ...
    sqac_squad_2024
        ...
```

## Models
We have trained 3 models of different size:
- HiTZ/Qwen2.5-14B-Instruct_ODESIA: https://huggingface.co/HiTZ/Qwen2.5-14B-Instruct_ODESIA
- HiTZ/Hermes-3-Llama-3.1-8B_ODESIA: https://huggingface.co/HiTZ/Hermes-3-Llama-3.1-8B_ODESIA
- HiTZ/gemma-2b-it_ODESIA: https://huggingface.co/HiTZ/gemma-2b-it_ODESIA

## Run Evaluation/Inference

You must run this command before launching the scripts below
```
export PYTHONPATH="$PYTHONPATH:$PWD"
```

You can evaluate any model on the development set with the following command:

```bash
torchrun --standalone --nproc_per_node=1 src/evaluate.py --tasks all --model_name HiTZ/Qwen2.5-14B-Instruct_ODESIA --output_dir results/finetune/Qwen2.5-14B-Instruct

```

To reproduce our leaderboard results, you can run inference on the test sets using the following command. The resulting output files are ready to be submitted to the ODESIA challenge:

```bash
torchrun --standalone --nproc_per_node=1 src/inference.py --tasks all --model_name HiTZ/Qwen2.5-14B-Instruct_ODESIA --output_dir results/finetune/Qwen2.5-14B-Instruct
```

You can also run inference in selected tasks
```bash
torchrun --standalone --nproc_per_node=1 src/inference.py \
--tasks \
exist_2022_t1_es \
exist_2022_t2_es \
exist_2023_t1_es \
exist_2023_t2_es \
exist_2023_t3_es \
dipromats_2023_t1_es \
dipromats_2023_t2_es \
dipromats_2023_t3_es \
diann_2023_t1_es \
squad_2024_t1_es \
--model_name HiTZ/Qwen2.5-14B-Instruct_ODESIA \
--output_dir results/finetune/Qwen2.5-14B-Instruct
```


> Warning: The test sets do not contain the labels. If you want to evaluate the predictions, you should submit them to the ODESIA leaderboard [https://leaderboard.odesia.uned.es/leaderboard/challenge](https://leaderboard.odesia.uned.es/leaderboard/challenge) or use the PyEvAll library [https://github.com/UNEDLENAR/PyEvALL/tree/main](https://github.com/UNEDLENAR/PyEvALL/tree/main)

> Warning: We randomly sample few-shot examples from the train split for every input. These few-shot examples vary each evaluation run, so the evaluation results may change slightly  each time you run an evaluation. 
 

### 4-bit quantization
If you do not have enough VRAM to run a model, you can use 4-bit quantization by adding the `--quantization` flag to the previous commands. Example:


```bash
torchrun --standalone --nproc_per_node=1 src/inference.py --quantization --tasks all --model_name HiTZ/Qwen2.5-14B-Instruct_ODESIA --output_dir results/finetune/Qwen2.5-14B-Instruct 
```

> Warning: Quantization affects the model performance. Expect lower scores when running the model with 4 bit quantization. 
 

## Run Training

To finetune a model, you first need to define a `Training config`. Config examples for LLama3.1, Gemma and qwen using Full-Finetuning and LoRA are available in the [train_configs/](train_configs/) directory. Full-Finetuning will achieve slightly better results but requires a lot of VRAM (We use 4x A100 80GB for the 8B model and 8xA100 80GB for the 14B model). LoRA uses much less VRAM and supports model quantization, so it can be run on a single GPU. 

We use Deepspeed to split the model across the GPUs. You can reproduce our fine-tuning results with the following command (It will split the model across 8 GPUs)

```bash
export PYTHONPATH="$PYTHONPATH:$PWD"
accelerate launch --config_file train_configs/deepspeed_8.json src/train.py train_configs/qwen14B.yaml

```


If you want to run LoRA finetuning with a single GPU, you can use the following command:


```bash
python3 -m src.train train_configs/gemma2B_LoRa.yaml
```

If you want to enable model quantization, you can set `quantization:4` in the training configuration file. 

> Warning: Our inputs are very long, as we use many few-shot examples. Therefore, training requires a lot of VRAM and might be very slow. You can reduce the number of few-shot examples by modifying the __init__ default parameter for every task in [src/tasks](src/tasks). 
