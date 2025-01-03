import argparse
import logging
import os
import time
from typing import List

import outlines
import torch
from transformers import AutoTokenizer

from src.evaluate import evaluate_task
from src.tasks import get_tasks


def inference(
    tasks: List[str],
    model_name: str,
    output_dir: str = "results",
    quantization: bool = False,
):
    """
    Evaluates the model on the given tasks

    Args:
        tasks (List[str]): List of tasks to evaluate the model on
        model_name (str): Name of the model
    """
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Loading model {model_name}")
    start = time.time()

    if quantization:
        from transformers import BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    else:
        bnb_config = None

    model = outlines.models.transformers(
        model_name,
        device="cuda",  # optional device argument, default is cpu
        model_kwargs={
            "torch_dtype": torch.bfloat16,
            "attn_implementation": "flash_attention_2",
            "quantization_config": bnb_config,
        },  # optional model kwargs
    )

    logging.info(f"Model loaded in {time.time() - start:.2f} seconds")

    tasks_dict = get_tasks(
        tasks=tasks, tokenizer=AutoTokenizer.from_pretrained(model_name)
    )
    for task_name in tasks_dict.keys():
        logging.info(f"\n\n--- Inference task {task_name} ---")
        task = tasks_dict[task_name]

        # Evaluate the task
        _ = evaluate_task(task=task, model=model, output_dir=output_dir, split="test")


if __name__ == "__main__":
    # Set logging level to INFO
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["all"],
        help="List of tasks to evaluate the model on",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
        help="Name of the model",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/Llama-3.1-8B-Instruct/test",
        help="Output directory for the predictions",
    )

    parser.add_argument(
        "--quantization",
        action="store_true",
        help="Whether to use quantization",
    )

    args = parser.parse_args()

    inference(args.tasks, args.model_name, args.output_dir, args.quantization)
