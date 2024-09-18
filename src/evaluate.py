import argparse
import json
import logging
import os
import time
from typing import List

import outlines
import torch
from accelerate import find_executable_batch_size
from tqdm import tqdm
from transformers import AutoTokenizer

from src.tasks import get_tasks
from src.tasks.task import Task


@torch.no_grad()
@find_executable_batch_size(starting_batch_size=64)
def evaluate_task(
    batch_size: int,
    task: Task,
    model: outlines.models.transformers,
    split="dev",
    output_dir: str = "results",
):
    """
    Runs the model on the given task
    Args:
        task (Task): Task object
        model (outlines.models.transformers): Model object
        batch_size (int): Batch size
    """
    logging.info(f"Evaluating on {split} set with batch size {batch_size}")
    # Get the dev dataset
    dev_data = task.get_dataset(split)
    num_batches = len(dev_data) // batch_size
    generator = outlines.generate.json(
        model, task.get_pydantic_model(), whitespace_pattern=""
    )
    predictions = []
    first = batch_size == 64
    with tqdm(
        total=num_batches, desc="Evaluating" if split == "dev" else "Inference"
    ) as pbar:
        for i in range(0, len(dev_data), batch_size):
            batch = dev_data[i : i + batch_size]
            inputs = [item["prompt"] for item in batch]
            if first:
                logging.info("Prompt sample:")
                logging.info(inputs[0])
                first = False
            outputs = generator(inputs)
            predictions.extend(outputs)
            pbar.update(1)

    if split == "dev":
        metric = task.evaluate(predictions, "dev")
        logging.warning(f"Dev metric: {metric}")
        task.build_validation_file(predictions, output_dir=output_dir)
    else:
        task.build_test_file(predictions, output_dir=output_dir)
        metric = None
    return metric


def evaluate(
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

    logging.info(f"Model {model.model} loaded in {time.time() - start:.2f} seconds")
    metric_dict = {}

    tasks_dict = get_tasks(
        tasks=tasks, tokenizer=AutoTokenizer.from_pretrained(model_name)
    )
    for task_name in tasks_dict.keys():
        logging.info(f"\n\n--- Evaluating task {task_name} ---")
        task = tasks_dict[task_name]

        # Evaluate the task
        metric = evaluate_task(task=task, model=model, output_dir=output_dir)
        metric_dict[task_name] = metric

    with open(os.path.join(output_dir, "dev_metrics.json"), "w") as f:
        json.dump(metric_dict, f, indent=4, ensure_ascii=False)


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
        default="results/Llama-3.1-8B-Instruct",
        help="Output directory for the predictions",
    )

    parser.add_argument(
        "--quantization",
        action="store_true",
        help="Whether to use quantization",
    )

    args = parser.parse_args()

    evaluate(args.tasks, args.model_name, args.output_dir, args.quantization)
