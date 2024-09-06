import json
import logging
import os
import time
from typing import List

import outlines
import torch
from accelerate import find_executable_batch_size
from tqdm import tqdm

from src.tasks import tasks_dict
from src.tasks.task import Task


@torch.no_grad()
@find_executable_batch_size(starting_batch_size=64)
def evaluate_task(
    batch_size: int, task: Task, model: outlines.models.transformers, split="dev"
):
    """
    Runs the task on the model

    Args:
        task (Task): Task object
        model (outlines.models.transformers): Model object
        batch_size (int): Batch size
    """
    logging.info(f"Evaluating on dev set with batch size {batch_size}")
    # Get the dev dataset
    dev_data = task.get_dataset(split)
    num_batches = len(dev_data) // batch_size
    generator = outlines.generate.json(model, task.get_pydantic_model())
    predictions = []
    first = batch_size == 64
    with tqdm(total=num_batches, desc="Evaluating") as pbar:
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
        logging.warning(f"Dev metric: {metric:.2f}")
        task.build_validation_file(predictions)
    else:
        task.build_test_file(predictions)
        metric = None
    return metric


def evaluate(tasks: List[str], model_name: str):
    """
    Evaluates the model on the given tasks

    Args:
        tasks (List[str]): List of tasks to evaluate the model on
        model_name (str): Name of the model
        model_path (str): Path to the model
    """

    logging.info(f"Loading model {model_name}")
    start = time.time()
    model = outlines.models.transformers(
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        device="cuda",  # optional device argument, default is cpu
        model_kwargs={
            "torch_dtype": torch.bfloat16,
            "attn_implementation": "flash_attention_2",
        },  # optional model kwargs
    )

    logging.info(f"Model loaded in {time.time() - start:.2f} seconds")
    metric_dict = {}
    for task_name in tasks:
        logging.info(f"\n\n--- Evaluating task {task_name} ---")
        task = tasks_dict[task_name]

        # Evaluate the task
        metric = evaluate_task(task, model)
        metric_dict[task_name] = metric

    os.makedirs("results", exist_ok=True)
    with open(f"results/{model_name.replace("/","_")}.json", "w") as f:
        json.dump(metric_dict, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    # Set logging level to INFO
    logging.basicConfig(level=logging.INFO)

    evaluate(
        ["exists_2023_t3_es"],
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
    )
