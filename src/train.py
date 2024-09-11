import json
import logging
import os
import sys

from datasets import load_dataset
from transformers import HfArgumentParser
from trl import SFTConfig, SFTTrainer, setup_chat_format

from src.config.config import ModelArguments
from src.model.load_model import load_model, merge_lora_model
from src.tasks import get_tasks


def train(training_args: SFTConfig, model_args: ModelArguments):
    model, tokenizer = load_model(
        inference=False,
        model_weights_name_or_path=model_args.model_name_or_path,
        quantization=model_args.quantization,
        use_lora=model_args.use_lora,
        lora_r=model_args.lora_r,
        lora_target_modules=model_args.lora_target_modules,
        torch_dtype=model_args.torch_dtype,
        force_auto_device_map=model_args.force_auto_device_map,
        use_gradient_checkpointing=training_args.gradient_checkpointing,
        use_better_transformer=model_args.use_better_transformer,
        trust_remote_code=model_args.trust_remote_code,
        use_flash_attention=model_args.use_flash_attention,
        fsdp_training=len(training_args.fsdp) > 1
        or training_args.fsdp_config is not None,
        max_memory_MB=model_args.max_memory_MB,
        rope_scaling_factor=model_args.rope_scaling_factor,
    )

    tasks = get_tasks(tokenizer=tokenizer, tasks="all")

    train_dataset = []
    validation_dataset = []
    for task in tasks:
        train_dataset.extend(task.get_dataset_training(split="train"))
        validation_dataset.extend(task.get_dataset_training(split="dev"))

    with open(os.path.join(training_args.output_dir, "train_dataset.jsonl"), "w") as f:
        for example in train_dataset:
            print(json.dumps(example, ensure_ascii=False), file=f)

    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(validation_dataset)}")

    with open(
        os.path.join(training_args.output_dir, "validation_dataset.jsonl"), "w"
    ) as f:
        for example in validation_dataset:
            print(json.dumps(example, ensure_ascii=False), file=f)

    model, tokenizer = setup_chat_format(model, tokenizer)
    dataset = load_dataset(
        "json",
        data_files={
            "train": os.path.join(training_args.output_dir, "train_dataset.jsonl"),
            "validation": os.path.join(
                training_args.output_dir, "validation_dataset.jsonl"
            ),
        },
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset,
    )

    trainer.train()

    merge_lora_model(
        weights_path=model_args.model_name_or_path,
        lora_weights_name_or_path=training_args.output_dir,
        output_path=training_args.output_dir,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = HfArgumentParser((ModelArguments, SFTConfig))
    logging.info(f"Sys args {sys.argv}")

    if len(sys.argv) > 0 and sys.argv[-1].endswith(".json"):
        # If we pass only one argument to the script, and it's the path to a json file,
        # let's parse it to get our arguments.
        logging.info(f"Loading json config {sys.argv[-1]}")
        model_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[-1])
        )

    elif len(sys.argv) > 0 and sys.argv[-1].endswith(".yaml"):
        # If we pass only one argument to the script, and it's the path to a yaml file,
        # let's parse it to get our arguments.
        logging.info(f"Loading yaml config {sys.argv[-1]}")
        model_args, training_args = parser.parse_yaml_file(
            yaml_file=os.path.abspath(sys.argv[-1])
        )
    else:
        logging.info("No config file passed, using command line arguments.")
        model_args, training_args = parser.parse_args_into_dataclasses()

    train(training_args, model_args)
