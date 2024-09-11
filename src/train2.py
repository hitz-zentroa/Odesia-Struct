import logging
import os
import sys

from datasets import load_dataset
from transformers import AutoTokenizer, HfArgumentParser
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel  # Also reqires pip install xformers

from src.config.config import ModelArguments
from src.model.model_utils import find_all_linear_names


def train(training_args: SFTConfig, model_args: ModelArguments):
    os.makedirs(training_args.output_dir, exist_ok=True)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_args.model_name_or_path,
        dtype=None,  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
        load_in_4bit=True,  # Use 4bit quantization to reduce memory usage. Can be False
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path
    )  # FastLanguageModel doesn't load the chat template

    model = FastLanguageModel.get_peft_model(
        model,
        r=model_args.lora_r,
        target_modules=find_all_linear_names(model),
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,  # Dropout = 0 is currently optimized
        bias="none",  # Bias = "none" is currently optimized
        use_gradient_checkpointing=True,
        random_state=3407,
    )

    """
    tasks = get_tasks(tokenizer=tokenizer, tasks="all")

    train_dataset = []
    validation_dataset = []
    for task_name, task in tqdm(tasks.items(), desc="Loading datasets"):
        print(f"Loading dataset for task {task_name}")
        train = task.get_dataset_training(split="train")
        train_dataset.extend(train)
        print(f"Train dataset size: {len(train)}")
        dev = task.get_dataset_training(split="dev")
        validation_dataset.extend(dev)
        print(f"Validation dataset size: {len(dev)}")

    with open(os.path.join(training_args.output_dir, "train_dataset.jsonl"), "w") as f:
        for example in train_dataset:
            print(json.dumps(example, ensure_ascii=False), file=f)

    print(f"Full training dataset size: {len(train_dataset)}")
    print(f"Full validation dataset size: {len(validation_dataset)}")


    with open(
        os.path.join(training_args.output_dir, "validation_dataset.jsonl"), "w"
    ) as f:
        for example in validation_dataset:
            print(json.dumps(example, ensure_ascii=False), file=f)
    """
    print("Loading datasets")
    dataset = load_dataset(
        "json",
        data_files={
            "train": os.path.join(training_args.output_dir, "train_dataset.jsonl"),
            "validation": os.path.join(
                training_args.output_dir, "validation_dataset.jsonl"
            ),
        },
    )

    print(dataset)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
    )

    trainer.train()

    trainer.save_model()


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
