import logging
import os
import sys

import torch.distributed as dist
from transformers import HfArgumentParser, Seq2SeqTrainingArguments

from src.training.config import ModelArguments
from src.training.dataset import DataCollatorForOdesia, OdesiaDataset
from src.training.load_model import load_model, merge_lora_model
from src.training.trainer import OdesiaTrainer


def train(training_args: Seq2SeqTrainingArguments, model_args: ModelArguments):
    """
    Train the model

    Args:
        training_args (Seq2SeqTrainingArguments): Training arguments
        model_args (ModelArguments): Model arguments
    """
    os.makedirs(training_args.output_dir, exist_ok=True)

    # Load model only on the main process

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
        trust_remote_code=model_args.trust_remote_code,
        use_flash_attention=model_args.use_flash_attention,
        fsdp_training=len(training_args.fsdp) > 1
        or training_args.fsdp_config is not None,
        max_memory_MB=model_args.max_memory_MB,
        rope_scaling_factor=model_args.rope_scaling_factor,
        #use_liger_kernel=training_args.use_liger_kernel,
    )

    print(f"Model_max_length: {tokenizer.model_max_length}")

    # Load dataset only on the main process
    if training_args.local_rank == 0:
        train_dataset = OdesiaDataset(tokenizer=tokenizer, split="train")
        validation_dataset = OdesiaDataset(tokenizer=tokenizer, split="dev")
    else:
        train_dataset = None
        validation_dataset = None

    # Broadcast the datasets to all processes
    if dist.is_initialized():
        object_list = [train_dataset, validation_dataset]
        dist.broadcast_object_list(object_list, src=0)
        train_dataset, validation_dataset = object_list

    trainer = OdesiaTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=DataCollatorForOdesia(tokenizer=tokenizer),
    )

    trainer.train()

    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")

    trainer.save_model()


def merge_lora(training_args: Seq2SeqTrainingArguments, model_args: ModelArguments):
    merge_lora_model(
        weights_path=model_args.model_name_or_path,
        lora_weights_name_or_path=training_args.output_dir,
        output_path=training_args.output_dir,
        torch_dtype=model_args.torch_dtype,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = HfArgumentParser((ModelArguments, Seq2SeqTrainingArguments))
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

    if model_args.use_lora:
        # Check if this is the main process (rank 0)
        if not dist.is_initialized() or dist.get_rank() == 0:
            merge_lora(training_args, model_args)

        # If using distributed training, wait for all processes to finish
        if dist.is_initialized():
            dist.barrier()
