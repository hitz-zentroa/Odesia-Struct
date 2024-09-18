from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from itertools import islice
from typing import Any, Dict, List, Optional, Union

import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import BatchEncoding, PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy

from src.tasks import get_tasks


def prepare_data(
    tokenizer: PreTrainedTokenizerBase, conversation: List[Dict[str, str]]
) -> BatchEncoding:
    """
    Prepare the data to be feeded into the model.

    Args:
        tokenizer (`PreTrainedTokenizerBase`):
            The tokenizer to use.
      conversation (`List[Dict[str]]`):
            The conversation to prepare.


    Returns:
        `BatchEncoding`: `BatchEncoding` with the prepared data.

    """

    conversation = conversation["messages"]

    formatted_prompt = tokenizer.apply_chat_template(
        conversation=conversation[:-1],
        add_generation_prompt=True,
        tokenize=False,
    )

    formatted_input = tokenizer.apply_chat_template(
        conversation=conversation,
        tokenize=False,
    )

    prompt_tokens = tokenizer(
        text=formatted_prompt,
        max_length=tokenizer.model_max_length,
        truncation=True,
        padding=False,
        return_tensors=None,
        add_special_tokens=False,
    )

    input_tokens = tokenizer(
        text=formatted_input,
        max_length=tokenizer.model_max_length,
        truncation=True,
        padding=False,
        return_tensors=None,
        add_special_tokens=False,
    )

    prompt_length = len(prompt_tokens["input_ids"])
    input_length = len(input_tokens["input_ids"])

    loss_weight_mask = np.ones(input_length, dtype=np.float32)
    loss_weight_mask[:prompt_length] = 0.0

    input_tokens["loss_weight_mask"] = loss_weight_mask
    input_tokens["labels"] = input_tokens["input_ids"].copy()

    return input_tokens


class OdesiaDataset(Dataset):
    """
    Dataset for the Odesia training.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        split: str,
        batch_size: int = 512,
        max_examples: int = 10000,
        tasks: List[str] = ["all"],
    ):
        """
        Args:
            tokenizer (`PreTrainedTokenizerBase`):
                The tokenizer to use.
            split (`str`):
                The split to use. Can be `train`, `dev`, or `test`.
            batch_size (`int`, optional):
                The batch size to use.
            max_examples (`int`, optional):
                The maximum number of examples to use per tasks
            tasks (`List[str]`, optional):
                The tasks to use. Can be `all` or a list of tasks to use.
        """

        print(f"Loading dataset for split {split}...")
        tasks = get_tasks(tokenizer=tokenizer, tasks=tasks)

        # Function to get dataset for a single task
        def get_task_dataset(task):
            data = task.get_dataset_training(split=split)
            return data

        # Function to prepare data for a batch of examples
        def prepare_batch(batch):
            return [
                prepare_data(tokenizer=tokenizer, conversation=example)
                for example in batch
            ]

        # Use ThreadPoolExecutor for parallel processing of tasks
        with ThreadPoolExecutor() as executor:
            # Get all datasets in parallel
            futures = [
                executor.submit(get_task_dataset, task) for task in tasks.values()
            ]
            all_task_data = []
            for future in tqdm(
                as_completed(futures),
                total=len(tasks),
                desc=f"Retrieving datasets for split {split}",
            ):
                task_data = future.result()
                # Limit the number of examples per task. Randomly sample if needed.
                if len(task_data) > max_examples:
                    # Sort task_data based on the length of the 'content' in each message
                    task_data = sorted(
                        task_data,
                        key=lambda x: sum(len(msg["content"]) for msg in x["messages"]),
                    )
                    # Select the first max_examples
                    task_data = task_data[:max_examples]

                all_task_data.extend(task_data)

            # Prepare all data in parallel
            self.dataset = []
            futures = []
            for i in range(0, len(all_task_data), batch_size):
                batch = list(islice(all_task_data, i, i + batch_size))
                futures.append(executor.submit(prepare_batch, batch))

            self.dataset = []
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"Preparing data for split {split}",
            ):
                prepared_batch = future.result()
                self.dataset.extend(
                    [data for data in prepared_batch if data is not None]
                )

        print(f"Dataset for split {split} has {len(self.dataset)} examples.")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx].copy()


@dataclass
class DataCollatorForOdesia:
    """
    Adapted from transformers.DataCollatorForSeq2Seq to handle CoLLIE data.

    Data collator that will dynamically pad the inputs received, as well as the labels.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        model ([`PreTrainedModel`]):
            The model that is being trained. If set and has the *prepare_decoder_input_ids_from_labels*, use it to
            prepare the *decoder_input_ids*

            This is useful when using *label_smoothing* to avoid calculating loss twice.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            - `True` or `'longest'` (default): Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'`: No padding (i.e., can output a batch with sequences of different lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (`int`, *optional*, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignored by PyTorch loss functions).
        return_tensors (`str`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """

    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors
        labels = (
            [feature["labels"] for feature in features]
            if "labels" in features[0].keys()
            else None
        )
        loss_weight_mask = (
            [feature["loss_weight_mask"] for feature in features]
            if "loss_weight_mask" in features[0].keys()
            else None
        )
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                    (max_label_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )

            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (
                    max_label_length - len(feature["labels"])
                )
                if isinstance(feature["labels"], list):
                    feature["labels"] = (
                        feature["labels"] + remainder
                        if padding_side == "right"
                        else remainder + feature["labels"]
                    )
                elif padding_side == "right":
                    feature["labels"] = np.concatenate(
                        [feature["labels"], remainder]
                    ).astype(np.int64)
                else:
                    feature["labels"] = np.concatenate(
                        [remainder, feature["labels"]]
                    ).astype(np.int64)

        if loss_weight_mask is not None:
            max_loss_weight_mask_length = max(len(l) for l in loss_weight_mask)
            if self.pad_to_multiple_of is not None:
                max_loss_weight_mask_length = (
                    (max_loss_weight_mask_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )

            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [0.0 if self.label_pad_token_id == -100 else 1.0] * (
                    max_loss_weight_mask_length - len(feature["loss_weight_mask"])
                )
                if isinstance(feature["loss_weight_mask"], list):
                    feature["loss_weight_mask"] = (
                        feature["loss_weight_mask"] + remainder
                        if padding_side == "right"
                        else remainder + feature["loss_weight_mask"]
                    )
                elif padding_side == "right":
                    feature["loss_weight_mask"] = np.concatenate(
                        [feature["loss_weight_mask"], remainder]
                    ).astype(np.float32)
                else:
                    feature["loss_weight_mask"] = np.concatenate(
                        [remainder, feature["loss_weight_mask"]]
                    ).astype(np.float32)

        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        # prepare decoder_input_ids
        if (
            labels is not None
            and self.model is not None
            and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        ):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(
                labels=features["labels"]
            )
            features["decoder_input_ids"] = decoder_input_ids

        return features
