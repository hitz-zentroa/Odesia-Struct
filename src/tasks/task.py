from typing import Dict, List

import jinja2
from pydantic import BaseModel
from transformers import PreTrainedTokenizer


class Task:
    def __init__(
        self,
        train_dataset: str,
        dev_dataset: str,
        test_dataset: str,
        output_path: str,
        tokenizer: PreTrainedTokenizer,
        num_examples_few_shot: int = 20,
    ):
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset
        self.output_path = output_path
        self.num_examples_few_shot = num_examples_few_shot

        # Read all datasets during initialization
        self.train_data = self.read_dataset(self.train_dataset)
        self.dev_data = self.read_dataset(self.dev_dataset)
        self.test_data = self.read_dataset(self.test_dataset)

        # Pre-compile the Jinja2 template and instruction
        self.system_prompt = self.get_system_prompt()
        self.template = jinja2.Template(self.get_input_template())
        self.instruction = self.get_instruction()
        self.pydantic_model = self.get_pydantic_model()
        self.tokenizer = tokenizer

    def get_input_template(self):
        return """
{{ instruction }}

Examples
--------

{% for example in examples %}
Input: {{ example.question }}
Output: {{ example.answer.model_dump_json() }}

{% endfor %}

--------

Now, analyze the following input:

Input: {{ question }}
""".strip()

    def get_system_prompt() -> str:
        """
        Returns the system prompt for the task
        """
        raise NotImplementedError

    def get_instruction(self):
        raise NotImplementedError

    def get_pydantic_model(self):
        raise NotImplementedError

    def build_prompt(self, question: str, examples: List[Dict[str, str]], answer=None):
        user_input = self.template.render(
            instruction=self.instruction, question=question, examples=examples
        )

        conversation = [
            # {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_input},
        ]

        if answer is not None:
            conversation.append({"role": "assistant", "content": answer})

        try:
            prompt = self.tokenizer.apply_chat_template(
                conversation=conversation,
                add_generation_prompt=answer is None,
                tokenize=False,
            )
        except jinja2.exceptions.TemplateError:
            conversation = conversation[1:]
            prompt = self.tokenizer.apply_chat_template(
                conversation=conversation,
                add_generation_prompt=answer is None,
                tokenize=False,
            )

        return prompt

    def get_conversation(
        self, question: str, examples: List[Dict[str, str]], answer=None
    ):
        user_input = self.template.render(
            instruction=self.instruction, question=question, examples=examples
        )

        conversation = [
            # {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_input},
        ]

        if answer is not None:
            conversation.append({"role": "assistant", "content": answer})

        return conversation

    def read_dataset(self, dataset: str) -> List[Dict[str, str]]:
        """
        Reads the dataset and returns a dictionary with the fields question and answer

        Args:
            dataset (str): Path to the dataset

        Returns:
            List[Dict[str,str]]: List of dictionaries with the fields question, answer, test_case and id.
            If there are no answers, the value of the key Answer should be None
        """
        raise NotImplementedError

    def get_split(self, split="train"):
        if split == "train":
            return self.train_data
        elif split == "dev":
            return self.dev_data
        elif split == "test":
            return self.test_data
        else:
            raise ValueError("Split must be one of 'train', 'dev', or 'test'")

    def get_few_shot(self):
        raise NotImplementedError

    def get_dataset(self, split="train") -> List[Dict[str, str]]:
        """
        Returns the dataset for the given split. This dataset is ready for evaluation
        The prompt will not include the answer
        Args:
            split (str, optional): Split to return. Defaults to "train".
        Returns:
            List[Dict[str, str]]: List of dictionaries with the fields prompt, answer, test_case and id
        """
        data = self.get_split(split)
        dataset = []

        for i, example in enumerate(data):
            question = example["question"]
            answer = example["answer"]
            if answer is not None:
                answer = answer.model_dump_json()

            test_case = example["test_case"]
            id = example["id"]

            few_shot_examples = self.get_few_shot()

            few_shot_examples = [
                x for x in few_shot_examples if x["question"] != question
            ]

            # Use pre-compiled instruction
            prompt = self.build_prompt(
                question=question,
                examples=few_shot_examples,
            )

            dataset.append(
                {"prompt": prompt, "answer": answer, "test_case": test_case, "id": id}
            )

        return dataset

    def get_dataset_training(self, split="train") -> List[Dict[str, str]]:
        """
        Returns the dataset for the given split. This dataset is ready for training
        The prompt will include the answer
        Args:
            split (str, optional): Split to return. Defaults to "train".
        Returns:
            List[Dict[str, str]]: List of conversations
        """
        data = self.get_split(split)
        dataset = []

        for i, example in enumerate(data):
            question = example["question"]
            answer = example["answer"]
            if answer is not None:
                answer = answer.model_dump_json()

            few_shot_examples = self.get_few_shot()

            few_shot_examples = [
                x for x in few_shot_examples if x["question"] != question
            ]

            # Use pre-compiled instruction
            prompt = self.get_conversation(
                question=question,
                examples=few_shot_examples,
                answer=answer,
            )

            dataset.append({"messages": prompt})

        return dataset

    def evaluate(self, predictions: List[BaseModel], split="dev") -> Dict[str, float]:
        """
        Evaluates the prediction on the given split

        Args:
            prediction (str): Prediction string
            split (str, optional): Split to evaluate on. Defaults to "dev".

        Returns:
            Dict[str, float]: Dictionary with the evaluation metric
        """
        raise NotImplementedError

    def build_test_file(self, predictions: List[BaseModel], split="dev"):
        """
        Builds a test file with the predictions

        Args:
            predictions (List[str]): List of predictions
            split (str, optional): Split to evaluate on. Defaults to "dev".
        """
        raise NotImplementedError

    def build_validation_file(self, predictions: List[BaseModel], output_dir: str):
        """
        Builds a validation file with the predictions

        Args:
            predictions (List[str]): List of predictions
            split (str, optional): Split to evaluate on. Defaults to "dev".
        """

        raise NotImplementedError
