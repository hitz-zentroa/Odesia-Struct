from typing import Dict, List

import jinja2
from pydantic import BaseModel


class Task:
    def __init__(
        self,
        train_dataset: str,
        dev_dataset: str,
        test_dataset: str,
        output_path: str,
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
        self.template = jinja2.Template(self.prompt())
        self.instruction = self.get_instruction()
        self.pydantic_model = self.get_pydantic_model()

    def prompt() -> jinja2.Template:
        """
        Returns the jinja template for the prompt
        """
        raise NotImplementedError

    def get_instruction(self):
        raise NotImplementedError

    def get_pydantic_model(self):
        raise NotImplementedError

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
            prompt = self.template.render(
                instruction=self.instruction,
                question=question,
                examples=few_shot_examples,
            )

            dataset.append(
                {"prompt": prompt, "answer": answer, "test_case": test_case, "id": id}
            )

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

    def build_validation_file(self, predictions: List[BaseModel]):
        """
        Builds a validation file with the predictions

        Args:
            predictions (List[str]): List of predictions
            split (str, optional): Split to evaluate on. Defaults to "dev".
        """

        raise NotImplementedError
