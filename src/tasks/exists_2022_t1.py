import json
import os
import random
from enum import Enum
from typing import Dict, List

from pydantic import BaseModel

from src.tasks.task import Task

random.seed(33)


class Exists2022T1(Task):
    def __init__(
        self,
        num_examples_few_shot: int = 20,
        **kwargs,
    ):
        super().__init__(
            num_examples_few_shot=num_examples_few_shot,
            **kwargs,
        )
        self._precompute_examples()

    def get_system_prompt(self):
        """
        Returns the system prompt for the task
        """
        return "You are an AI assistant trained to identify sexist content in text. Your task is to analyze the given input and determine whether it contains sexist language or attitudes."

    def get_instruction(self):
        """
        Returns the guidelines for the task
        """
        return """
Analyze the given text to determine if it contains sexist content. 

Definition of sexism: Sexism is prejudice, stereotyping, or discrimination, typically against women, on the basis of sex. It includes attitudes, behaviors, and practices that promote stereotyping of social roles based on gender.

- 'sexist': The text expresses or implies sexist attitudes, stereotypes, or discriminatory views.
- 'non-sexist': The text does not contain any sexist language, attitudes, or implications.

Output: Provide your answer as a JSON object with the key 'label' and the value set to either 'sexist' or 'non-sexist'.

""".strip()

    def get_pydantic_model(self):
        """
        Returns the Pydantic model for the task output
        """

        class LabelEnum(str, Enum):
            sexist = "sexist"
            non_sexist = "non-sexist"

        class Identification(BaseModel):
            label: LabelEnum

        return Identification

    def _precompute_examples(self):
        """
        Divide the training examples into classes from which we will sample the few-shot examples.
        This allows to select a equal number of few-shot examples from each class
        """
        train_data = self.get_split("train")
        model = self.get_pydantic_model()
        # Change this line
        self.class_examples = {
            label.value: [] for label in model.model_fields["label"].annotation
        }
        for ex in train_data:
            self.class_examples[ex["answer"].label].append(ex)

    def get_few_shot(self):
        examples_per_class = self.num_examples_few_shot // len(self.class_examples)
        few_shot_examples = []
        for class_examples in self.class_examples.values():
            few_shot_examples.extend(
                random.sample(
                    class_examples, min(examples_per_class, len(class_examples))
                )
            )
        random.shuffle(few_shot_examples)
        return few_shot_examples

    def read_dataset(self, dataset: str):
        with open(dataset, "r", encoding="utf-8") as file:
            data = json.load(file)

        model = self.get_pydantic_model()
        return [
            {
                "question": item["text"],
                "answer": model(label=item["value"]) if "value" in item else None,
                "test_case": item["test_case"],
                "id": item["id"],
            }
            for item in data
        ]

    def evaluate(self, predictions: List[BaseModel], split="dev") -> Dict[str, float]:
        """
        Evaluates the prediction on the given split

        Args:
            prediction (str): Prediction string
            split (str, optional): Split to evaluate on. Defaults to "dev".

        Returns:
            Dict[str, float]: Dictionary with the evaluation metric
        """

        data = self.get_split(split)

        assert len(data) == len(predictions)

        acc = sum(
            prediction.label.lower().strip() == data[i]["answer"].label.lower().strip()
            for i, prediction in enumerate(predictions)
        ) / len(predictions)

        return {"accuracy": acc}

    def build_test_file(self, predictions: List[BaseModel], output_dir: str):
        """
        Builds a test file with the predictions

        Args:
            predictions (List[str]): List of predictions
            split (str, optional): Split to evaluate on. Defaults to "dev".
        """
        data = self.get_split("test")

        assert len(data) == len(predictions)

        test_data = [
            {
                "test_case": data[i]["test_case"],
                "id": data[i]["id"],
                "value": prediction.label.value,
            }
            for i, prediction in enumerate(predictions)
        ]
        self._write_json(os.path.join(output_dir, self.output_path), test_data)

    def build_validation_file(self, predictions: List[BaseModel], output_dir: str):
        """
        Builds a validation file with the predictions

        Args:
            predictions (List[str]): List of predictions
            split (str, optional): Split to evaluate on. Defaults to "dev".
        """

        data = self.get_split("dev")

        assert len(data) == len(predictions)

        for i, prediction in enumerate(predictions):
            data[i]["prediction"] = prediction.label.value
            data[i]["answer"] = data[i]["answer"].label.value

        output_path = os.path.join(
            output_dir,
            self.output_path.replace(".json", "_val.json"),
        )
        self._write_json(output_path, data)

    @staticmethod
    def _write_json(path: str, data: List[Dict]):
        with open(path, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=4, ensure_ascii=False)
