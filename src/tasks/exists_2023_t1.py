import json
import os
import random
from typing import Dict, List

import torch
import torch.nn.functional as F
from pydantic import BaseModel, Field, field_validator

from src.tasks.task import Task

random.seed(33)


class Exists2023T1(Task):
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

- 'Sexist': The text expresses or implies sexist attitudes, stereotypes, or discriminatory views.
- 'Non-Sexist': The text does not contain any sexist language, attitudes, or implications.

Output: Provide your answer as a JSON object with the probabilities for each category between 0 and 1.

""".strip()

    def get_pydantic_model(self):
        """
        Returns the Pydantic model for the task output
        """

        class Identification(BaseModel):
            sexist: float = Field(
                ...,
                ge=0,
                le=1,
            )
            non_sexist: float = Field(
                ...,
                ge=0,
                le=1,
            )

            @field_validator("sexist", "non_sexist", mode="before")
            @classmethod
            def clamp_values(cls, v):
                return max(0, min(v, 1))

        return Identification

    def _precompute_examples(self):
        """
        Divide the training examples into classes from which we will sample the few-shot examples.
        This allows to select a equal number of few-shot examples from each class
        """
        train_data = self.get_split("train")
        self.sexist_examples = []
        self.non_sexist_examples = []
        for example in train_data:
            if example["answer"].sexist > 0.5:
                self.sexist_examples.append(example)
            else:
                self.non_sexist_examples.append(example)

    def get_few_shot(self):
        num_examples = self.num_examples_few_shot
        num_sexist = num_examples // 2
        num_non_sexist = num_examples - num_sexist

        sexist_samples = random.sample(
            self.sexist_examples, min(num_sexist, len(self.sexist_examples))
        )
        non_sexist_samples = random.sample(
            self.non_sexist_examples, min(num_non_sexist, len(self.non_sexist_examples))
        )

        few_shot_examples = sexist_samples + non_sexist_samples
        random.shuffle(few_shot_examples)

        return few_shot_examples

    def _process_answer(self, answer: List[str]):
        yes_count = sum(1 for ans in answer if ans.lower() == "yes")
        total_count = len(answer)
        sexist_prob = round(yes_count / total_count, 2)
        non_sexist_prob = round(1 - sexist_prob, 2)
        return {"sexist": sexist_prob, "non_sexist": non_sexist_prob}

    def read_dataset(self, dataset: str):
        with open(dataset, "r", encoding="utf-8") as file:
            data = json.load(file)

        model = self.get_pydantic_model()

        return [
            {
                "question": item["tweet"] if "tweet" in item else item["text"],
                "answer": model(**self._process_answer(item["value"]))
                if "value" in item
                else None,
                "test_case": item["test_case"],
                "id": item["id"],
            }
            for item in data
        ]

    def _normalize_prediction(self, prediction: BaseModel) -> BaseModel:
        """
        Normalizes the prediction probabilities to ensure they sum to 1.
        """
        model = self.get_pydantic_model()
        total = prediction.sexist + prediction.non_sexist
        return model(
            sexist=round(prediction.sexist / total, 2),
            non_sexist=round(prediction.non_sexist / total, 2),
        )

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

        correct = 0
        total_loss = 0.0

        for i, prediction in enumerate(predictions):
            normalized_prediction = self._normalize_prediction(prediction)
            true_label = data[i]["answer"].sexist > 0.5
            pred_label = normalized_prediction.sexist > 0.5

            if true_label == pred_label:
                correct += 1

            # Compute cross-entropy loss
            true_probs = torch.tensor(
                [data[i]["answer"].sexist, data[i]["answer"].non_sexist]
            )
            pred_probs = torch.tensor(
                [normalized_prediction.sexist, normalized_prediction.non_sexist]
            )
            loss = F.cross_entropy(pred_probs.unsqueeze(0), true_probs.unsqueeze(0))
            total_loss += loss.item()

        accuracy = correct / len(predictions)
        avg_loss = total_loss / len(predictions)

        return {"accuracy": accuracy, "cross_entropy_loss": avg_loss}

    def build_test_file(self, predictions: List[BaseModel], output_dir: str):
        """
        Builds a test file with the predictions

        Args:
            predictions (List[str]): List of predictions
            split (str, optional): Split to evaluate on. Defaults to "dev".
        """
        data = self.get_split("test")
        assert len(data) == len(predictions)

        predictions = [
            self._normalize_prediction(prediction) for prediction in predictions
        ]
        test_data = [
            {
                "test_case": data[i]["test_case"],
                "id": data[i]["id"],
                "value": {
                    "YES": prediction.sexist,
                    "NO": prediction.non_sexist,
                },
            }
            for i, prediction in enumerate(predictions)
        ]
        self._write_json(os.path.join(output_dir, self.output_path), test_data)

    def build_validation_file(self, predictions: List[BaseModel], output_dir: str):
        data = self.get_split("dev")
        assert len(data) == len(predictions)

        for i, prediction in enumerate(predictions):
            data[i]["prediction"] = {
                "YES": prediction.sexist,
                "NO": prediction.non_sexist,
            }
            data[i]["answer"] = {
                "YES": data[i]["answer"].sexist,
                "NO": data[i]["answer"].non_sexist,
            }

        output_path = os.path.join(
            output_dir,
            self.output_path.replace(".json", "_val.json"),
        )
        self._write_json(output_path, data)

    @staticmethod
    def _write_json(path: str, data: List[Dict]):
        with open(path, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=4, ensure_ascii=False)
