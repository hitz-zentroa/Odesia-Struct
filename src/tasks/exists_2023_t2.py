import json
import os
import random
from typing import Dict, List

import torch
import torch.nn.functional as F
from pydantic import BaseModel, Field

from src.tasks.task import Task

random.seed(33)


class Exists2023T2(Task):
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
        return "You are an AI assistant trained to identify and classify sexist content in text. Your task is to analyze the given input and determine whether it contains sexist language or attitudes, and if so, to classify it into specific categories."

    def get_instruction(self):
        return """
Analyze the given text to determine if it contains sexist content. If it does, classify it into one of the following categories based on the intention of the author 'direct', 'reported', 'judgemental' or 'non-sexist' if it doesn't contain sexist content.

Definition of sexism: Sexism is prejudice, stereotyping, or discrimination, typically against women, on the basis of sex. It includes attitudes, behaviors, and practices that promote stereotyping of social roles based on gender.

- 'non-sexist': The text does not contain any sexist language, attitudes, or implications.
- 'direct': The intention was to write a message that is sexist by itself or incites to be sexist
- 'reported': The intention is to report and share a sexist situation suffered by a woman or women in first or third person.
- 'judgemental': The intention was to judge, since the tweet describes sexist situations or behaviours with the aim of condemning them.

Output: Provide your answer as a JSON object with the key 'label' and the value set to one of the categories listed above.

""".strip()

    def get_pydantic_model(self):
        class Identification(BaseModel):
            non_sexist: float = Field(
                ...,
                ge=0,
                le=1,
            )
            direct: float = Field(
                ...,
                ge=0,
                le=1,
            )
            reported: float = Field(
                ...,
                ge=0,
                le=1,
            )
            judgemental: float = Field(
                ...,
                ge=0,
                le=1,
            )

        return Identification

    def _precompute_examples(self):
        train_data = self.get_split("train")
        self.examples = {
            "non_sexist": [],
            "direct": [],
            "reported": [],
            "judgemental": [],
        }
        for example in train_data:
            # Use a threshold to determine the label
            threshold = 0.5
            for label, value in example["answer"].__dict__.items():
                if value >= threshold:
                    self.examples[label].append(example)
                    break
            else:
                # If no label meets the threshold, use the max value
                label = max(example["answer"].__dict__.items(), key=lambda x: x[1])[0]
                self.examples[label].append(example)

    def get_few_shot(self):
        num_examples = self.num_examples_few_shot
        examples_per_class = num_examples // 4
        remaining = num_examples % 4

        few_shot_examples = []
        for label in self.examples:
            samples = random.sample(
                self.examples[label],
                min(examples_per_class, len(self.examples[label])),
            )
            few_shot_examples.extend(samples)

        # Add remaining examples randomly
        all_examples = [ex for examples in self.examples.values() for ex in examples]
        few_shot_examples.extend(random.sample(all_examples, remaining))

        random.shuffle(few_shot_examples)
        return few_shot_examples

    def _process_answer(self, answer: List[str]):
        total_count = len(answer)
        counts = {"-": 0, "direct": 0, "reported": 0, "judgemental": 0}
        for item in answer:
            item = item.lower()
            if item in counts:
                counts[item] += 1

        probs = {
            "non_sexist": counts["-"] / total_count,
            "direct": counts["direct"] / total_count,
            "reported": counts["reported"] / total_count,
            "judgemental": counts["judgemental"] / total_count,
        }

        # Round to 2 decimal places
        rounded_probs = {k: round(v, 2) for k, v in probs.items()}

        # Ensure probabilities sum to 1
        total = sum(rounded_probs.values())
        if total != 1:
            diff = 1 - total
            max_key = max(rounded_probs, key=rounded_probs.get)
            rounded_probs[max_key] = round(rounded_probs[max_key] + diff, 2)

        return rounded_probs

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
        Normalizes the prediction probabilities using softmax.
        """
        model = self.get_pydantic_model()
        probs = torch.tensor(
            [
                prediction.non_sexist,
                prediction.direct,
                prediction.reported,
                prediction.judgemental,
            ]
        )
        normalized_probs = F.softmax(probs, dim=0)
        return model(
            non_sexist=normalized_probs[0].item(),
            direct=normalized_probs[1].item(),
            reported=normalized_probs[2].item(),
            judgemental=normalized_probs[3].item(),
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
            true_label = max(data[i]["answer"].__dict__.items(), key=lambda x: x[1])[0]
            pred_label = max(
                normalized_prediction.__dict__.items(), key=lambda x: x[1]
            )[0]

            if true_label == pred_label:
                correct += 1

            # Calculate cross-entropy loss
            true_probs = torch.tensor(
                [
                    getattr(data[i]["answer"], k)
                    for k in ["non_sexist", "direct", "reported", "judgemental"]
                ]
            )
            pred_probs = torch.tensor(
                [
                    getattr(normalized_prediction, k)
                    for k in ["non_sexist", "direct", "reported", "judgemental"]
                ]
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
                    "NO": predictions[i].non_sexist,
                    "DIRECT": predictions[i].direct,
                    "REPORTED": predictions[i].reported,
                    "JUDGEMENTAL": predictions[i].judgemental,
                },
            }
            for i, prediction in enumerate(predictions)
        ]
        self._write_json(os.path.join(output_dir, self.output_path), test_data)

    def build_validation_file(self, predictions: List[BaseModel], output_dir: str):
        data = self.get_split("dev")
        assert len(data) == len(predictions)

        for i, prediction in enumerate(predictions):
            prediction = self._normalize_prediction(prediction)
            data[i]["prediction"] = {
                "NO": prediction.non_sexist,
                "DIRECT": prediction.direct,
                "REPORTED": prediction.reported,
                "JUDGEMENTAL": prediction.judgemental,
            }
            data[i]["answer"] = {
                "NO": data[i]["answer"].non_sexist,
                "DIRECT": data[i]["answer"].direct,
                "REPORTED": data[i]["answer"].reported,
                "JUDGEMENTAL": data[i]["answer"].judgemental,
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
