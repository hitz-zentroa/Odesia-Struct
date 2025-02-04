import json
import os
import random
from typing import Dict, List

import torch
import torch.nn.functional as F
from pydantic import BaseModel, Field, field_validator

from src.tasks.task import Task

random.seed(33)


class Exists2023T3(Task):
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
        return "You are an AI assistant trained to identify and classify sexist content in text. Your task is to analyze the given input and determine whether it contains sexist language or attitudes, and if so, to classify it into specific categories."

    def get_instruction(self):
        """
        Returns the guidelines for the task
        """
        return """
Analyze the given text to determine if it contains sexist content. If it does, classify it into one of the following categories: 'ideological-inequality', 'misogyny-non-sexual-violence', 'objectification', 'sexual-violence', 'stereotyping-dominance'. If it doesn't contain sexist content, classify it as 'non-sexist'.

Definition of sexism: Sexism is prejudice, stereotyping, or discrimination, typically against women, on the basis of sex. It includes attitudes, behaviors, and practices that promote stereotyping of social roles based on gender.

- 'non-sexist': The text does not contain any sexist language, attitudes, or implications.
- 'ideological-inequality': The text discredits feminism, denies gender inequality, or portrays men as victims of gender-based oppression.
- 'stereotyping-dominance': The text expresses stereotypes about gender roles or claims superiority of one gender over another.
- 'objectification': The text reduces women to their physical attributes or treats them as objects for male pleasure.
- 'sexual-violence': The text contains sexual harassment, assault, or rape-related content.
- 'misogyny-non-sexual-violence': The text expresses hatred, contempt, or non-sexual violence towards women.

Output: Provide your answer as a JSON object with the probabilities for each category lised above between 0 and 1.

""".strip()

    def get_pydantic_model(self):
        """
        Returns the Pydantic model for the task output
        """

        class Identification(BaseModel):
            non_sexist: float = Field(
                ...,
                ge=0,
                le=1,
            )
            ideological_inequality: float = Field(
                ...,
                ge=0,
                le=1,
            )
            stereotyping_dominance: float = Field(
                ...,
                ge=0,
                le=1,
            )
            objectification: float = Field(
                ...,
                ge=0,
                le=1,
            )
            sexual_violence: float = Field(
                ...,
                ge=0,
                le=1,
            )
            misogyny_non_sexual_violence: float = Field(
                ...,
                ge=0,
                le=1,
            )

            @field_validator(
                "non_sexist",
                "ideological_inequality",
                "stereotyping_dominance",
                "objectification",
                "sexual_violence",
                "misogyny_non_sexual_violence",
                mode="before",
            )
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
        self.examples = {
            "non_sexist": [],
            "ideological_inequality": [],
            "stereotyping_dominance": [],
            "objectification": [],
            "sexual_violence": [],
            "misogyny_non_sexual_violence": [],
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

    def _process_answer(self, answer: List[List[str]]):
        total_count = len(answer)
        answer: List[str] = [item for sublist in answer for item in sublist]
        counts = {
            "-": 0,
            "ideological_inequality": 0,
            "stereotyping_dominance": 0,
            "objectification": 0,
            "sexual_violence": 0,
            "misogyny_non_sexual_violence": 0,
        }
        for item in answer:
            if item == "-":
                counts["-"] += 1
            else:
                item = item.lower().replace("-", "_")
                if item in counts:
                    counts[item] += 1

        probs = {
            "non_sexist": counts["-"] / total_count,
            "ideological_inequality": counts["ideological_inequality"] / total_count,
            "stereotyping_dominance": counts["stereotyping_dominance"] / total_count,
            "objectification": counts["objectification"] / total_count,
            "sexual_violence": counts["sexual_violence"] / total_count,
            "misogyny_non_sexual_violence": counts["misogyny_non_sexual_violence"]
            / total_count,
        }

        # Round to 2 decimal places
        rounded_probs = {k: round(v, 3) for k, v in probs.items()}

        # Ensure probabilities sum to 1
        # total = sum(rounded_probs.values())
        # if total != 1:
        #    diff = 1 - total
        #    max_key = max(rounded_probs, key=rounded_probs.get)
        #    rounded_probs[max_key] = round(rounded_probs[max_key] + diff, 2)

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
        Normalizes the prediction probabilities using the same method as in _process_answer.
        """
        model = self.get_pydantic_model()
        probs = {
            "non_sexist": prediction.non_sexist,
            "ideological_inequality": prediction.ideological_inequality,
            "stereotyping_dominance": prediction.stereotyping_dominance,
            "objectification": prediction.objectification,
            "sexual_violence": prediction.sexual_violence,
            "misogyny_non_sexual_violence": prediction.misogyny_non_sexual_violence,
        }

        # Round to 3 decimal places
        rounded_probs = {k: round(v, 3) for k, v in probs.items()}

        # Ensure probabilities sum to 1
        total = sum(rounded_probs.values())
        #if total != 1:
        #    diff = 1 - total
        #    max_key = max(rounded_probs, key=rounded_probs.get)
        #    rounded_probs[max_key] = round(rounded_probs[max_key] + diff, 3)

        return model(**rounded_probs)

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
                    for k in [
                        "non_sexist",
                        "ideological_inequality",
                        "misogyny_non_sexual_violence",
                        "objectification",
                        "stereotyping_dominance",
                        "sexual_violence",
                    ]
                ]
            )
            pred_probs = torch.tensor(
                [
                    getattr(normalized_prediction, k)
                    for k in [
                        "non_sexist",
                        "ideological_inequality",
                        "misogyny_non_sexual_violence",
                        "objectification",
                        "stereotyping_dominance",
                        "sexual_violence",
                    ]
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
                    "MISOGYNY-NON-SEXUAL-VIOLENCE": predictions[
                        i
                    ].misogyny_non_sexual_violence,
                    "IDEOLOGICAL-INEQUALITY": predictions[i].ideological_inequality,
                    "STEREOTYPING-DOMINANCE": predictions[i].stereotyping_dominance,
                    "SEXUAL-VIOLENCE": predictions[i].sexual_violence,
                    "OBJECTIFICATION": predictions[i].objectification,
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
                "IDEOLOGICAL-INEQUALITY": prediction.ideological_inequality,
                "STEREOTYPING-DOMINANCE": prediction.stereotyping_dominance,
                "OBJECTIFICATION": prediction.objectification,
                "SEXUAL-VIOLENCE": prediction.sexual_violence,
                "MISOGYNY-NON-SEXUAL-VIOLENCE": prediction.misogyny_non_sexual_violence,
            }
            data[i]["answer"] = {
                "NO": data[i]["answer"].non_sexist,
                "IDEOLOGICAL-INEQUALITY": data[i]["answer"].ideological_inequality,
                "STEREOTYPING-DOMINANCE": data[i]["answer"].stereotyping_dominance,
                "OBJECTIFICATION": data[i]["answer"].objectification,
                "SEXUAL-VIOLENCE": data[i]["answer"].sexual_violence,
                "MISOGYNY-NON-SEXUAL-VIOLENCE": data[i][
                    "answer"
                ].misogyny_non_sexual_violence,
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
