import json
import os
import random
from enum import Enum
from typing import Dict, List

from pydantic import BaseModel
from sklearn.metrics import f1_score

from src.tasks.task import Task

random.seed(33)


class Exists2022T2(Task):
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
Analyze the given text to determine if it contains sexist content. If it does, classify it into one of the following categories: 'ideological-inequality', 'misogyny-non-sexual-violence', 'objectification', 'sexual-violence', 'stereotyping-dominance'. If it doesn't contain sexist content, classify it as 'non-sexist'.

Definition of sexism: Sexism is prejudice, stereotyping, or discrimination, typically against women, on the basis of sex. It includes attitudes, behaviors, and practices that promote stereotyping of social roles based on gender.

- 'non-sexist': The text does not contain any sexist language, attitudes, or implications.
- 'ideological-inequality': The text discredits feminism, denies gender inequality, or portrays men as victims of gender-based oppression.
- 'stereotyping-dominance': The text expresses stereotypes about gender roles or claims superiority of one gender over another.
- 'objectification': The text reduces women to their physical attributes or treats them as objects for male pleasure.
- 'sexual-violence': The text contains sexual harassment, assault, or rape-related content.
- 'misogyny-non-sexual-violence': The text expresses hatred, contempt, or non-sexual violence towards women.

Output: Provide your answer as a JSON object with the key 'label' and the value set to one of the categories listed above.

""".strip()

    def get_pydantic_model(self):
        class LabelEnum(str, Enum):
            non_sexist = "non-sexist"
            ideological_inequality = "ideological-inequality"
            misogyny_non_sexual_violence = "misogyny-non-sexual-violence"
            objectification = "objectification"
            sexual_violence = "sexual-violence"
            stereotyping_dominance = "stereotyping-dominance"

        class Identification(BaseModel):
            label: LabelEnum

        return Identification

    def _precompute_examples(self):
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

        true_labels = [item["answer"].label for item in data]
        pred_labels = [prediction.label for prediction in predictions]

        macro_f1 = f1_score(true_labels, pred_labels, average="macro")

        return {"macro_f1": macro_f1}

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
                "value": prediction.label,
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
            data[i]["prediction"] = prediction.label
            data[i]["answer"] = data[i]["answer"].label

        output_path = os.path.join(
            output_dir,
            self.output_path.replace(".json", "_val.json"),
        )
        self._write_json(output_path, data)

    @staticmethod
    def _write_json(path: str, data: List[Dict]):
        with open(path, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=4, ensure_ascii=False)
