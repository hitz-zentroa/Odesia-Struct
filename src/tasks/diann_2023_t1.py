import json
import os
import random
from typing import Dict, List

from pydantic import BaseModel
from seqeval.metrics import f1_score, precision_score, recall_score
from seqeval.scheme import IOB2

from src.tasks.task import Task

random.seed(33)


class Diann2023T1(Task):
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
        return "You are an AI assistant trained for named entity recognition. You are given a text and you need to extract disabilities mentioned in the text."

    def get_instruction(self):
        return """
Analyze the given text and extract all named entities that are disabilities.

Disabilities encompass impairments, activity limitations, and participation restrictions. They can be expressed as:
1. Specific terms (e.g., "blindness")
2. Limitations of human functions (e.g., "lack of vision")

Only extract disabilities from this list:
   Aphasia, Apraxia, Ataxia, Blindness, Deaf-mute, Deafness, Dementia, Dysarthria, Dysautonomia, Dyskinesias, Dysphagia, Hemiplegia, Hyperactivity

Rules:
- Extract disabilities exactly as they appear in the text.
- Do not include disabilities not mentioned in the text.
- If no disabilities are mentioned, return an empty list.

Output Format:
Provide your answer as a JSON object:
{
    "disabilities": ["disability1", "disability2", ...]
}
""".strip()

    def get_pydantic_model(self):
        class SequenceLabelling(BaseModel):
            disabilities: List[str]

        return SequenceLabelling

    def _precompute_examples(self):
        train_data = self.get_split("train")
        self.examples = train_data

    def get_few_shot(self):
        return random.sample(
            self.examples, min(self.num_examples_few_shot, len(self.examples))
        )

    def _extract_disabilities(self, tokens, labels):
        disabilities = []
        current_disability = []
        for token, label in zip(tokens, labels):
            if label.startswith("B-DIS"):
                if current_disability:
                    disabilities.append(" ".join(current_disability))
                    current_disability = []
                current_disability.append(token)
            elif label.startswith("I-DIS"):
                current_disability.append(token)
            elif current_disability:
                disabilities.append(" ".join(current_disability))
                current_disability = []
        if current_disability:
            disabilities.append(" ".join(current_disability))
        return disabilities

    def read_dataset(self, dataset: str):
        with open(dataset, "r", encoding="utf-8") as file:
            data = json.load(file)

        processed_data = []
        for item in data:
            text = " ".join(item["tokens"])
            disabilities = (
                self._extract_disabilities(item["tokens"], item["value"])
                if "value" in item
                else None
            )
            processed_data.append(
                {
                    "question": text,
                    "answer": self.get_pydantic_model()(disabilities=disabilities)
                    if disabilities is not None
                    else None,
                    "test_case": item["test_case"],
                    "id": item["id"],
                }
            )

        return processed_data

    def _convert_to_iob(self, tokens: List[str], disabilities: List[str]) -> List[str]:
        """
        Convert tokens and disabilities to IOB2 format, handling multiple occurrences
        """
        iob_tags = ["O"] * len(tokens)
        for disability in disabilities:
            disability_tokens = disability.split()
            for i in range(len(tokens) - len(disability_tokens) + 1):
                if tokens[i : i + len(disability_tokens)] == disability_tokens:
                    if iob_tags[i] == "O":  # Only change if not already tagged
                        iob_tags[i] = "B-DIS"
                        for j in range(1, len(disability_tokens)):
                            if iob_tags[i + j] == "O":
                                iob_tags[i + j] = "I-DIS"
                            else:
                                break

        # Ensure valid IOB2 annotation
        for i in range(1, len(iob_tags)):
            if iob_tags[i] == "I-DIS" and iob_tags[i - 1] == "O":
                iob_tags[i] = "B-DIS"

        return iob_tags

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

        true_labels = []
        pred_labels = []

        for i, prediction in enumerate(predictions):
            true_disabilities = data[i]["answer"].disabilities
            pred_disabilities = prediction.disabilities

            true_iob = self._convert_to_iob(
                data[i]["question"].split(), true_disabilities
            )
            pred_iob = self._convert_to_iob(
                data[i]["question"].split(), pred_disabilities
            )

            true_labels.append(true_iob)
            pred_labels.append(pred_iob)

        precision = precision_score(true_labels, pred_labels, scheme=IOB2)
        recall = recall_score(true_labels, pred_labels, scheme=IOB2)
        f1 = f1_score(true_labels, pred_labels, scheme=IOB2)

        return {"precision": precision, "recall": recall, "f1": f1}

    def build_test_file(self, predictions: List[BaseModel], output_dir: str):
        """
        Builds a test file with the predictions

        Args:
            predictions (List[BaseModel]): List of predictions
        """
        data = self.get_split("test")
        assert len(data) == len(predictions)

        test_data = []
        for i, prediction in enumerate(predictions):
            tokens = data[i]["question"].split()
            iob_tags = self._convert_to_iob(tokens, prediction.disabilities)
            test_data.append(
                {
                    "id": data[i]["id"],
                    "test_case": data[i]["test_case"],
                    "value": iob_tags,
                }
            )

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
            data[i]["prediction"] = prediction.disabilities
            data[i]["answer"] = data[i]["answer"].disabilities

        output_path = os.path.join(
            output_dir,
            self.output_path.replace(".json", "_val.json"),
        )
        self._write_json(output_path, data)

    @staticmethod
    def _write_json(path: str, data: List[Dict]):
        with open(path, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=4, ensure_ascii=False)
