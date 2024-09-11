import json
import os
import random
from typing import Dict, List

from pydantic import BaseModel, Field

from src.tasks.task import Task

random.seed(33)


class SQUAD2024T1(Task):
    def __init__(
        self,
        num_examples_few_shot: int = 5,
        **kwargs,
    ):
        super().__init__(
            num_examples_few_shot=num_examples_few_shot,
            **kwargs,
        )
        self._precompute_examples()

    def get_system_prompt(self):
        return "You are an AI assistant trained for extractive question answering in SQUAD."

    def get_instruction(self):
        return """
Answer the questions about a text in such a way that the answer is a fragment extracted directly from the text. The answer to be provided corresponds to the shortest span needed to answer the question. In all cases, the answers are fragments of the text and all questions can be answered from the text.

Given a question and context, your task is to identify the shortest span of text that answers the question.

Rules:
- Extract the answer exactly as it appears in the text. Copy the text word for word.
- Do not answer anything that is not in the text. Do not modify the text in any way.

Output Format:
Provide your answer as a JSON object:
{
    "answer": "The answer to the question",
}

""".strip()

    def get_pydantic_model(self):
        class QuestionAnwering(BaseModel):
            answer: str = Field(..., description="The answer to the question")

        return QuestionAnwering

    def _precompute_examples(self):
        train_data = self.get_split("train")
        self.examples = train_data

    def get_few_shot(self):
        return random.sample(
            self.examples, min(self.num_examples_few_shot, len(self.examples))
        )

    def _format_input(self, context: str, question: str):
        return f"""
Context:
{context} 
Question:
{question}
""".strip()

    def read_dataset(self, dataset: str):
        with open(dataset, "r", encoding="utf-8") as file:
            data = json.load(file)

        processed_data = []
        model = self.get_pydantic_model()
        for item in data:
            processed_data.append(
                {
                    "question": self._format_input(item["context"], item["question"]),
                    "answer": model(answer=item["value"]) if "value" in item else None,
                    "test_case": item["test_case"],
                    "id": item["id"],
                }
            )

        return processed_data

    @staticmethod
    def _compute_f1(ground_truth: str, prediction: str) -> float:
        """
        Computes F1 score between ground truth and prediction

        Args:
            ground_truth (str): Ground truth answer
            prediction (str): Predicted answer

        Returns:
            float: F1 score
        """

        def normalize_answer(s):
            return " ".join(s.lower().split())

        ground_truth = normalize_answer(ground_truth)
        prediction = normalize_answer(prediction)

        if ground_truth == prediction:
            return 1.0

        gt_tokens = ground_truth.split()
        pred_tokens = prediction.split()

        common = set(gt_tokens) & set(pred_tokens)
        num_same = len(common)

        if num_same == 0:
            return 0.0

        precision = num_same / len(pred_tokens)
        recall = num_same / len(gt_tokens)

        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    def evaluate(self, predictions: List[BaseModel], split="dev") -> Dict[str, float]:
        """
        Evaluates the predictions using SQUAD F1 score

        Args:
            predictions (List[BaseModel]): List of predictions
            split (str, optional): Split to evaluate on. Defaults to "dev".

        Returns:
            Dict[str, float]: Dictionary with the evaluation metric
        """
        data = self.get_split(split)
        assert len(data) == len(predictions)

        f1_scores = []
        for ground_truth, prediction in zip(data, predictions):
            f1 = self._compute_f1(ground_truth["answer"].answer, prediction.answer)
            f1_scores.append(f1)

        average_f1 = sum(f1_scores) / len(f1_scores)
        return {"f1": average_f1}

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
            test_data.append(
                {
                    "id": data[i]["id"],
                    "test_case": data[i]["test_case"],
                    "value": prediction.answer,
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
            data[i]["prediction"] = prediction.answer
            data[i]["answer"] = data[i]["answer"].answer

        output_path = os.path.join(
            output_dir,
            self.output_path.replace(".json", "_val.json"),
        )
        self._write_json(output_path, data)

    @staticmethod
    def _write_json(path: str, data: List[Dict]):
        with open(path, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=4, ensure_ascii=False)
