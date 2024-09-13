import json
import os
import random
from enum import Enum
from typing import Dict, List

from pydantic import BaseModel
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer

from src.tasks.task import Task

random.seed(33)


class Dipromats2023T2(Task):
    def __init__(
        self,
        num_examples_few_shot: int = 20,
        **kwargs,
    ):
        self.number2simple = {
            "1 appeal to commonality": "appeal-to-commonality",
            "2 discrediting the opponent": "discrediting-the-opponent",
            "3 loaded language": "loaded-language",
            "4 appeal to authority": "appeal-to-authority",
            "false": "non-propaganda",
        }

        self.simple2number = {
            "appeal-to-commonality": "1 appeal to commonality",
            "discrediting-the-opponent": "2 discrediting the opponent",
            "loaded-language": "3 loaded language",
            "appeal-to-authority": "4 appeal to authority",
            "non-propaganda": "false",
        }

        super().__init__(
            num_examples_few_shot=num_examples_few_shot,
            **kwargs,
        )
        self._precompute_examples()

    def get_system_prompt(self):
        return "You are an AI assistant trained to identify propaganda content in text. Your task is to analyze the given tweet and determine whether it contains propaganda techniques."

    def get_instruction(self):
        return """
Analyze the given text to determine if it contains propaganda techniques. If it does, classify them into at least one of the following categories: 'appeal-to-commonality', 'discrediting-the-opponent', 'loaded-language', 'appeal-to-authority'. If it doesn't cointain any propaganda technique classify them as 'non-propaganda'

Definition of propaganda: The deliberate, systematic attempt to shape perceptions, manipulate cognitions, and direct behavior to achieve a response that furthers the desired intent of the propagandist

- 'appeal-to-commonality': The tweet uses appeals to a community's will, tradition, or history to support its argument. It may include exaggerated praise of a nation, reverence for patriotic symbols, self-praise, or depiction of someone as a hero.
- 'discrediting-the-opponent': Include discrediting them through negative labels, attacking their behavior or morality, scapegoating specific individuals or groups, accusing others of spreading propaganda, and making personal attacks on their private lives. These techniques also involve instilling fear, portraying the opponent's ideas as absurd, generating hatred by presenting them as existential threats, casting doubt on their credibility, and associating their actions with universally disliked figures or concepts to undermine their position.
- 'loaded-language': Includes hyperbolic language, evocative metaphors and words with strong emotional connotations.
- 'appeal-to-authority': includes a third person or institution to support an idea, message, or behavior for which they should not be considered as a valid expert.
- 'non-propaganda': The text does not contain propaganda content.

Output: Provide your answer as a JSON object with 'label' as a list of the categories that apply to the text.

""".strip()

    def get_pydantic_model(self):
        class LabelEnum(str, Enum):
            non_propaganda = "non-propaganda"
            appeal_to_commonality = "appeal-to-commonality"
            discrediting_the_opponent = "discrediting-the-opponent"
            loaded_language = "loaded-language"
            appeal_to_authority = "appeal-to-authority"

        class Identification(BaseModel):
            label: List[LabelEnum]

        return Identification

    def _precompute_examples(self):
        train_data = self.get_split("train")
        model = self.get_pydantic_model()
        LabelEnum = model.model_fields["label"].annotation.__args__[0]

        # Create the class_examples dictionary using the enum members
        self.class_examples = {label.value: [] for label in LabelEnum}

        for ex in train_data:
            for label in ex["answer"].label:
                self.class_examples[label].append(ex)

        # Pre-compute all_labels for get_few_shot
        self.all_labels = list(self.class_examples.keys())

    def get_few_shot(self):
        # Ensure at least one example per label
        few_shot_examples = [
            random.choice(examples) if examples else None
            for examples in self.class_examples.values()
        ]
        few_shot_examples = [ex for ex in few_shot_examples if ex is not None]

        # Fill remaining slots
        remaining_slots = self.num_examples_few_shot - len(few_shot_examples)
        if remaining_slots > 0:
            additional_examples = [
                ex
                for examples in self.class_examples.values()
                for ex in examples
                if ex not in few_shot_examples
            ]
            few_shot_examples.extend(
                random.sample(
                    additional_examples, min(remaining_slots, len(additional_examples))
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
                "answer": model(
                    label=[self.number2simple[label] for label in item["value"]]
                )
                if "value" in item
                else None,
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

        # Use MultiLabelBinarizer to transform labels into binary format
        mlb = MultiLabelBinarizer()
        true_labels_bin = mlb.fit_transform(true_labels)
        pred_labels_bin = mlb.transform(pred_labels)

        # Calculate micro-averaged F1 score
        micro_f1 = f1_score(true_labels_bin, pred_labels_bin, average="micro")

        return {"micro_f1": micro_f1}

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
                "value": list(
                    set([self.simple2number[label] for label in prediction.label])
                ),
            }
            for i, prediction in enumerate(predictions)
        ]

        for item in test_data:
            item["value"].sort()

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
            data[i]["prediction"] = list(
                set([self.simple2number[label] for label in prediction.label])
            )
            data[i]["prediction"].sort()

            data[i]["answer"] = list(
                set([self.simple2number[label] for label in data[i]["answer"].label])
            )
            data[i]["answer"].sort()

        output_path = os.path.join(
            output_dir,
            self.output_path.replace(".json", "_val.json"),
        )
        self._write_json(output_path, data)

    @staticmethod
    def _write_json(path: str, data: List[Dict]):
        with open(path, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=4, ensure_ascii=False)
