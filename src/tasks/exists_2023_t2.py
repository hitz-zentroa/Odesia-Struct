import json
from typing import Dict, List, Literal

from pydantic import BaseModel

from src.tasks.task import Task
from sklearn.metrics import f1_score
import random

random.seed(33)


class Exists2023T2(Task):
    def __init__(
        self,
        train_dataset: str,
        dev_dataset: str,
        test_dataset: str,
        output_path: str,
        num_examples_few_shot: int = 20,
    ):
        super().__init__(
            train_dataset=train_dataset,
            dev_dataset=dev_dataset,
            test_dataset=test_dataset,
            output_path=output_path,
            num_examples_few_shot=num_examples_few_shot,
        )
        self._precompute_examples()

    def prompt(self):
        return """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an AI assistant trained to identify and classify sexist content in text. Your task is to analyze the given input and determine whether it contains sexist language or attitudes, and if so, to classify it into specific categories.

Cutting Knowledge Date: December 2023
Today Date: 26 Jul 2024

<|eot_id|><|start_header_id|>user<|end_header_id|>

{{ instructions }}

Examples
--------

{% for example in examples %}
Input: {{ example.question }}
Output: {{ example.answer.model_dump_json() }}

{% endfor %}

--------

Now, analyze the following input:

Input: {{ question }}<|eot_id|><|start_header_id|>assistant<|end_header_id|>{{ "\n\n" }}"""

    def get_instruction(self):
        return """
Analyze the given text to determine if it contains sexist content. If it does, classify it into one of the following categories based on the intention of the author 'direct', 'reported', 'judgemental' or 'non-sexist' if it doesn't contain sexist content.

Definition of sexism:
Sexism is prejudice, stereotyping, or discrimination, typically against women, on the basis of sex. It includes attitudes, behaviors, and practices that promote stereotyping of social roles based on gender.

Task:
1. Carefully read and understand the given text.
2. Identify any language, attitudes, or implications that could be considered sexist.
3. Classify the text into one of the following categories:

- 'non-sexist': The text does not contain any sexist language, attitudes, or implications.
- 'direct': The intention was to write a message that is sexist by itself or incites to be sexist
- 'reported': The intention is to report and share a sexist situation suffered by a woman or women in first or third person.
- 'judgemental': The intention was to judge, since the tweet describes sexist situations or behaviours with the aim of condemning them.

Output:
Provide your answer as a JSON object with the key 'label' and the value set to one of the categories listed above.

""".strip()

    def get_pydantic_model(self):
        class Identification(BaseModel):
            label: Literal["non-sexist", "direct", "reported", "judgemental"]

        return Identification

    def _precompute_examples(self):
        train_data = self.get_split("train")
        model = self.get_pydantic_model()
        self.class_examples = {
            label: [] for label in model.model_fields["label"].annotation.__args__
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

    def _process_answer(self, answer: List[str]):
        # count the number of times each label appears and return the most common one
        answer = [x.lower() for x in answer]
        answer = [x for x in answer if x in ["direct", "reported", "judgemental","-"]]
        answer = max(set(answer), key=answer.count)
        if answer == "-":
            answer = "non-sexist"
        return answer

    def read_dataset(self, dataset: str):
        with open(dataset, "r", encoding="utf-8") as file:
            data = json.load(file)

        model = self.get_pydantic_model()
        return [
            {
                "question": item["tweet"] if "tweet" in item else item["text"],
                "answer": model(label=self._process_answer(item["value"]))
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

        macro_f1 = f1_score(true_labels, pred_labels, average="macro")

        return macro_f1

    def build_test_file(self, predictions: List[BaseModel]):
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
        self._write_json(self.output_path, test_data)

    def build_validation_file(self, predictions: List[BaseModel]):
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

        output_path = self.dev_dataset.replace(".json", "_pred.json")
        self._write_json(output_path, data)

    @staticmethod
    def _write_json(path: str, data: List[Dict]):
        with open(path, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=4, ensure_ascii=False)
