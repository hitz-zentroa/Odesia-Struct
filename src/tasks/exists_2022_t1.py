import json
from typing import Dict, List, Literal

from pydantic import BaseModel

from src.tasks.task import Task

import random

random.seed(33)


class Exists2022T1(Task):
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

You are an AI assistant trained to identify sexist content in text. Your task is to analyze the given input and determine whether it contains sexist language or attitudes.

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
Analyze the given text to determine if it contains sexist content. 

Definition of sexism:
Sexism is prejudice, stereotyping, or discrimination, typically against women, on the basis of sex. It includes attitudes, behaviors, and practices that promote stereotyping of social roles based on gender.

Task:
1. Carefully read and understand the given text.
2. Identify any language, attitudes, or implications that could be considered sexist.
3. Classify the text as either 'sexist' or 'non-sexist'.

Guidelines:
- 'Sexist': The text expresses or implies sexist attitudes, stereotypes, or discriminatory views.
- 'Non-Sexist': The text does not contain any sexist language, attitudes, or implications.

Note: Be aware of subtle forms of sexism, including benevolent sexism or seemingly positive stereotypes that still reinforce gender inequality.

Output:
Provide your answer as a JSON object with the key 'label' and the value set to either 'sexist' or 'non-sexist'.

""".strip()

    def get_pydantic_model(self):
        class Identification(BaseModel):
            label: Literal["sexist", "non-sexist"]

        return Identification

    def _precompute_examples(self):
        train_data = self.get_split("train")
        model = self.get_pydantic_model()
        self.class_examples = {
            label: [] for label in model.model_fields['label'].annotation.__args__
        }
        for ex in train_data:
            self.class_examples[ex["answer"].label].append(ex)

    def get_few_shot(self):
        examples_per_class = self.num_examples_few_shot // len(self.class_examples)
        few_shot_examples = []
        for class_examples in self.class_examples.values():
            few_shot_examples.extend(random.sample(class_examples, min(examples_per_class, len(class_examples))))
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

        return sum(
            prediction.label.lower().strip() == data[i]["answer"].label.lower().strip()
            for i, prediction in enumerate(predictions)
        ) / len(predictions)

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