from typing import List, Union

from src.tasks import (
    diann_2023_t1,
    dipromats_2023_t1,
    dipromats_2023_t2,
    dipromats_2023_t3,
    exists_2022_t1,
    exists_2022_t2,
    exists_2023_t1,
    exists_2023_t2,
    exists_2023_t3,
    squad_2024_t1,
)


def get_tasks(tokenizer, tasks: Union[List[str], str] = "all"):
    all_tasks = {
        "exist_2022_t1_es": exists_2022_t1.Exists2022T1,
        # "exist_2022_t1_en": exists_2022_t1.Exists2022T1,
        "exist_2022_t2_es": exists_2022_t2.Exists2022T2,
        # "exist_2022_t2_en": exists_2022_t2.Exists2022T2,
        "exist_2023_t1_es": exists_2023_t1.Exists2023T1,
        # "exist_2023_t1_en": exists_2023_t1.Exists2023T1,
        "exist_2023_t2_es": exists_2023_t2.Exists2023T2,
        # "exist_2023_t2_en": exists_2023_t2.Exists2023T2,
        "exist_2023_t3_es": exists_2023_t3.Exists2023T3,
        # "exist_2023_t3_en": exists_2023_t3.Exists2023T3,
        "dipromats_2023_t1_es": dipromats_2023_t1.Dipromats2023T1,
        # "dipromats_2023_t1_en": dipromats_2023_t1.Dipromats2023T1,
        "dipromats_2023_t2_es": dipromats_2023_t2.Dipromats2023T2,
        # "dipromats_2023_t2_en": dipromats_2023_t2.Dipromats2023T2,
        "dipromats_2023_t3_es": dipromats_2023_t3.Dipromats2023T3,
        # "dipromats_2023_t3_en": dipromats_2023_t3.Dipromats2023T3,
        "diann_2023_t1_es": diann_2023_t1.Diann2023T1,
        # "diann_2023_t1_en": diann_2023_t1.Diann2023T1,
        "squad_2024_t1_es": squad_2024_t1.SQUAD2024T1,
        # "squad_2024_t1_en": squad_2024_t1.SQUAD2024T1,
    }

    tasks_dict = {}

    if tasks == "all" or tasks == ["all"]:
        tasks = all_tasks.keys()

    for task_name in tasks:
        if task_name in all_tasks:
            task_class = all_tasks[task_name]
            dataset, year, task_number, lang = task_name.split("_")
            if dataset == "squad":
                dataset = "sqac_squad"
                tasks_dict[task_name] = task_class(
                    train_dataset=f"data/{dataset}_{year}/train_{task_number}_{lang}.json",
                    dev_dataset=f"data/{dataset}_{year}/val_{task_number}_{lang}.json",
                    test_dataset=f"data/{dataset}_{year}/test_{task_number}_{lang}.json",
                    output_path=f"{dataset.upper()}_{year}_{lang}.json",
                    tokenizer=tokenizer,
                )
            else:
                tasks_dict[task_name] = task_class(
                    train_dataset=f"data/{dataset}_{year}/train_{task_number}_{lang}.json",
                    dev_dataset=f"data/{dataset}_{year}/val_{task_number}_{lang}.json",
                    test_dataset=f"data/{dataset}_{year}/test_{task_number}_{lang}.json",
                    output_path=f"{dataset.upper()}_{year}_{task_number.upper()}_{lang}.json",
                    tokenizer=tokenizer,
                )

    return tasks_dict
