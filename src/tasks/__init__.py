from src.tasks import exists_2022_t1
from src.tasks import exists_2022_t2
from src.tasks import exists_2023_t1
from src.tasks import exists_2023_t2
from src.tasks import exists_2023_t3
tasks_dict = {
    "exists_2022_t1_es": exists_2022_t1.Exists2022T1(
        train_dataset="data/exist_2022/train_t1_es.json",
        dev_dataset="data/exist_2022/val_t1_es_mini.json",
        test_dataset="data/exist_2022/test_t1_es.json",
        output_path="data/exist_2022/EXIST_2022_T1_es.json",
    ),
    "exists_2022_t1_en": exists_2022_t1.Exists2022T1(
        train_dataset="data/exist_2022/train_t1_en.json",
        dev_dataset="data/exist_2022/val_t1_en.json",
        test_dataset="data/exist_2022/test_t1_en.json",
        output_path="data/exist_2022/EXIST_2022_T1_en.json",
    ),
    "exists_2022_t2_es": exists_2022_t2.Exists2022T2(
        train_dataset="data/exist_2022/train_t2_es.json",
        dev_dataset="data/exist_2022/val_t2_es_mini.json",
        test_dataset="data/exist_2022/test_t2_es.json",
        output_path="data/exist_2022/EXIST_2022_T2_es.json",
    ),
    "exists_2022_t2_en": exists_2022_t2.Exists2022T2(
        train_dataset="data/exist_2022/train_t2_en.json",
        dev_dataset="data/exist_2022/val_t2_en.json",
        test_dataset="data/exist_2022/test_t2_en.json",
        output_path="data/exist_2022/EXIST_2022_T2_en.json",
    ),
    "exists_2023_t1_es": exists_2023_t1.Exists2023T1(
        train_dataset="data/exist_2023/train_t1_es.json",
        dev_dataset="data/exist_2023/val_t1_es_mini.json",
        test_dataset="data/exist_2023/test_t1_es.json",
        output_path="data/exist_2023/EXIST_2023_T1_es.json",
    ),
    "exists_2023_t1_en": exists_2023_t1.Exists2023T1(
        train_dataset="data/exist_2023/train_t1_en.json",
        dev_dataset="data/exist_2023/val_t1_en.json",
        test_dataset="data/exist_2023/test_t1_en.json",
        output_path="data/exist_2023/EXIST_2023_T1_en.json",
    ),
    "exists_2023_t2_es": exists_2023_t2.Exists2023T2(
        train_dataset="data/exist_2023/train_t2_es.json",
        dev_dataset="data/exist_2023/val_t2_es_mini.json",
        test_dataset="data/exist_2023/test_t2_es.json",
        output_path="data/exist_2023/EXIST_2023_T2_es.json",
    ),
    "exists_2023_t2_en": exists_2023_t2.Exists2023T2(
        train_dataset="data/exist_2023/train_t2_en.json",
        dev_dataset="data/exist_2023/val_t2_en.json",
        test_dataset="data/exist_2023/test_t2_en.json",
        output_path="data/exist_2023/EXIST_2023_T2_en.json",
    ),
    "exists_2023_t3_es": exists_2023_t3.Exists2023T3(
        train_dataset="data/exist_2023/train_t3_es.json",
        dev_dataset="data/exist_2023/val_t3_es_mini.json",
        test_dataset="data/exist_2023/test_t3_es.json",
        output_path="data/exist_2023/EXIST_2023_T3_es.json",
    ),
    "exists_2023_t3_en": exists_2023_t3.Exists2023T3(
        train_dataset="data/exist_2023/train_t3_en.json",
        dev_dataset="data/exist_2023/val_t3_en.json",
        test_dataset="data/exist_2023/test_t3_en.json",
        output_path="data/exist_2023/EXIST_2023_T3_en.json",
    ),
}

