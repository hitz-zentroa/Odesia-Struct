# DATASETS FOR THE TASK  DIANN 2023: Disability Detection

The task is  performed on the datasets DIANN 2018 (training and
validation partitions) and  DIANN 2023 (private test set), which are
available in  English and Spanish. The dataset contains abstracts of
biomedical articles related to rare diseases. Tha abstracts are annotated with disabilities. The
task is a named entity recognition task, consisting in detecting mentions of disabilities.

DIANN 2018 was created for the competition  “Disability annotation on
documents from the biomedical domain (DIANN)” (Fabregat et al., 2018)
that took place with IberLEF 2018. DIANN 2023 was created as a private test partition for the Leaderboard ODESIA. Both  corpora have been
annotated following the same guidelines.

The task is a sequence labeling task that follows a BIO format. Each token in the abstract is annotated as being at the beginning
(“B-DIS”), inside ("I-DIS”) or outside ("O") the mention of a disability.

More information about the task is provided in the description paper:

Hermenegildo Fabregat, Juan Martínez-Romo, Lourdes Araujo (2018) Overview of the DIANN Task: Disability Annotation Task. Proceedings of the Third Workshop on Evaluation of Human Language Technologies for Iberian Languages (IberEval 2018).

### Dataset size

For both languages 400 abstracts are provided for training, 98 for validation and 99 for test.

### Files provided

Test, training, and validation files in json format are provided for English and Spanish:

- test_t1_en.json
- test_t1_es.json
- train_t1_en.json
- train_t1_es.json
- val_t1_en.json
- val_t1_es.json

The files to be submitted are:

- DIANN_2023_T1_en.json
- DIANN_2023_T1_es.json

### File format for the task DIANN 2023: Disability Detection

The format of the training and validation files is as follows:

[
    {
        "id": "849",
        "test_case": "DIANN2018",
        "tokens": [
            "botulinum",
            "toxin",
            "was",
            "present",
            ",",
            "causing",
            "a",
            "retinal",
            "tear",
            "."
        ],
        "value": [
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "B-DIS",
            "I-DIS",
            "O"
        ]
    },
    {
        "id": "850",
        "test_case": "DIANN2018",
        "tokens": [
            "Abstract",
            "Background",
            "Previous",
            "studies",
        ],
        "value": [
            "O",
            "O",
            "O",
            "O"
        ]
		}
]


- "id": a unique identifier of the abstract.
- "test_case": tag needed in the PyEvALL library for evaluating classification tasks.
- "tokens": list of all the tokens in the abstract.
- "value":  a list of values with the BIO tags "B-DIS", "I-DIS", "O"
  for every token in "tokens". The lists "tokens" and "value" need to
  have the same number of elements.


The format of the test files is the same as the training and validation files, without the "value" element:

[
    {
        "id": "849",
        "test_case": "DIANN2023",
        "tokens": [
            "botulinum",
            "toxin",
            "was",
            "present",
            ",",
            "causing",
            "a",
            "retinal",
            "tear",
            "."
        ]
    },
    {
        "id": "850",
        "test_case": "DIANN2018",
        "tokens": [
            "Abstract",
            "Background",
            "Previous",
            "studies",
        ]
		}
]


### Submission format for the task DIANN 2023

The prediction files must have the following names:

- For Spanish: DIANN_2023_T1_es.json 

- For English: DIANN_2023_T1_en.json 

The content of the file must be a json list with the following format:.

- "test_case": tag needed in the PyEvALL library for evaluating classification tasks. This tag is set to "DIANN2023".
- "id": the unique identifier of the abstract.
- "value": a list with your system prediction for every token in the abstract. 

### Example of a submission file:  DIANN_2023_T1_es.json

[
    {
        "id": "105",
        "test_case": "DIANN2023",
        "value": [
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "B-DIS",
            "I-DIS",
            "O"
        ]
    },
    {
        "id": "106",
        "test_case": "DIANN2023",
         "value": [
            "O",
            "O",
            "B-DIS",
            "I-DIS",
            "O",
            "O",
            "O"
        ]
	}
]
