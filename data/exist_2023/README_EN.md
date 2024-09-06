#	DATASET EXIST 2023

### Dataset Description

The EXIST 2023 dataset is composed of tweets, both in English and Spanish, that may contain sexist expressions and are tagged following the Learning with Disagreement paradigm (i.e., each tweet is labeled by 6 different persons and all labels are provided in the dataset).

For a complete description of the dataset, please read: 

Plaza, L., Carrillo-de-Albornoz, J. *et al*. 2023. *Overview of EXIST 2023 – Learning with Disagreement for Sexism Identification and Characterization*. In Experimental IR Meets Multilinguality, Multimodality, and Interaction: 14th International Conference of the CLEF Association, CLEF 2023, pages 316–342. 



### EXIST 2023 Tasks

The EXIST 2023 dataset addresses three tasks:

* **TASK 1: Sexism Identification**. A binary classification task where systems have to decide whether or not a given tweet is sexist.

* **TASK 2: Source Intention**. A three-class, mono-label classification task where each sexist tweet must be classified according to the intention of the person who wrote it.

* **TASK 3: Sexism Categorization**. A five-class, multi-label classification task where each sexist tweet must be classified according to the type of sexism.



### Training, Validation and Test Files


* **train_t1_en** and **train_t1_es**. These are the train files, in English and Spanish respectively, for Task 1. The format is the same for the two files:

        * "id": a unique identifier of the tweet.
        * "lang": the language of the text. Possible values are “en” or “es”.
        * "tweet": the text of the tweet.
        * "number_annotators": number of persons who annotated the tweet. This tag is set to "6".
        * "annotators": a list of the identifiers of the different annotators.
        * "gender_annotators": a list of the genders of the annotators. Possible values are "F" or "M".
        * "age_annotators": a list of the ages of the annotators. Possible values are "18-22", "23-45" or "46+".
        * "value": a list of values indicating if the tweet contains sexist expressions. Each value represents the label assigned by each annotator. Possible values are “YES” or “NO”. 
        * "test_case": tag needed in the PyEvALL library for evaluating classification tasks. This tag is set to “EXIST2023”.

 
* **val_t1_en** and **val_t1_es**. These are the validation files, in English and Spanish respectively, for Task 1. The format is the same that the train files.


* **test_t1_en** and **test_t1_es**. These are the testing files, in English and Spanish respectively, for Task 1. The format is the same for the two files:

	    * "test_case": tag needed in the PyEvALL library for evaluating classification tasks. This tag is set to “EXIST2023”.
        * "id": a unique identifier of the tweet.
        * "text": the text of the tweet.



* **train_t2_en** and **train_t2_es**. These are the train files, in English and Spanish respectively, for Task 2. The format is the same for the two files:

        * "id": a unique identifier of the tweet.
        * "lang": the language of the text. Possible values are “en” or “es”.
        * "tweet": the text of the tweet.
        * "number_annotators": number of persons who annotated the tweet. This tag is set to "6".
        * "annotators": a list of the identifiers of the different annotators.
        * "gender_annotators": a list of the genders of the annotators. Possible values are "F" or "M".
        * "age_annotators": a list of the ages of the annotators. Possible values are "18-22", "23-45" or "46+". 
        * "value": a list of values indicating if the tweet contains sexist expressions. Each value represents the label assigned by each annotator. Possible values are "DIRECT", "REPORTED", "JUDGEMENTAL" or “-” (for non-sexist text). 
        * "test_case": tag needed in the PyEvALL library for evaluating classification tasks. This tag is set to “EXIST2023”.



* **val_t2_en** and **val_t2_es**. These are the validation files, in English and Spanish respectively, for Task 2. The format is the same that the train files.


* **test_t2_en** and **test_t2_es**. These are the testing files, in English and Spanish respectively, for Task 2. The format is the same for the two files:

	    * "test_case": tag needed in the PyEvALL library for evaluating classification tasks. This tag is set to “EXIST2023”.
        * "id": a unique identifier of the tweet.
        * "text": the text of the tweet.


* **train_t3_en** and **train_t3_es**. These are the train files, in English and Spanish respectively, for Task 3. The format is the same for the two files:

       * "id": a unique identifier of the tweet.
        * "lang": the language of the text. Possible values are “en” or “es”.
        * "tweet": the text of the tweet.
        * "number_annotators": number of persons who annotated the tweet. This tag is set to "6".
        * "annotators": a list of the identifiers of the different annotators.
        * "gender_annotators": a list of the genders of the annotators. Possible values are "F" or "M".
        * "age_annotators": a list of the ages of the annotators. Possible values are "18-22", "23-45" or "46+".
        * "value": a list of values indicating if the tweet contains sexist expressions. Each value represents the label assigned by each annotator. Possible values are "IDEOLOGICAL-INEQUALITY", "STEREOTYPING-DOMINANCE",  "OBJECTIFICATION", "SEXUAL-VIOLENCE", "MISOGYNY-NON-SEXUAL-VIOLENCE" or “-” (for non-sexist text). 
        * "test_case": tag needed in the PyEvALL library for evaluating classification tasks. This tag is set to “EXIST2023”.

* **val_t3_en** and **val_t3_es**. These are the validation files, in English and Spanish respectively, for Task 3. The format is the same that the train files.


* **test_t3_en** and **test_t3_es**. These are the testing files, in English and Spanish respectively, for Task 3. The format is the same for the two files:


	    * "test_case": tag needed in the PyEvALL library for evaluating classification tasks. This tag is set to “EXIST2023”.
        * "id": a unique identifier of the tweet.
        * "text": the text of the tweet.


### Formatting Predictions


The prediction files are already included in the dataset folder, and named as:


- EXIST_2023_T1_en.json
- EXIST_2023_T1_es.json
- EXIST_2023_T2_en.json
- EXIST_2023_T2_es.json
- EXIST_2023_T3_en.json
- EXIST_2023_T3_es.json


You must modify these files to include your system predictions as explained below:


* **EXIST_2023_T1_en.json** and **EXIST_2023_T1_es.json**. These are json lists containing your system's predictions for Task 1. Each prediction is a json object with the following format:


       * "test_case": tag needed in the PyEvALL library for evaluating classification tasks. This tag is set to "EXIST2023".
       * "id": the unique identifier of the tweet.
       * "value": your system prediction for the instance, as the set of probabilities of the possible labels, i.e., "YES" and “NO”.


Example of a submission file EXIST_2023_T1_en.json/EXIST_2023_T1_es.json:


```python

[
    {"test_case":"EXIST2023",
     "id":"600001", 
     "value":{
        "YES":0.3334,
        "NO":0.6667}
    }, 

...

    {"test_case":"EXIST2023",
     "id":"600978", 
     "value":{
	   "YES":0.0,
	   "NO":1.0}
    }
]

```

* **EXIST_2023_T2_en.json** and **EXIST_2023_T2_es.json**. These are json lists containing your system's predictions for Task 2. Each prediction is a json object with the following format:


       * "test_case": tag needed in the PyEvALL library for evaluating classification tasks. This tag is set to "EXIST2023".
       * "id": the unique identifier of the tweet.
       * "value": your system prediction for the instance, as the set of probabilities of the possible labels, i.e., "DIRECT", "REPORTED", "JUDGEMENTAL" and "NO".

Example of a submission file EXIST_2023_T2_en.json/EXIST_2023_T2_es.json:

```python

[
    {"test_case":"EXIST2023",
     "id":"600001", 
     "value":{
        "REPORTED": 0.1667,
        "DIRECT": 0.1667,
        "NO": 0.6667,
        "JUDGEMENTAL": 0.0}
    }, 

...

    {"test_case":"EXIST2023", 
     "id": "600978",
     "value":{
        "REPORTED": 1.0,
        "DIRECT": 0.0,
        "NO": 0.0,
        "JUDGEMENTAL": 0.0}
    }
]
```

* **EXIST_2023_T3_en.json** and **EXIST_2023_T3_es.json**. These are json lists containing your system's predictions for Task 3. Each prediction is a json object with the following format:

       * "test_case": tag needed in the PyEvALL library for evaluating classification tasks. This tag is set to "EXIST2023".
       * "id": the unique identifier of the tweet.
       * "value": your system prediction for the instance, as the set of probabilities of the possible labels, i.e., "IDEOLOGICAL-INEQUALITY", "STEREOTYPING-DOMINANCE", "OBJECTIFICATION", "SEXUAL-VIOLENCE", "MISOGYNY-NON-SEXUAL-VIOLENCE" and "NO".


Example of a submission file EXIST_2023_T3_en.json/EXIST_2023_T3_es.json:

```python

[
    {"test_case":"EXIST2023", 
     "id": "600001", 
     "value":{
        "MISOGYNY-NON-SEXUAL-VIOLENCE": 0.1667,
        "IDEOLOGICAL-INEQUALITY": 0.1667,
        "NO": 0.6667,
        "STEREOTYPING-DOMINANCE": 0.0,
        "SEXUAL-VIOLENCE": 0.0,
        "OBJECTIFICATION": 0.0}
    },

...

    {"test_case":"EXIST2023",
     "id":"600978", 
     "value":{
        "MISOGYNY-NON-SEXUAL-VIOLENCE": 0.0,
        "IDEOLOGICAL-INEQUALITY": 0.0,
        "NO": 0.0,
        "STEREOTYPING-DOMINANCE": 0.0,
        "SEXUAL-VIOLENCE": 1.0,
        "OBJECTIFICATION": 1.0}
    }
]
```

