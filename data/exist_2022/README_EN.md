#	DATASET EXIST 2022

### Dataset Description
The EXIST 2022 dataset is composed of tweets and gabs, both in English and Spanish, that may contain sexist expressions.
For a complete description of the dataset, please read: 

Rodríguez-Sánchez, F., Carrillo-de-Albornoz, J., Plaza, L. *et al*. 2022. *Overview of EXIST 2022: sEXism Identification in Social neTworks*. Procesamiento del Lenguaje Natural, 69, pp. 229-240.


### EXIST 2022 Tasks

The EXIST 2022 dataset addresses two tasks:

* **TASK 1: Sexism Identification**. A binary classification task where systems have to decide whether or not a given tweet is sexist.

* **TASK 2: Sexism Categorization**. A five-class, mono-label classification task where each sexist tweet must be classified according to the type of sexism.


### Training, Validation and Test Files


* **train_t1_en** and **train_t1_es**. These are the train files, in English and Spanish respectively, for Task 1. The format is the same for the two files:

        * "test_case": tag needed in the PyEvALL library for evaluating classification tasks. This tag is set to "EXIST2021".
        * "id": a unique identifier of the tweet or gab.
        * "source": the social network where the text was crawled, "twitter" or "gab".
        * "language": the language of the text. Possible values are "en" or "es".
        * "text": the text of the tweet or gab. 
        * "value": indicates if the tweet or gab contains sexist expressions. Possible values are "sexist" or "non-sexist".  

* **val_t1_en** and **val_t1_es**. These are the validation files, in English and Spanish respectively, for Task 1. The format is the same that the train files.


* **test_t1_en** and **test_t1_es**. These are the testing files, in English and Spanish respectively, for Task 1. The format is the same for the two files:

	    * "test_case": tag needed in the PyEvALL library for evaluating classification tasks. This tag is set to "EXIST2022".
        * "id": a unique identifier of the tweet or gab.
        * "text": the text of the tweet.


* **train_t2_en** and **train_t2_es**. These are the train files, in English and Spanish respectively, for Task 2. The format is the same for the two files:

        * "test_case": tag needed in the PyEvALL library for evaluating classification tasks. This tag is set to "EXIST2021".
        * "id": a unique identifier of the tweet or gab.
        * "source": the social network where the text was crawled, "twitter" or "gab".
        * "language": the language of the text. Possible values are "en" or "es".
        * "text": the text of the tweet or gab. 
        * "value": indicates the type of sexism expressed in the text, if any. Possible values are "non-sexist",  "ideological-inequality", "stereotyping-dominance" "objectification",  "sexual-violence",  "misogyny-non-sexual-violence".  


* **val_t2_en** and **val_t2_es**. These are the validation files, in English and Spanish respectively, for Task 2. The format is the same that the train files.


* **test_t2_en** and **test_t2_es**. These are the testing files, in English and Spanish respectively, for Task 2. The format is the same for the two files:

	    * "test_case": tag needed in the PyEvALL library for evaluating classification tasks. This tag is set to "EXIST2022".
        * "id": a unique identifier of the tweet or gab.
        * "text": the text of the tweet.



### Formatting Predictions


The prediction files are already included in the dataset folder, and named as:


- EXIST_2022_T1_en.json
- EXIST_2022_T1_es.json
- EXIST_2022_T2_en.json
- EXIST_2022_T2_es.json


You must modify these files to include your system predictions as explained below:

* **EXIST_2022_T1_en.json** and **EXIST_2022_T1_es.json**. These are json lists containing your system's predictions for Task 1. Each prediction is a json object with the following format:


       * "test_case": tag needed in the PyEvALL library for evaluating classification tasks. This tag is set to "EXIST2022",
       * "id": the unique identifier of the tweet or gab,
       * "value": your system prediction for the instance. Possible values are "sexist" or "non-sexist".


Example of a submission file EXIST_2022_T1_en.json/EXIST_2022_T1_es.json:

```python 
[
    { "test_case":"EXIST2022",
      "id":"1", 
      "value":"non-sexist"
    },

...
    {"test_case":"EXIST2022",
     "id":"527", 
     "value":"sexist"
     }

]
```


* **EXIST_2022_T2_en.json** and **EXIST_2022_T2_es.json**. These are json lists containing your system's predictions for Task 2. Each prediction is a json object with the following format:


        * "test_case": tag needed in the PyEvALL library for evaluating classification tasks. This tag is set to "EXIST2022",
        * "id": the unique identifier of the tweet or gab,
        * "value": your system prediction for the instance. Possible values are "non-sexist",  "ideological-inequality", "stereotyping-dominance" "objectification",  "sexual-violence",  "misogyny-non-sexual-violence".


Example of a submission file EXIST_2022_T2_en.json/EXIST_2022_T2_es.json:

```python 
[
    { "test_case":"EXIST2022",
      "id":"1", 
      "value":"non-sexist"
    },

...
    {"test_case":"EXIST2022",
     "id":"527", 
     "value":"objectification"
     }

]
```