# DATASET DIPROMATS 2023

DIPROMATS is a dataset of tweets in Spanish and English issued by
diplomats and authorities of four world powers: USA, Russia, China and
the EU. The tweets have been annotated with a binary label that
indicates whether the tweet contains some kind of propaganda or not, and with the type of propaganda technique used, both with coarse-grained (four categories) and fine-grained (fifteen subcategories) categories. 


Three tasks are performed on this dataset:

- Task 1 - Propaganda Identification: It consists on  determining whether in a tweet propaganda techniques are used or not. This is a classification task and the labels are "true" or "false".

- Task 2 - Coarse propaganda characterisation: The task seeks to classify the tweet into four classes of propaganda techniques: appeal to commonality, discrediting the opponent, loaded language, appeal to authority.

- Task 3 - Fine-grained propaganda characterisation. The task consists on  categorising propagandistic tweets in 15 propaganda techniques: Flag Waving, Ad Populum / Ad antiquitatem, Name Calling/Labelling, Undiplomatic Assertiveness / Whataboutism, Appeal to Fear, Doubt, and Loaded Language.

More details about the tasks can be found in the publication:

- Pablo Moral, Guillermo Marco, Julio Gonzalo, Jorge Carrillo-de-Albornoz, Iván Gonzalo-Verdugo (2023) Overview of DIPROMATS 2023: automatic detection and characterisation of propaganda techniques in messages from diplomats and authorities of world powers. Procesamiento del Lenguaje Natural, Revista nº 71, Septiembre de 2023, pp. 397-407.

### Dataset size

Number of tweets in training, validation and test:

- English		6726	1682		3604
- Spanish		4896	1224		3470


### Files provided

Test, training, and validation files in json format are provided for English and Spanish.

- test_t1_en.json
- test_t1_es.json
- test_t2_en.json
- test_t2_es.json
- test_t3_en.json
- test_t3_es.json
- train_t1_en.json
- train_t1_es.json
- train_t2_en.json
- train_t2_es.json
- train_t3_en.json
- train_t3_es.json
- val_t1_en.json
- val_t1_es.json
- val_t2_en.json
- val_t2_es.json
- val_t3_en.json
- val_t3_es.json

The files to be submitted are:

- DIPROMATS_2023_T1_en.json
- DIPROMATS_2023_T1_es.json
- DIPROMATS_2023_T2_en.json
- DIPROMATS_2023_T2_es.json
- DIPROMATS_2023_T3_en.json
- DIPROMATS_2023_T3_es.json


### Files format

The TRAINING and VALIDATION files have the following format:

- Task 1

[
    {
        "test_case": "DIPROMATS2023",
        "id": "7288",
        "country": "European Union",
        "user_name": "josepborrellf",
        "tweet_type": "Tweet",
        "tweet_id": 1308804332725338112,
        "UTC": "2020-09-23 16:24:03+00:00",
        "rts&fav": 187,
        "language": "en",
        "text": "Read my new blog post on Ukraine: https://t.co/JkcKuun67L \nDuring my first visit in Ukraine as HR/VP, I reaffirmed the EU’s strong support to Ukraine and our continued non-recognition of the illegal annexation of Crimea. https://t.co/BDSD7MvDyh",
        "value": "false"
    },
    {
        "test_case": "DIPROMATS2023",
        "id": "5749",
        "country": "USA",
        "user_name": "usambisrael",
        "tweet_type": "Tweet",
        "tweet_id": 1291041977199472643,
        "UTC": "2020-08-05 16:02:48+00:00",
        "rts&fav": 2925,
        "language": "en",
        "text": "Given how Iran used JCPOA proceeds to develop ballistic missiles and advance terrorism across Yemen, Syria, Iraq &amp; Lebanon, maintains its mantra of “Death to Israel,” and was caught red-handed breaching the deal itself, this position is unfathomable. https://t.co/JOUoONejxc",
        "value": "false"
        ]
		}
		
]

        - "test_case": string with the tag needed in the PyEvALL library for evaluating classification tasks. This tag is set to "DIPROMATS2023".
        - "id":  a unique identifier of the tweet in this dataset.
        - "country":  string with the country that the diplomat belongs to.
        - "user_name": string with the username of the diplomat.
        - "tweet_type": string with the type of tweet ("Tweet", "Reply", "Retweet", "quoted").
        - "tweet_id":identifier of the tweet.
        - "UTC": string with the time that the tweet was posted. Format: "2020-08-05 16:02:48+00:00".
        - "rts&fav": number of retuits and likes.
        - "language": string with the language of the tweet 
        - "text": string with the text of the tweet.
        - "value": a string indicating whether the tweet contains propaganda or not. Possible values: "false", "true".


- Task 2

[
    {
        "test_case": "DIPROMATS2023",
        "id": "2945",
        "country": "Russia",
        "user_name": "dpol_un",
        "tweet_type": "quoted",
        "tweet_id": 1257138215502458880,
        "UTC": "2020-05-04 02:41:21+00:00",
        "rts&fav": 235,
        "language": "en",
        "text": "We categorically and consistently denounce any attempts to interfere in #Venezuela internal affairs. This incursion should be thoroughly investigated. It clearly goes in the direction set up by recent #US provocations and rethoric which #Russia clearly condemns #HandsOffVenezuela https://t.co/riz02FO4F5",
        "value": [
            "3 loaded language"
        ]
    },
    {
        "test_case": "DIPROMATS2023",
        "id": "4812",
        "country": "USA",
        "user_name": "state_sca",
        "tweet_type": "quoted",
        "tweet_id": 1279162012178755592,
        "UTC": "2020-07-03 21:16:04+00:00",
        "rts&fav": 597,
        "language": "en",
        "text": "Today, the US reaffirmed its commitment to helping Pakistan fight COVID-19 with a delivery of 100 new U.S.-produced ventilators that will help save lives. Our partnership with Pakistan is strong and we can defeat this together. https://t.co/g5EFqUDbGF",
        "value": [
            "false"
			]
	}
]


        - "test_case": string with the tag needed in the PyEvALL library for evaluating classification tasks. This tag is set to "DIPROMATS2023".
        - "id":  a unique identifier of the tweet in this dataset.
        - "country":  string with the country that the diplomat belongs to.
        - "user_name": string with the username of the diplomat.
        - "tweet_type": string with the type of tweet (Tweet, Reply, Retweet, quoted).
        - "tweet_id":identifier of the tweet.
        - "UTC": string with the time that the tweet was posted. Format: "2020-08-05 16:02:48+00:00".
        - "rts&fav": number of retuits and likes.
        - "language": string with the language of the tweet 
        - "text": string with the text of the tweet.
        - "value": a list of values indicating the type of propaganda  technique used in the tweet. Possible values: false, appeal to commonality, discrediting the opponent, loaded language, appeal to authority.



- Task 3

[
    {
        "test_case": "DIPROMATS2023",
        "id": "8315",
        "country": "China",
        "user_name": "zlj517",
        "tweet_type": "Tweet",
        "tweet_id": 1321781855671050240,
        "UTC": "2020-10-29 11:52:06+00:00",
        "rts&fav": 620,
        "language": "en",
        "text": "The 200-year history of the #US is filled with its hegemonic interference in others' affairs. It forces its values on others, and is \"the most warlike nation in the history of the world\" as former President #Carter said. https://t.co/d5cdPOqpUD",
        "value": [
            "2 discrediting the opponent - name calling",
            "2 discrediting the opponent - undiplomatic assertiveness/whataboutism",
            "3 loaded language"
        ]
    },
    {
        "test_case": "DIPROMATS2023",
        "id": "6596",
        "country": "China",
        "user_name": "caoyi_mfa",
        "tweet_type": "Tweet",
        "tweet_id": 1301520930326417409,
        "UTC": "2020-09-03 14:02:25+00:00",
        "rts&fav": 11,
        "language": "en",
        "text": "Anyone or any force who attempts to distort the history of the Chinese Communist Party and vilify the nature and purpose of the Chinese Communist Party, the Chinese people will never agree!",
        "value": [
            "1 appeal to commonality - ad populum"
        ]
		}
]


        - "test_case": string with the tag needed in the PyEvALL library for evaluating classification tasks. This tag is set to "DIPROMATS2023".
        - "id":  a unique identifier of the tweet in this dataset.
        - "country":  string with the country that the diplomat belongs to.
        - "user_name": string with the username of the diplomat.
        - "tweet_type": string with the type of tweet (Tweet, Reply, Retweet, quoted).
        - "tweet_id":identifier of the tweet.
        - "UTC": string with the time that the tweet was posted. Format: "2020-08-05 16:02:48+00:00".
        - "rts&fav": number of retuits and likes.
        - "language": string with the language of the tweet 
        - "text": string with the text of the tweet.
        - "value": a list of values indicating the type of propaganda
          technique used in the tweet. Possible values: false, Flag Waving, Ad Populum / Ad antiquitatem, Name Calling/Labelling, Undiplomatic Assertiveness / Whataboutism, Appeal to Fear, Doubt, and Loaded Language.


The TEST file has the following format for the three tasks:


[
    {
        "test_case": "DIPROMATS2023",
        "id": "8408",
        "text": "A vote for me and the Republican Party is a vote for the American Dream! Over the next four years, we will make America into the Manufacturing Superpower of the World, and we will end our reliance on China once and for all. https://t.co/gsFSghkmdM https://t.co/S7PAIqWd0y"
    },
    {
        "test_case": "DIPROMATS2023",
        "id": "8409",
        "text": "Met with Chairman of the Centre for Conflict and Peace Studies and Former Deputy Foreign Minister @HekmatKarzai yesterday. We exchanged ideas on the #AfghanPeaceProcess and friendly bilateral relations. China firmly supports the process that is \"Afghan led and Afghan owned\". https://t.co/nisXbSxZhA"
		}
]


        - "test_case": string with the tag needed in the PyEvALL library for evaluating classification tasks. This tag is set to "DIPROMATS2023".
        - "id":  a unique identifier of the tweet in this dataset.
        - "text": string with the text of the tweet.


### Submission format 

The prediction files must have the following names:

- In Spanish it should be named: 

    DIPROMATS_2023_T1_es.json
    DIPROMATS_2023_T2_es.json
    DIPROMATS_2023_T3_es.json

- In English it should be named: 


    DIPROMATS_2023_T1_en.json
    DIPROMATS_2023_T2_en.json
    DIPROMATS_2023_T3_en.json


The content of the file must be a json list with the following format:

- "test_case": tag needed in the PyEvALL library for evaluating classification tasks. This tag is set to "DIPROMATS2023".
- "id": the unique identifier of the tweet in this dataset.
- "value" for Task 1: a string with the system prediction.
- "value" for Task 2: a list with the system prediction(s) of coarse labels. IMPORTANT: Having an empty list will cause an error. If you do not wish to make predictions for a specific item, you should remove that item from the file of predictions.
- "value" for Task 3: a list with the system prediction(s) of fine-grained labels. IMPORTANT: Having an empty list will cause an error. If you do not wish to make predictions for a specific item, you should remove that item from the file of predictions.


### Example of a submission file:  DIPROMATS_2023_T1_es.json

[
    {
        "test_case": "DIPROMATS2023",
        "id": "8408",
        "value": "false"
    },
    {
        "test_case": "DIPROMATS2023",
        "id": "8409",
        "value": "true"
	}
]

### Example of a submission file:  DIPROMATS_2023_T2_en.json

[
    {
        "test_case": "DIPROMATS2023",
        "id": "8407",
        "value": [
			"false"
		]
    },
    {
        "test_case": "DIPROMATS2023",
        "id": "8409",
        "value": [
		            "1 appeal to commonality",
					"2 discrediting the opponent",
					"3 loaded language"
					]
		}
]

### Example of a submission file: DIPROMATS_2023_T3_en.json

[
    {
        "test_case": "DIPROMATS2023",
        "id": "8460",
        "value": ["false"]
    },
    {
        "test_case": "DIPROMATS2023",
        "id": "8461",
        "value": ["1 appeal to commonality - flag waving",
		    "1 appeal to commonality - ad populum"
	             	]
		}
]


