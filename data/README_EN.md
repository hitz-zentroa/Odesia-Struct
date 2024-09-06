# ODESIA Leaderboard Challenge 2024
https://leaderboard.odesia.uned.es/en/leaderboard/challenge

This is the README of the participants package of the ODESIA Leaderboard Challenge 2024.

This challenge aims to promote the development and **evaluation of
language models in Spanish** using the evaluation platform and
datasets provided by the ODESIA project (Espacio de Observación del
Desarrollo del Español en la Inteligencia Artificial). The challenge
consists of solving 10 discriminative tasks in Spanish, that belong to
the **ODESIA Leaderboard v2** (https://leaderboard.odesia.uned.es) and
are evaluated on private data. The ODESIA Leaderboard is an
application that provides an evaluation infrastructure for pre-trained
language models in English and Spanish that allows a direct comparison
between the performance of models in one and the other language. The
**10 tasks** of this challenge, with **private evaluation data**,
belong to the ODESIA-CORE section of the leaderboard. Additionally,
the leaderboard has an ODESIA-EXTENDED section with 4 tasks with
pre-existing public evaluation data, but these are not part of the
challenge. Although for all tasks there are bilingual data (Spanish
and English), this challenge focuses only on the Spanish tasks
(Spanish portion of ODESIA-CORE). More information about the challenge
is to be found on the website.

The tasks are performed on 5 different datasets, which have been developed for the leaderboard. This package contains the training and validation partitions of the datasets for each of the tasks, as well as the test partitions  (without gold labels) and the prediction templates per task. Both the English and Spanish datasets are provided, even if the challenge focuses only on Spanish. Throughout this README the contents of this package are described. Additionally, each dataset has its specific README with a description of the dataset and the tasks that apply to it.


## Package content

The challenge.zip package is composed of 5 folders, one for each dataset:

* diann_2023
* dipromats_2023
* exist_2022
* exist_2023
* sqac_squad_2024

Each dataset folder contains a README and the following files for each of the tasks that are performed on the dataset, and for each of the languages of the dataset (Spanish and English), even if the challenge does not include the English tasks:

* training partition
* validation partition
* test partition (without gold labels)
* predictions template

The datasets included are the following:

* DIANN 2023, which is a collection of abstracts of biomedical articles.

* DIPROMATS 2023, which is composed of tweets issued by diplomats from four world powers (the European Union, Russia, China and the United States). 

* EXIST 2022, which is composed of tweets.

* EXIST 2023, which is composed of tweets. This dataset is created following the Learning with Disagreement (LeWiDi) paradigm (Uma et al., 2021a).

* SQAC-SQUAD 2024, which is composed of popular science articles from CSIC for Spanish and Cambridge University for English. 

A more detailed description of the tasks per dataset, references to the papers, description of the labels, etc., can be found in the README of each dataset and in the challenge website.


## Publication

Please, cite this paper if you use the datasets contained in this package:


@inproceedings{DBLP:conf/sepln/AmigoCFGLMMPPSS23,
  author       = {Enrique Amig{\'{o}} and
                  Jorge Carrillo{-}de{-}Albornoz and
                  Andr{\'{e}}s Fern{\'{a}}ndez and
                  Julio Gonzalo and
                  Miguel Lucas and
                  Guillermo Marco and
                  Roser Morante and
                  Jacobo Pedrosa and
                  Laura Plaza and
                  Ibo Sanz and
                  Be{\~{n}}at San{-}Sebasti{\'{a}}n},
  editor       = {Eugenio Mart{\'{\i}}nez{-}C{\'{a}}mara and
                  Arturo Montejo{-}R{\'{a}}ez and
                  Mar{\'{\i}}a Teresa Mart{\'{\i}}n Valdivia and
                  Luis Alfonso Ure{\~{n}}a L{\'{o}}pez},
  title        = {{ODESIA:} Space for Observing the Development of Spanish in Artificial
                  Intelligence},
  booktitle    = {Proceedings of the Annual Conference of the Spanish Association for
                  Natural Language Processing 2023: Projects and System Demonstrations
                  {(SEPLN-PD} 2023) co-located with the International Conference of
                  the Spanish Society for Natural Language Processing {(SEPLN} 2023),
                  Ja{\'{e}}n, Spain, September 27-29, 2023},
  series       = {{CEUR} Workshop Proceedings},
  volume       = {3516},
  pages        = {50--54},
  publisher    = {CEUR-WS.org},
  year         = {2023},
  url          = {https://ceur-ws.org/Vol-3516/paper11.pdf},
  timestamp    = {Thu, 26 Oct 2023 16:59:55 +0200},
  biburl       = {https://dblp.org/rec/conf/sepln/AmigoCFGLMMPPSS23.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}


