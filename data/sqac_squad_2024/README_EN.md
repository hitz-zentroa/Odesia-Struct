#DATASET FOR THE TASK  SQAC-SQUAD 2024: Question answering 


###	TASK SQAC-SQUAD 2024: Question answering 

This is an extractive text comprehension task formulated as a question-answering task. It consists in answering questions about a text in such a way that the answer is a fragment extracted directly  from the text. The answer to be provided corresponds to the shortest span needed to answer the question. The texts are academic news from CSIC (for Spanish) and Cambridge University (for English). In all cases, the answers are fragments of the text and all questions can be answered from the text.

The task is offered in English and Spanish.  The evaluation is performed on the  SQUAD-SQAC 2024 dataset. SQUAD-SQAC 2024 is an extension of the SQUAD/SQAC datasets. SQAC (Spanish Question Answering Corpus) (Guti�rrez-Fandi�o et al., 2021) is a question-answering dataset, with extractive answers in Spanish. In the associated NLP task, given a question and a paragraph, the system must locate the smallest span (fragment) containing the answer. The methodology used to create it is based on that of SQuAD v1.1 (Stanford Question Answering Dataset) (Rajpurkar et al., 2016), an extractive question-answering dataset in English.
 
The news are of varied scientific domains and are usually short, between 712 and 2,760 words in English,  and between 514 and 2,818 words in Spanish. In addition, they are aimed at the general public, so especialized language is not used.

IMPORTANT: 

Although to create the SQUAD-SQAC 2024 dataset  a similar methodology has been used as in the original SQUAD/SQAC datasets, the types of texts that have been annotated are not the same. The original SQUAD/SQAC datasets contain mostly wikipedia articles, while the SQUAD-SQAC 2024 dataset contains academic news. Additionally, in the SQUAD/SQAC datasets questions are put on fragments of wikipedia texts, while in the SQUAD-SQAC 2024 dataset questions are put on full texts of academic news. The reason is that the news are not too long and making
fragments would result in too short fragments.


### Dataset size

The training and validation files belong to the original SQUAD/SQAC datasets.

- English training: 78839 question/answer pairs
- English validation: 10570 questions/answer pairs

- Spanish training: 15036 question/answer pairs
- Spanish validation: 1864 question/answer pairs

The test files belong to the  SQUAD-SQAC 2024 dataset  and have the following size:

- English: 1182 pairs question/answer from 110 texts.
- Spanish: 1145 pairs question/answer from 110 texts.

### Files provided

Test, training, and validation files in json format are provided:

- test_t1_en.json
- test_t1_es.json
- train_t1_en.json
- train_t1_es.json
- val_t1_en.json
- val_t1_es.json

The files to be submitted are:

- SQAC_SQUAD_2024_en.json
- SQAC_SQUAD_2024_es.json

### Files format

The format of the TRAINING and VALIDATION files is as follows:

[
    {
        "id": "56be4db0acb8001400a502ec",
        "title": "Super_Bowl_50",
        "context": "Super Bowl 50 was an American football game to determine the champion of the National Football League (NFL) for the 2015 season. The American Football Conference (AFC) champion Denver Broncos defeated the National Football Conference (NFC) champion Carolina Panthers 24–10 to earn their third Super Bowl title. The game was played on February 7, 2016, at Levi's Stadium in the San Francisco Bay Area at Santa Clara, California. As this was the 50th Super Bowl, the league emphasized the \"golden anniversary\" with various gold-themed initiatives, as well as temporarily suspending the tradition of naming each Super Bowl game with Roman numerals (under which the game would have been known as \"Super Bowl L\"), so that the logo could prominently feature the Arabic numerals 50.",
        "question": "Which NFL team represented the AFC at Super Bowl 50?",
        "test_case": "SQAC_SQUAD_2024",
        "value": "Denver Broncos"
    },
    {
        "id": "56be4db0acb8001400a502ed",
        "title": "Super_Bowl_50",
        "context": "Super Bowl 50 was an American football game to determine the champion of the National Football League (NFL) for the 2015 season. The American Football Conference (AFC) champion Denver Broncos defeated the National Football Conference (NFC) champion Carolina Panthers 24–10 to earn their third Super Bowl title. The game was played on February 7, 2016, at Levi's Stadium in the San Francisco Bay Area at Santa Clara, California. As this was the 50th Super Bowl, the league emphasized the \"golden anniversary\" with various gold-themed initiatives, as well as temporarily suspending the tradition of naming each Super Bowl game with Roman numerals (under which the game would have been known as \"Super Bowl L\"), so that the logo could prominently feature the Arabic numerals 50.",
        "question": "Which NFL team represented the NFC at Super Bowl 50?",
        "test_case": "SQAC_SQUAD_2024",
        "value": "Carolina Panthers"
		}
]


- "id": a unique identifier of the question.
- "title": a string with the title of the text.
- "context": a string with the fragment of text that the question belongs to.
- "question": a string with the question.
- "test_case": tag needed in the PyEvALL library for evaluating. In this case the value is "SQAC_SQUAD_2024".
- "value":  a string with the fragment of text that is the answer to the question.

The format of the TEST files is the same as in the training files, but without the "value" element: 

[
    {
        "id": "f2f1214b-bd38-4ad6953df31eda69f",
        "title": "COVID-19 showed the importance of genomic surveillance – we need it to help fight antimicrobial resistance",
        "context": "During the COVID-19 pandemic, genomic surveillance proved vital in helping understand the evolution and spread of the SARS-CoV-2 virus. Now, an international group of researchers is calling for its potential to be harnessed to tackle antimicrobial resistance (AMR), a major global challenge that could ultimately result in many more deaths than the coronavirus pandemic.\nAMR already causes substantial sickness and death worldwide, responsible for approximately 1.27 million deaths in 2019. Some estimates suggest that by 2050, it could kill as many as 10 million people each year.\nProfessor Sharon Peacock at the University of Cambridge – the driving force behind the UK’s pioneering COVID-19 Genomics UK Consortium – said: “Over the past century, antibiotics have transformed our ability to treat infection and illness and reduce mortality. But bacteria are becoming increasingly resistant, and with a limited pipeline of new antibiotics, we risk effectively returning to the pre-antibiotic era where we can no longer treat infections.\n“When the world was hit by the COVID-19 pandemic, we showed how powerful a tool genomic surveillance could be in helping us fight back. This work grew out of its increasing application to real-world problems such as detecting outbreaks in hospitals and in the community – including food borne outbreaks. We now need to take what we learned from the pandemic including its bold and largescale use and reapply it to the complex problem of AMR.”\nThe genome, which is ‘written’ in DNA or RNA, consists of a string of nucleotides. Each time a copy of the genome is made, errors can arise – for example, one of the A, C, G and T nucleotides of DNA might get swapped. These changes allow scientists to create lineages – family trees – showing how the genome has evolved and spread. In the case of SARS-CoV-2, they allowed scientists to identify sources of infection, spot so-called ‘variants of concern’ and see whether public health measures such as lockdown, travel restrictions and vaccination were working.\nThe potential to improve surveillance of AMR pathogens may be even higher than for SARS-CoV-2 as the genome data can detect and track outbreaks, provide a prediction for effective antibiotic treatment, reveal the mechanism for resistance including mutations and the acquisition of new DNA, and help understand the movement of resistance mechanisms between bacteria.\nAlthough surveillance of AMR bacteria is already used in some settings, the growing evidence of its potential has largely not translated into routine use. Writing today in The Lancet Microbe, a working group has set out how genomic surveillance could be applied to the problem of AMR more widely, including the barriers that need to be overcome, presenting a series of recommendations including building capacity, training of existing and new workforces, standardising the way that surveillance is done to detect AMR, and agreeing equitable data sharing and governance.\nThe group, which is funded by Wellcome, was initiated by Professor Peacock in conjunction with Wellcome SEDRIC (Surveillance and Epidemiology of Drug Resistance Infection Consortium) and delivered by a team of nearly 100 experts co-led by Professor Kate Baker and Dr Elita Jauneikaite. Five papers will be published in the same edition of the journal, highlighting the breadth of review and analysis undertaken by the team.\nThe series covers multiple areas for the application of genomic AMR surveillance including in hospital settings to help identify outbreaks and inform infection prevention and control and informing clinical decision-making at a patient level. They also highlight applications at a public health level to detect emerging threats and to design and assess suitable interventions like vaccination. It even has the potential to track AMR pathogens moving between humans, animals, and the environment. The team also considered future innovations in genomic surveillance of AMR, looking at how the next phase of genomic technologies and analysis methods might further transform the surveillance landscape.  \nA number of barriers will first need to be overcome, however. These include a lack of resources and political will, and the need for more training, particularly around bioinformatics (the analysis of genome data). There are also practical barriers, including in many countries a weak epidemiological surveillance and microbiology infrastructure, poor supply chains and pricing structures, and issues around effective cooperation and data sharing.\nProfessor Kate Baker, University of Cambridge, said: “We are on the cusp of realising the full potential for genomics in tackling AMR, but there is still a lot of work that needs to be done. We need the scientific, public health and political communities to work together to make this happen. AMR is an urgent problem. It is not something that will happen in years to come – it is happening now.”\nDr Elita Jauneikaite, Imperial College London, said: “We are going to be locked in an ongoing arms race with bacterial pathogens indefinitely. Genomic surveillance offers real promise to help us fight back, providing invaluable information to limit the spread and impact of AMR.”\nProfessor Peacock added: “It was clear from the pandemic that sequencing was a vital tool that was needed in every country worldwide. AMR is a global problem and once again we need to make sure countries worldwide are in a position both to contribute to, and benefit from genomic surveillance data.”\nJanet Midega, AMR Research Lead at Wellcome and SEDRIC Board member, said: “Genomic research and surveillance are pivotal to detect pathogens and understand the transmission and trends of drug resistance in both high- and low-income settings. In order to respond effectively to this data, we need to ensure that the tools being developed are accessible and can be utilised by public health agencies around the world.”\nReference\nBaker, K, et al. Overview: Harnessing genomics for antimicrobial resistance surveillance. The Lancet Microbe; 14 Nov 2023; DOI: 10.1016/S2666-5247(23)00281-1\n\nThe text in this work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License. Images, including our videos, are Copyright ©University of Cambridge and licensors/contributors as identified.  All rights reserved. We make our image and video content available in a number of ways – as here, on our main website under its Terms and conditions, and on a range of channels including social media that permit your use and sharing of our content under their respective Terms.\n",
        "question": "Who should receive more in-depth instruction on how to correctly process genomic information?",
        "test_case": "SQAC_SQUAD_2024"
    },
    {
        "id": "51a117f2-81b2-4a83b6ceff20b74e6",
        "title": "COVID-19 showed the importance of genomic surveillance – we need it to help fight antimicrobial resistance",
        "context": "During the COVID-19 pandemic, genomic surveillance proved vital in helping understand the evolution and spread of the SARS-CoV-2 virus. Now, an international group of researchers is calling for its potential to be harnessed to tackle antimicrobial resistance (AMR), a major global challenge that could ultimately result in many more deaths than the coronavirus pandemic.\nAMR already causes substantial sickness and death worldwide, responsible for approximately 1.27 million deaths in 2019. Some estimates suggest that by 2050, it could kill as many as 10 million people each year.\nProfessor Sharon Peacock at the University of Cambridge – the driving force behind the UK’s pioneering COVID-19 Genomics UK Consortium – said: “Over the past century, antibiotics have transformed our ability to treat infection and illness and reduce mortality. But bacteria are becoming increasingly resistant, and with a limited pipeline of new antibiotics, we risk effectively returning to the pre-antibiotic era where we can no longer treat infections.\n“When the world was hit by the COVID-19 pandemic, we showed how powerful a tool genomic surveillance could be in helping us fight back. This work grew out of its increasing application to real-world problems such as detecting outbreaks in hospitals and in the community – including food borne outbreaks. We now need to take what we learned from the pandemic including its bold and largescale use and reapply it to the complex problem of AMR.”\nThe genome, which is ‘written’ in DNA or RNA, consists of a string of nucleotides. Each time a copy of the genome is made, errors can arise – for example, one of the A, C, G and T nucleotides of DNA might get swapped. These changes allow scientists to create lineages – family trees – showing how the genome has evolved and spread. In the case of SARS-CoV-2, they allowed scientists to identify sources of infection, spot so-called ‘variants of concern’ and see whether public health measures such as lockdown, travel restrictions and vaccination were working.\nThe potential to improve surveillance of AMR pathogens may be even higher than for SARS-CoV-2 as the genome data can detect and track outbreaks, provide a prediction for effective antibiotic treatment, reveal the mechanism for resistance including mutations and the acquisition of new DNA, and help understand the movement of resistance mechanisms between bacteria.\nAlthough surveillance of AMR bacteria is already used in some settings, the growing evidence of its potential has largely not translated into routine use. Writing today in The Lancet Microbe, a working group has set out how genomic surveillance could be applied to the problem of AMR more widely, including the barriers that need to be overcome, presenting a series of recommendations including building capacity, training of existing and new workforces, standardising the way that surveillance is done to detect AMR, and agreeing equitable data sharing and governance.\nThe group, which is funded by Wellcome, was initiated by Professor Peacock in conjunction with Wellcome SEDRIC (Surveillance and Epidemiology of Drug Resistance Infection Consortium) and delivered by a team of nearly 100 experts co-led by Professor Kate Baker and Dr Elita Jauneikaite. Five papers will be published in the same edition of the journal, highlighting the breadth of review and analysis undertaken by the team.\nThe series covers multiple areas for the application of genomic AMR surveillance including in hospital settings to help identify outbreaks and inform infection prevention and control and informing clinical decision-making at a patient level. They also highlight applications at a public health level to detect emerging threats and to design and assess suitable interventions like vaccination. It even has the potential to track AMR pathogens moving between humans, animals, and the environment. The team also considered future innovations in genomic surveillance of AMR, looking at how the next phase of genomic technologies and analysis methods might further transform the surveillance landscape.  \nA number of barriers will first need to be overcome, however. These include a lack of resources and political will, and the need for more training, particularly around bioinformatics (the analysis of genome data). There are also practical barriers, including in many countries a weak epidemiological surveillance and microbiology infrastructure, poor supply chains and pricing structures, and issues around effective cooperation and data sharing.\nProfessor Kate Baker, University of Cambridge, said: “We are on the cusp of realising the full potential for genomics in tackling AMR, but there is still a lot of work that needs to be done. We need the scientific, public health and political communities to work together to make this happen. AMR is an urgent problem. It is not something that will happen in years to come – it is happening now.”\nDr Elita Jauneikaite, Imperial College London, said: “We are going to be locked in an ongoing arms race with bacterial pathogens indefinitely. Genomic surveillance offers real promise to help us fight back, providing invaluable information to limit the spread and impact of AMR.”\nProfessor Peacock added: “It was clear from the pandemic that sequencing was a vital tool that was needed in every country worldwide. AMR is a global problem and once again we need to make sure countries worldwide are in a position both to contribute to, and benefit from genomic surveillance data.”\nJanet Midega, AMR Research Lead at Wellcome and SEDRIC Board member, said: “Genomic research and surveillance are pivotal to detect pathogens and understand the transmission and trends of drug resistance in both high- and low-income settings. In order to respond effectively to this data, we need to ensure that the tools being developed are accessible and can be utilised by public health agencies around the world.”\nReference\nBaker, K, et al. Overview: Harnessing genomics for antimicrobial resistance surveillance. The Lancet Microbe; 14 Nov 2023; DOI: 10.1016/S2666-5247(23)00281-1\n\nThe text in this work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License. Images, including our videos, are Copyright ©University of Cambridge and licensors/contributors as identified.  All rights reserved. We make our image and video content available in a number of ways – as here, on our main website under its Terms and conditions, and on a range of channels including social media that permit your use and sharing of our content under their respective Terms.\n",
        "question": "Do scientists wish to study, research, and monitor the spread of viruses and their propensity to withstand antibiotics in wealthy or poor contexts?",
        "test_case": "SQAC_SQUAD_2024"
		}
]


### Submission format 

The prediction files must have the following names:

- In Spanish it should be named:  SQAC_SQUAD_2024_es.json 
- In English it should be named:  SQAC_SQUAD_2024_en.json 

The content of the file must be a json list with the following format:

- "test_case": tag needed in the PyEvALL library for evaluating classification tasks. This tag is set to "SQAC_SQUAD_2024".
- "id": the unique identifier of the question.
- "value": a string with the fragment of text that is the answer to the question.


### Example of a submission file:  SQAC_SQUAD_2024_en.json
 
 [
    {
        "id":"f2f1214b-bd38-4ad6953df31eda69f",
        "test_case":"SQAC_SQUAD_2024",
        "value":"anything"
    },
    {
        "id":"51a117f2-81b2-4a83b6ceff20b74e6",
        "test_case":"SQAC_SQUAD_2024",
        "value":"anything"
		}
]
