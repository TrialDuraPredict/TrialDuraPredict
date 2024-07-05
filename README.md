<h1 align="center">TrialDuraPredict</h1>
<p align="center"><i>Predicting and Interpreting Clinical Trial Duration Using Machine Learning to Enhance Trial Design</i></p>

## Table of Contents
- [Description](#description)
- [Getting Started](#getting-started)
- [Contact](#contact)
- [Acknowledgments](#acknowledgments)

## Description
This project is to develop machine learning methods to predict clinical trial duration based on registered study information, including study description, enrollment, study type, eligibility criteria, study plan, start date and so on. Simultaneously, we seek to understand the contributions of various features to prolonged trial durations and provide recommendations to optimize study protocols and reduce trial duration.

## Getting Started
### Installation
- Clone the repo `git clone https://github.com/TrialDuraPredict/TrialDuraPredict`, and go to repo dir `cd TrialDuraPredict`
- Create a virtual environment `virtualenv venv`, and activate it `source venv/bin/activate`
- Install packages `pip install -r requirements.txt`

This installation is designed for macOS or Linux. Please adjust the commands accordingly if you are using Windows.

### Raw Data
The raw data can be downloaded from [ClinicalTrial.gov](https://clinicaltrials.gov/). The dataset for this project, dated July 1st, 2024, comprises 500,534 study records. Each record provides comprehensive information about the trial, including NCT ID (study identifiers), study description, participant enrollment (inclusion/exclusion criteria), treatment or intervention methods, disease or condition being studied, outcome measures, among others. Below is a brief overview of the key information used to predict the duration of the trial.\
\
![clinical_trial_overview](assets/clinical_trial.jpg)

### Data Cleaning
Example data includes 1000 studies and are stored as separate json files in *./data_example/ctg-studies.json*. To clean the data, direct to *TrialDuraPredict* and run the code `python ./src/data_clean.py`. Cleaned dataset is generated in *./data_example/data_cleaned.csv*


```bash
xxx
```

## Contact
Created by TrialDuraPredict Team. Contact us at: xxx

## Acknowledgments
Inspiration, etc\
[TrialDura](https://arxiv.org/pdf/2404.13235)\
[HINT: clinical trial outcome prediction](https://github.com/futianfan/clinical-trial-outcome-prediction)\
[clinical-trial-prediction](https://github.com/lenlan/clinical-trial-prediction/tree/main)
