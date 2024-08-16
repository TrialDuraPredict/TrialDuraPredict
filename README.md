<h1 align="center">TrialDuraPredict</h1>
<p align="center"><i>Predicting and Interpreting Clinical Trial Duration Using Machine Learning to Enhance Trial Design</i></p>

## Table of Contents
- [Description](#description)
- [Getting Started](#getting-started)
- [Contact](#contact)
- [Acknowledgments](#acknowledgments)

## Description
This project is meant to develop machine learning methods to predict clinical trial duration based on registered study information including study description, enrollment, study type, eligibility criteria, study plan, start date, and so on. Simultaneously, we seek to understand the contributions of various features learned by the model to prolonged trial durations and provide recommendations to optimize study protocols and appropriately reduce trial duration.
Since many clinical trials are performed and they are lengthy with increasing costs as time goes on, any improvement in design protocols would allow for more efficient trials, with faster phases and even follow up studies, to be performed. Markey et al. (2024) have noted many external factors such as regulatory changes and increasing competitive pressures, which in turn may have a hand in unnecessarily increasing the overall complexity of these trials. With this in mind, we can use our project to help with assessing what makes for faster and cheaper turnarounds that do not sacrifice the scientific method so that interventional treatments can more swiftly be tested and released that may drastically improve healthcare outcomes.


## Getting Started
### Installation
- Clone the repo `git clone https://github.com/TrialDuraPredict/TrialDuraPredict`, and go to repo dir `cd TrialDuraPredict`
- Create a virtual environment `virtualenv venv`, and activate it `source venv/bin/activate`
- Install packages `pip install -r requirements.txt`

This installation is designed for macOS or Linux. Please adjust the commands accordingly if you are using Windows.

### Raw Data
The raw data in JSON format can be downloaded from [ClinicalTrial.gov](https://clinicaltrials.gov/data-api/how-download-study-records). The dataset for this project, dated July 1st, 2024, comprises 500,534 study records. Each record provides comprehensive information about the trial, including NCT ID (study identifiers), study description, participant enrollment (inclusion/exclusion criteria), treatment or intervention methods, disease or condition being studied, outcome measures, among others. Below is a brief overview of the key information used to predict the duration of the trial.\
\
![clinical_trial_overview](assets/clinical_trial.jpg)
**Raw data for all studies are included in the [Google Drive](https://drive.google.com/drive/folders/1j2HmWfcUaOqDSOtjxndmW5EGHvuZjTA_?usp=sharing)**

### Data Cleaning
We use example data for this tutorial, which includes 1000 studies in *./data_example/ctg-studies.json*. Firstly direct to *TrialDuraPredict* folder,
- To generate the **training**, **test** and **incompleted** study IDs , run the code `python ./src_example/study_ids_split.py`. Datasets including separate study IDs are generated in *./results_example/*
- To generate the **clinical duration** (output, Months as unit), run the code `python ./src_example/trial_duration.py`. Duration dataset is generated in *./results_example/trial_duration.csv*
- To generate the embedding of **clinical description**, run the code `python ./src_example/description2embedding.py`. The embedding dataset is generated in *./results_example/description2embedding.pkl*
- To generate the embedding of **inclusion_eligibility** and **exclusion_eligibility**, run the code `python ./src_example/eligibility2embedding.py`. The embedding dataset is generated in *./results_example/eligibility2embedding.pkl*
- To generate the embedding of **treatment**, run the code `python ./src_example/treatment2embedding.py`. The embedding dataset is generated in *./results_example/treatment2embedding.pkl*
- To generate the embedding of **disease**, run the code `python ./src_example/disease2embedding.py`. The embedding dataset is generated in *./results_example/disease2embedding.pkl*
- To generate the embedding of **outcome measures**, run the code `python ./src_example/outcome2embedding.py`. The embedding dataset is generated in *./results_example/outcome2embedding.pkl*

Once these have all been completed,
- Combine together and split into train/test/incompleted datasets found in *./results_example/X_train (or X_test, X_incompleted, y_train, y_test, y_incompleted).pkl* using the code `python ./src_example/combine_split_dfs.py`
- Run PCA analysis to reduce the dimensions found in *./results_example/X_train_pca (or X_test_pca, X_incompleted_pca).pkl* using the code `python ./src_example/reduce_dimention_dfs.py`

**Codes to clean data for all studies are included in *./src* folder.**

### Modeling
We tried a few different ML algorithms for the data training. Below are mdoels trained on the 1000 studies sample data:
- Run `python ./src_example/model_lr.py` to generate the optimized linear regression model in *./results_example/model_lr.sav*
- Run `python ./src_example/model_rf.py` to generate the optimized random forest model in *./results_example/model_rf.sav*
- Run `python ./src_example/model_xgb.py` to generate the optimized XGBoost model in *./results_example/model_xgb.sav*
- Run `python ./src_example/model_ffnn.py` to generate the feedforward neural network model in *./results_example/model_ffnn.keras*
- Run `python ./src_example/model_cnn.py` to generate the convolutional neural network model in *./results_example/model_cnn.keras*

**Final model results for all studies are included in the [Google Drive](https://drive.google.com/drive/folders/10naZGa5eEZjSpfilxRpIHefsXObLGXeO?usp=drive_link)**

Downloading the model results and loading them for trial duration predictions for incompleted studies can be found in the `src/all_model_predictions.ipynb` Jupyter Notebook, which exports the predicted durations by NCTID into `results/incompleted_preds_df.csv`. In this notebook, we also evaluated the models and scored the predicitons to determine which model best fits our data.

### Interactive Dashboard
To create and use the interactive Plotly Dash dashboard, run the `src/TrialDuraPredict_Dashboard.ipynb` Jupyter Notebook through and find the dashboard at the bottom cell. Instructions and filtering suggestions will be displayed above the DataTable and other interactive outputs. The initial cells download, setup, and merge the `study_info_df.csv`and `results/incompleted_preds_df.csv` dataframes pulled from the Google Drive and created from the models, respectively, listed above if the file is not currently present on your local machine since it is relatively large and necessitates the use of the `gdown` library to bypass the large file warning.

## Contact
Created by TrialDuraPredict Team.

## Acknowledgments
[TrialDura](https://arxiv.org/pdf/2404.13235)\
[HINT: clinical trial outcome prediction](https://github.com/futianfan/clinical-trial-outcome-prediction)\
[clinical-trial-prediction](https://github.com/lenlan/clinical-trial-prediction/tree/main)
