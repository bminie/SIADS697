# Team Care: Hospital Recommendation System
## Overview
Elective procedures in the US have been frequently canceled over the past 2 years since COVID began; just as things were returning to normal, hospitals are now facing capacity issues and canceling elective procedures. 
However, disease has no regard for COVID and continues to adversely human health, untreated. 
Receiving healthcare is already an overly opaque and complex process - recent events have only made things more difficult. 
We've created an open-source, web-based tool to help people navigate which US hospitals to get treatment at for elective procedures based on value-based care measures and convenience. 
Our recommendation engine takes user input (e.g. type of procedure needed, location, etc), along with weighted user priorities for model features (e.g. cost of care, quality of care, etc), to provide a “top 3 hospitals” to attain care with regard to COVID-19 (or future disease) prevalence in various geographies.

To build the web-based tool we are using the Streamlit python library as it enables rapid prototyping while producing production-quality dashboards. User will easily be able to enter parameters into the web app and visualize the results via tables and maps.
## Getting Started
### Create Virtual Environment
It's highly recommended that you use a virtual environment when working with the app with conda being recommended.

Using either the requirements.txt or environemnt.yml file, create a virtual environment using conda
```commandline
conda env create -f environment.yml
conda create --name <env_name> --file requirements.txt
```
### Starting the Streamlit App
Ensure that you are in the app directory and run the following command to start the streamlit app.

The first command to activate the virtual environment is what will be used if you used the environment.yml to create the conda environment.

The command may different slightly depending on your OS and conda version. Update the command appropriately for your OS and conda version.
```commandline
conda activate team_care
streamlit run streamlit_app.py
```
### Streamlit App Usage
Fill this section in with information about using the app and parameter explanations. Would be good to include pictures here.
## Recommendation System Overview
Our recommendations are based off of the cosine similarity measurement. 
Using the user inputs as our query vector, we calculate the cosine similarity for all hospitals in the data set and return our top 3 as our recommendation.
Users also have the ability to specify how important each feature is to them and these weights are taken into account during the cosine similarity determination.
## Application Data Overview
Current data in data folder was acquired on February 23, 2022 by manually downloading files. In the future, these files should be automatically pulled down using API's or other access methods in order to be working off the most recent data.
* Community_Profile_Report_Counties.csv
    * County level COVID-19 data pulled from https://protect-public.hhs.gov/
* Community_Profile_Report_Counties.zip
    * County level COVID-19 data pulled from https://protect-public.hhs.gov/ . This is a zipped GEOJSON file that includes coordinates to draw counties on a map. File as zipped due to file size restrictions with standard GitHub account
* hospitalizations.csv
    * Hospitalization data collected from state websites by the COVID-19 Hospitalization Tracking Project Team, pulled from https://carlsonschool.umn.edu/mili-misrc-covid19-tracking-project/download-data
* hospitalizations.json
    * Hospitalization data collected from state websites by the COVID-19 Hospitalization Tracking Project Team, pulled from https://carlsonschool.umn.edu/mili-misrc-covid19-tracking-project/download-data
* hospitalizations.xlsx
    * Hospitalization data collected from state websites by the COVID-19 Hospitalization Tracking Project Team, pulled from https://carlsonschool.umn.edu/mili-misrc-covid19-tracking-project/download-data
* Data from The Centers for Medicare & Medicaid Services website
    * Hospital Readmissions Reduction Program: https://data.cms.gov/provider-data/dataset/9n3s-kdb3
    * Hospital-Acquired Condition (HAC) Reduction Program: https://data.cms.gov/provider-data/dataset/yq43-i98g
    * Unplanned Hospital Visits - Hospital: https://data.cms.gov/provider-data/dataset/632h-zaca
    * Payment and value of care - Hospital: https://data.cms.gov/provider-data/dataset/c7us-v4mf
    * Patient survey (HCAHPS) - Hospital: https://data.cms.gov/provider-data/dataset/dgck-syfz
    * Hospital Value-Based Purchasing (HVBP) - Efficiency Scores:https://data.cms.gov/provider-data/dataset/su9h-3pvj
    * CMS Stars: https://data.cms.gov/provider-data/dataset/xubh-q36u
