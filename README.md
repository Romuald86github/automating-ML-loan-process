This project is aimed at predicting loan approvals using machine learning models and providing an interactive web interface for users to input their loan details and receive a prediction.

# Getting Started

Follow these steps to set up and run the project on your local machine.

## Prerequisites
Make sure you have the following installed:

Python (version 3.7 or higher)
Pip (Python package installer)
Git (optional, for cloning the repository)


# Installation
## 1.Clone the Repository:

Clone this repository to your local machine using Git (if you have Git installed) or download the ZIP file and extract it

```
git clone https://github.com/Romuald86github/automating-ML-loan-process.git
cd automating-ML-loan-process
```


## 2.Set Up Python Virtual Environment:

It's recommended to use a Python or conda virtual environment to manage dependencies.

### python
```
python -m venv loan
source loan/bin/activate  # For Linux/Mac
loan\Scripts\activate     # For Windows

```

### conda
```
# conda init (if you ar not in a conda-like environment, make sure you install anaconda before executing this script)
conda create -n 'environment_name'
conda activate 'environment_name'
```




## 3.Install Required Packages:

Install the necessary Python packages using Pip.

```
pip install -r requirements.txt
```



# Data Generation and Processing
## 1.Generate Synthetic Loan Data:

Run the script to generate synthetic loan data with various attributes.


```
python src/data/data_loader.py
```


## 2.Clean and Preprocess Data:

Clean the raw loan data by handling missing values, ensuring data types consistency, and saving the cleaned data.

```
python src/data/data_cleaning.py
```


# Feature Selection and Model Training
## 1.Feature Selection:

Perform feature selection to choose relevant features for training the model.

```
python src/features/feature_selection.py
```


## 2.Train and Evaluate Models:

Train multiple machine learning models, perform hyperparameter tuning, and select the best-performing model

```
python src/models/train_model.py
```

This step saves the best-performing model, preprocessing pipeline, and selected feature names to the models/ directory.


# Running the Streamlit App
## 1.Run the Streamlit App:

Start the Streamlit web application for loan prediction.

```
streamlit run src/app/app.py
```


# you can also run the entire project by running the following single script:

```
run_project.py
```



Open your web browser and navigate to the URL provided by Streamlit (usually http://localhost:8501) to interact with the loan prediction app.


# Accessing the Loan Prediction App
The loan prediction app is accessible via the following link:

[Loan Prediction App](http://localhost:8501/)



# Additional Notes
Project Structure: The project follows a modular structure with separate directories (src/data, src/features, src/models, src/app) for different components of data handling, feature engineering, model training, and the Streamlit application.

Dependencies: All Python dependencies are listed in requirements.txt. Ensure they are installed in your virtual environment to run the project smoothly.

Troubleshooting: If you encounter any issues during installation or execution, ensure that your environment matches the prerequisites and that all steps were followed correctly.


