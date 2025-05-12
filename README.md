# ✈️ MLFlow Data Production Pipeline for Flight Delay Prediction

## Project Overview
This project builds a data production pipeline using **MLFlow** to predict flight delays at **LAX**. The pipeline consists of data importation, filtering, cleaning, and applying a polynomial regression model to forecast delays. The pipeline is modular, with each stage split across different scripts.

## Pipeline Architecture
The pipeline consists of the following key components:
- main.py: The entry point of the pipeline, which coordinates the flow of data through the stages of the pipeline.
- steps/: The folder containing the core scripts of the pipeline:
    - b_import_and_format.py: Handles the importation and formatting of raw data.
    - c_filter_and_clean.py: Filters and cleans the data, preparing it for model training.
    - d_ml_experiment.py: Trains the polynomial regression model and logs the experiment results using MLFlow.
- MLProject.ipynb: This notebook showcases the entire pipeline.

## Methodology
- **MLFlow** manages the machine learning pipeline, from importing and cleaning data to training and evaluating the model.
- **Polynomial Regression** is used to predict flight delays based on various features such as scheduled departure and destination.
- **DVC** is utilized for version control of the data and model outputs.

## Key Findings
- A successful data production pipeline was created and executed, integrating MLFlow for tracking experiment parameters, metrics, and performance plots, ensuring reproducibility.
- The model predicted flight delays with a **mean squared error** of **115.74** and an **average delay** of **10.76 minutes**.

## Technologies Used
- Python
- MLFlow
- DVC

## Full Report
To explore the full analysis, including code and key takeaways, view the complete [Jupyter Notebook](./MLProject.ipynb).