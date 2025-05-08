# ✈️ MLFlow Data Production Pipeline for Flight Delay Prediction

## Project Overview
This project builds a data production pipeline using **MLFlow** to predict flight delays at **LAX**. The pipeline consists of data importation, filtering, cleaning, and applying a polynomial regression model to forecast delays. **MLFlow** is used to manage the workflow, track experiments, and record the performance of the model.

**Research Question:**  
How can we predict flight delays using a polynomial regression model, utilizing data from LAX airport?

## Methodology
- **MLFlow** manages the machine learning pipeline, from importing and cleaning data to training and evaluating the model.
- **Polynomial Regression** is used to predict flight delays based on various features such as scheduled departure and destination.
- **DVC** is utilized for version control of the data and model outputs.

## Key Findings
- The model predicted flight delays with a **mean squared error** of **115.74** and an **average delay** of **10.76 minutes**.
- **MLFlow** successfully tracked experiment parameters, metrics, and performance plots, allowing for a reproducible pipeline.

## Technologies Used
- Python
- MLFlow
- DVC

## Full Report
To explore the full analysis, including code and key takeaways, view the complete [Jupyter Notebook](./MLProject.ipynb).

You can also view the formal write up.
